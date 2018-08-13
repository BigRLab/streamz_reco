from random import shuffle
from uuid import uuid4

import numpy as np
import fakeredis

CONN = fakeredis.FakeStrictRedis()


def get_redis_connection():
    return CONN


def _matrix_hset_key(model_id, name):
    return '{}:{}_matrix'.format(model_id, name).encode()


class _RedisRowMatrix:

    def __init__(self, key, mat):
        conn = get_redis_connection()
        self.key = key
        self.shape = mat.shape
        self._from_matrix(conn, mat)

    def set_row(self, idx, data, conn=None, pipe=False):
        if conn is None:
            conn = get_redis_connection()
        bytes_arr = data.astype(np.float64).tobytes()
        new = conn.hset(self.key, str(idx).encode(),
                           bytes_arr)
        if new and not pipe:
            self.shape = (self.shape[0] + 1, self.shape[1])
            conn.hincrby(self.key, 'nrows')

    def get_row(self, idx, conn=None):
        if conn is None:
            conn = get_redis_connection()
        bytes_arr = conn.hget(self.key,
                              str(idx).encode())
        if bytes_arr is None:
            raise KeyError('This vector does not exist')
        return np.frombuffer(bytes_arr, dtype=np.float64)

    def _from_matrix(self, conn, mat):
        with conn.pipeline() as pipe:
            pipe.hset(self.key, 'nrows', str(mat.shape[0]))
            pipe.hset(self.key, 'ncols', str(mat.shape[1]))
            pipe.hset(self.key, 'dtype', str(mat.dtype))
            for idx in range(mat.shape[0]):
                pipe.hset(self.key, str(idx).encode(), mat[idx].tobytes())
            pipe.execute()

    def get_row_slice(self, idx=None, conn=None):
        if conn is None:
            conn = get_redis_connection()
        if idx is None:
            data = conn.hgetall(self.key)
        else:
            pipe = conn.pipeline()
            for j in idx:
                pipe.hget(self.key, str(j).encode())
            data = dict(zip(idx, pipe.execute()))
        nrows = max(len(data) - 3, 0)
        mat = np.empty((nrows, self.shape[1]))
        for idx, bytes_ in data.items():
            try:
                i = int(idx)
            except ValueError:
                continue
            mat[i] = np.frombuffer(bytes_)
        return mat

    def set_row_slice(self, idx, arr, conn=None):
        if conn is None:
            conn = get_redis_connection()
        assert len(idx) == arr.shape[0]
        with conn.pipeline() as pipe:
            for i, k in enumerate(idx):
                self.set_row(k, arr[i], pipe, True)
            increase = max(idx) - self.shape[0]
            if increase > 0:
                self.shape = (self.shape[0] + increase, self.shape[1])
                pipe.hset(self.key, 'nrows', self.shape[0])
            pipe.execute()


class ISGDRecommender:

    def __init__(self, k, id_=None, l2_reg=0.01, learn_rate=0.05):
        if id_ is None:
            self.id = str(uuid4())
        else:
            self.id = id_
        self.k = k
        self.l2_reg = l2_reg
        self.learn_rate = learn_rate

        self.known_users = {}
        self.known_items = {}

        self.A = _RedisRowMatrix(self.user_key, np.empty((0, k)))
        self.B = _RedisRowMatrix(self.item_key, np.empty((0, k)))

    @property
    def user_key(self):
        return _matrix_hset_key(self.id, 'user')

    @property
    def item_key(self):
        return _matrix_hset_key(self.id, 'item')

    def get_user_vector(self, uid):
        try:
            return self.A.get_row(uid)
        except KeyError:
            return np.random.normal(0, 0.1, self.k)

    def get_item_vector(self, iid):
        try:
            return self.B.get_row(iid)
        except KeyError:
            return np.random.normal(0, 0.1, self.k)

    def update(self, uid, iid):
        if uid not in self.known_users:
            self.known_users[uid] = len(self.known_users)
        if iid not in self.known_items:
            self.known_items[iid] = len(self.known_items)

        u_index = self.known_users[uid]
        i_index = self.known_items[iid]

        u_vec = self.get_user_vector(u_index)
        i_vec = self.get_item_vector(i_index)

        new_ivec, new_uvec = self._update_vectors(i_vec, u_vec)

        self._update_matrices(i_index, new_ivec, new_uvec, u_index)

    def _update_vectors(self, i_vec, u_vec):
        err = 1. - np.inner(u_vec, i_vec)
        new_uvec = u_vec + self.learn_rate * \
                           (err * i_vec - self.l2_reg * u_vec)
        new_ivec = i_vec + self.learn_rate * \
                           (err * u_vec - self.l2_reg * i_vec)
        return new_ivec, new_uvec

    def update_batch(self, interactions, iter=1):
        users, items = map(set, zip(*interactions))
        known_users = set(map(self.get_uidx, set(self.known_users.keys()) & users))
        unkown_users = list(users - known_users)

        for u in unkown_users:
            self.known_users[u] = len(self.known_users)

        known_items = set(map(self.get_iidx, set(self.known_items.keys()) & items))
        unkown_items = list(items - known_items)

        for i in unkown_items:
            self.known_items[i] = len(self.known_items)

        unkown_users = list(map(self.get_uidx, unkown_users))
        unkown_items = list(map(self.get_iidx, unkown_items))
        known_users = list(known_users)
        known_items = list(known_items)

        A = self.A.get_row_slice(known_users)
        B = self.B.get_row_slice(known_items)

        A = np.append(A, np.random.normal(0, 0.1, (len(unkown_users), self.k)), axis=0)
        B = np.append(B, np.random.normal(0, 0.1, (len(unkown_items), self.k)), axis=0)

        self._update_batch(A, B, interactions, iter)

        uidx = known_users + unkown_users
        iidx = known_items + unkown_items

        self.A.set_row_slice(uidx, A)
        self.B.set_row_slice(iidx, B)

    def _update_batch(self, A, B, interactions, iter=1):
        for i in range(iter):
            shuffle(interactions)
            for uid, iid in interactions:
                u_idx = self.get_uidx(uid)
                i_idx = self.get_iidx(iid)

                A[u_idx], B[i_idx] = self._update_vectors(A[u_idx], B[i_idx])

    def get_uidx(self, uid):
        return self.known_users.get(uid, -1)

    def get_iidx(self, uid):
        return self.known_items.get(uid, -1)

    def _update_matrices(self, i_index, new_ivec, new_uvec, u_index):
        self.A.set_row(u_index, new_uvec)
        self.B.set_row(i_index, new_ivec)

    def _get_item_matrix(self):
        return self.A.get_row_slice()

    def _get_user_matrix(self):
        return self.B.get_row_slice()

    def all_scores(self):
        A = self._get_user_matrix()
        B = self._get_item_matrix()
        return np.dot(A, B.T)

    def recommend(self, uid):
        """
        Recommend Top-N items for the user u
        """

        if uid not in self.known_users:
            raise KeyError('The user is not known.')

        uvec = self.get_user_vector(uid)
        bmat = self._get_item_matrix()
        scores = np.abs(1. - np.dot(np.array([uvec]), bmat.T)).flatten()

        return scores
