from random import shuffle

import pytest
import numpy as np
from pathlib import Path

from streamz_reco.core import ISGDRecommender, _RedisRowMatrix
from streamz_reco.test.eval import evaluate_recall


@pytest.fixture()
def interactions():
    file = Path(__file__).parent.joinpath('assets', 'u.data')
    res = []
    with file.open('r') as fp:
        for line in fp.read().splitlines():
            data = line.split()
            if int(data[2]) >= 3:
                res.append((int(data[0]), int(data[1])))
    shuffle(res)
    return res[:70000], res[70000:]


def test_matrix_get_row_slice():
    data = np.identity(10)
    A = _RedisRowMatrix('test', data)
    res = A.get_row_slice()

    assert np.allclose(data, res)


def test_matrix_set_row():
    data = np.identity(10)
    A = _RedisRowMatrix('test_2', data)
    data[0, 3] = 1
    A.set_row(0, data[0])

    res = A.get_row_slice()
    assert np.allclose(data, res)


def test_matrix_set_row_slice():
    data = np.identity(10)
    A = _RedisRowMatrix('test_3', data)
    data[0, 3] = 1
    data[1, 4] = 1
    A.set_row_slice([0, 1], data[[0,1]])

    res = A.get_row_slice()
    assert np.allclose(data, res)


def test_model(interactions):
    train, test = interactions
    mdl = ISGDRecommender(20, 'test_mdl_2')
    for uid, iid in train:
        mdl.update(uid, iid)
    #mdl.update_batch(train, iter=1)

    recall = evaluate_recall(mdl, train, 25)

    assert recall > 0.02
