from itertools import groupby

import numpy as np


def evaluate_recall(mdl, test, at=10):
    recalls = []
    predictions = np.argpartition(-mdl.all_scores(), kth=at)[:, :10]
    for k, g in groupby(test, key=lambda x: x[0]):
        correct = set(map(lambda x: mdl.known_items.get(x[1], -1), list(g)[:at]))
        preds = set(predictions[mdl.known_users[k]])
        recalls.append(calculate_recall(correct, preds))

    return np.asarray(recalls).mean()


def calculate_recall(correct, predictions):
    size = min(len(correct), len(predictions))
    recall = len(correct & predictions) / size
    return recall