import numpy as np
from math import sqrt

def accuracy_score(y_true, y_predict):
    assert y_true.shape[0] == y_predict.shape[0],\
        "the sample number of y_true must be equal to y_predict"
    return np.sum(y_true == y_predict) / len(y_true)

def mean_squared_error(y_true, y_predict):
    assert len(y_true) == len(y_predict), "The length of y_true and y_predict must be the same."
    return (1/len(y_true)) * np.sum((y_predict - y_true) ** 2)

def root_mean_squared_error(y_true, y_predict):
    return sqrt(mean_squared_error(y_true, y_predict))

def mean_absolute_error(y_true, y_predict):
    return (1/len(y_true)) * np.sum(abs(y_predict - y_true))

def r2_score(y_true, y_predict):
    assert len(y_true) == len(y_predict), "The length of y_true and y_predict must be the same."
    return 1 - mean_squared_error(y_true, y_predict) / np.var(y_true)

def TN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 0))

def TP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 1))

def FP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 1))

def FN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 0))

def confusion_matrix(y_true, y_predict):
    tn = TN(y_true, y_predict)
    fp = FP(y_true, y_predict)
    fn = FN(y_true, y_predict)
    tp = TP(y_true, y_predict)
    return np.array([
        [tn, fp],
        [fn, tp]
    ])

def precision_score(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    tp = TP(y_true, y_predict)
    fp = FP(y_true, y_predict)
    try:
        return tp / (tp + fp)
    except:
        return 0.0

def recall_score(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    tp = TP(y_true, y_predict)
    fn = FN(y_true, y_predict)
    try:
        return tp / (tp + fn)
    except:
        return 0.0

def f1_score(y_true, y_predict):
    precision = precision_score(y_true, y_predict)
    recall = recall_score(y_true, y_predict)
    try:
        return 2 * precision * recall / (precision + recall)
    except:
        return 0.0

def TPR(y_true, y_predict):
    return recall_score(y_true, y_predict)

def FPR(y_true, y_predict):
    tn = TN(y_true, y_predict)
    fp = FP(y_true, y_predict)
    try:
        return fp / (tn + fp)
    except:
        return 0.0