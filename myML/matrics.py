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