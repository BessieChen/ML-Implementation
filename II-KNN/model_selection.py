import numpy as np
import sys
sys.path.append('..')

def my_train_test_split(X, y, test_ratio = 0.2, seed = None):
    assert X.shape[0] == y.shape[0],\
        "The number of sample of X and y must be the same."
    assert 0.0 <= test_ratio <= 1.0, \
        "test_ratio must be valid"

    if seed:
        np.random.seed(seed) #TODO: debug的时候，若指定一个seed，我们能确保每次产生的随机数都是相同的。

    shuffled_indexes = np.random.permutation(len(X))
    test_size = int(len(X) * test_ratio)
    train_indexes = shuffled_indexes[test_size:]
    test_indexes = shuffled_indexes[:test_size]
    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, X_test, y_train, y_test