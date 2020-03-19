import numpy as np
from math import sqrt
from collections import Counter

def kNN_classify(k, X_train, y_train, X):
    assert 1 <= k <= X_train.shape[0], "k must be valid"
    assert X_train.shape[0] == y_train.shape[0], "X_train and y_train doesn't have same size of sample"
    assert X_train.shape[1] == X.shape[0], "the number of feature of X must equals to that of X_train"

    distances = [sqrt(np.sum((x_train - X)**2)) for x_train in X_train]
    topKIndex = np.argsort(distances)
    # 这个也可以，只不过返回的不是python自带list，而是np的list：topY = y_train[topKIndex[:k]]
    topY = [y_train[i] for i in topKIndex[:k]]
    votes = Counter(topY)
    return votes.most_common(1)[0][0]

def cal():
    #s = "2 2 2 2 1 2 1 1 2 2 3 2 2 2 2" #28 = 1.4
    #s = "1 1 1 1 1 2" #7 = 0.35
    #s = "1 1 1 1 2 1 1 1" #9 = 0.45
    #s = "2 1 1 1 1 1" #7 = 0.35
    s = "1 1 1 2 1" #6 = 0.3
    v = "+".join(s.split(" "))
    #print(v)
    #print(eval(v))
