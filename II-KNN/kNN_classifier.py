import numpy as np
from math import sqrt
from collections import Counter
from matrics import accuracy_score

class myKNNClassifier:

    def __init__(self, k):
        '''初始化knn分类器'''
        assert k >= 1, "k must be valid"
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        '''根据训练数据集X_train 和 y_train，训练分类器'''

        assert X_train.shape[0] == y_train.shape[0], \
            "the size of sample of X_train must be equal to y_train"
        assert self.k <= X_train.shape[0],\
            "the size of X_train must be at least k"
        self._X_train = X_train
        self._y_train = y_train
        return self #需要return self，因为sklearn也是这么做的，如此的话，我们可以和sklearn无缝连接。 TODO: 返回的是__repr__函数

    def predict(self, X_predict):
        assert self._X_train is not None and self._y_train is not None, \
            "must fit before predict!"
        assert self._X_train.shape[1] == X_predict.shape[1],\
            "the feature number of X_predict must be equal to X_train"

        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict) #TODO: 注意这里需要返回的是类型numpy下的array，而不是python内置array

    def _predict(self, x_predict):
        assert x_predict.shape[0] == self._X_train.shape[1],\
            "the feature number of x must be equal to X_train"
        distances = [sqrt(np.sum((x_train - x_predict) ** 2)) for x_train in self._X_train]
        nearest = np.argsort(distances)[:self.k]
        topY = [self._y_train[i] for i in nearest]
        votes = Counter(topY)
        return votes.most_common(1)[0][0]

    def __repr__(self):
        return "Bessie: from __repr__: KNN(k = %d)" % self.k

    def score(self, X_test, y_test):
        '''根据测试数据集X_test， y_test来判断模型accuracy'''
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)


