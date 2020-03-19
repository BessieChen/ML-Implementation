import numpy as np
from .matrics import r2_score

class myLinearRegression:

    def __init__(self):
        self.coef_ = None
        self.interception_ = None
        self._theta = None
        #TODO: 私有变量：_theta

    def fit_normal(self, X_train, y_train):
        """训练数据集"""
        assert X_train.shape[0] == y_train.shape[0], "The sample number of X_train and y_train must be the same."

        X_b = np.hstack([np.ones( (X_train.shape[0], 1) ), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict(self, X_predict):
        '''测试训练集'''
        assert self.coef_ is not None and self.interception_ is not None, "must fit before predict."
        assert len(self.coef_) == X_predict.shape[1], "The feature number of X_train and X_test must be the same."

        X_b = np.hstack([np.ones((X_predict.shape[0], 1)), X_predict]) #X_predict.shape[0] 和 len(X_predict)都行
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "Bessie: myLinearRegression."