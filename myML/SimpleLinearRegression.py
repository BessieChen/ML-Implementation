import numpy as np
from .matrics import r2_score

class mySimpleLinearRegression1:
    def __int__(self):
        """初始化"""
        self.a_ = None
        self.b_ = None

    def fit(self, x, y): #注意是小x，因为简单线性回归只有一个特征，所以小x是向量，不需要用大X矩阵
        assert x.ndim == 1, "Simple Linear Regression can only resolve single feature training data."
        #注意是np.array的是ndim，而不是dim
        assert x.shape[0] == y.shape[0], "Length of x and y must be the same."

        num = 0.0
        d = 0.0

        x_mean = np.mean(x)
        y_mean = np.mean(y)
        for x_i, y_i in zip(x, y):
            num += (x_i - x_mean) * (y_i - y_mean)
            d += (x_i - x_mean) ** 2

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        assert x_predict.ndim == 1, "Simple Linear Regression can only resolve single feature data."
        assert self.a_ is not None and self.b_ is not None, "Fit must be done before predict."

        return np.array([ self.a_ * x + self.b_ for x in x_predict])

    def _predict(self, x_single):
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return "Bessie's SimpleLinearRegression1()"


class mySimpleLinearRegression2:
    def __int__(self):
        """初始化"""
        self.a_ = None
        self.b_ = None

    def fit(self, x, y):  # 注意是小x，因为简单线性回归只有一个特征，所以小x是向量，不需要用大X矩阵
        assert x.ndim == 1, "Simple Linear Regression can only resolve single feature training data."
        # 注意是np.array的是ndim，而不是dim
        assert x.shape[0] == y.shape[0], "Length of x and y must be the same."


        x_mean = np.mean(x)
        y_mean = np.mean(y)

        # 向量化运算：
        num = (x - x_mean).dot(y - y_mean)
        d = (x - x_mean).dot(x - x_mean)

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        assert x_predict.ndim == 1, "Simple Linear Regression can only resolve single feature data."
        assert self.a_ is not None and self.b_ is not None, "Fit must be done before predict."

        return np.array([self.a_ * x + self.b_ for x in x_predict])

    def _predict(self, x_single):
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return "Bessie's SimpleLinearRegression2()"

class mySimpleLinearRegression: #same as mySimpleLinearRegression2
    def __int__(self):
        """初始化"""
        self.a_ = None
        self.b_ = None

    def fit(self, x, y):  # 注意是小x，因为简单线性回归只有一个特征，所以小x是向量，不需要用大X矩阵
        assert x.ndim == 1, "Simple Linear Regression can only resolve single feature training data."
        # 注意是np.array的是ndim，而不是dim
        assert x.shape[0] == y.shape[0], "Length of x and y must be the same."


        x_mean = np.mean(x)
        y_mean = np.mean(y)

        # 向量化运算：
        num = (x - x_mean).dot(y - y_mean)
        d = (x - x_mean).dot(x - x_mean)

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        assert x_predict.ndim == 1, "Simple Linear Regression can only resolve single feature data."
        assert self.a_ is not None and self.b_ is not None, "Fit must be done before predict."

        return np.array([self.a_ * x + self.b_ for x in x_predict])

    def _predict(self, x_single):
        return self.a_ * x_single + self.b_

    def score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "Bessie's SimpleLinearRegression()"

