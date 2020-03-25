import numpy as np
from .matrics import accuracy_score #因为是解决的事分类问题

class myLogisticRegression:

    def __init__(self):
        self.coef_ = None

        self.interception_ = None
        self._theta = None
        #TODO: 私有变量：_theta

    # 没有正规方程解
    # def fit_normal(self, X_train, y_train):
    #     """训练数据集"""
    #     assert X_train.shape[0] == y_train.shape[0], "The sample number of X_train and y_train must be the same."
    #
    #     X_b = np.hstack([np.ones( (X_train.shape[0], 1) ), X_train])
    #     self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
    #     self.interception_ = self._theta[0]
    #     self.coef_ = self._theta[1:]
    #
    #     return self

    def _sigmoid(self, t):
        return 1./ (1. + np.exp(-t))

    def fit(self, X_train, y_train, eta = 0.01, n_iters = 1e4, epsilon = 1e-8):
        assert X_train.shape[0] == y_train.shape[0], "The sample number of X_train and y_train must be the same."

        def J(theta, X_b, y):  # 损失函数J，其中X_b是已经加了第一列是1
            try:
                y_hat = self._sigmoid(X_b.dot(theta))
                return - np.sum(y * np.log(y_hat) + (1 - y) * np.log(1-y_hat)) / len(y)
            except:
                return float('inf')

        def dJ(theta, X_b, y): #向量化
            return X_b.T.dot(self._sigmoid(X_b.dot(theta)) - y) / len(X_b)

        #不需要改变gradient_descent
        def gradient_descent(X_b, y, initial_theta, eta, n_iters, epsilon):
            theta = initial_theta
            i_iter = 0
            while i_iter <= n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta + (-1) * eta * gradient
                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break
                i_iter += 1

            return theta

        X_b = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])  # 矩阵

        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters, epsilon)
        self.coef_ = self._theta[1:]
        self.interception_ = self._theta[0]
        return self
    # def fit_sgd(self, X_train, y_train, n_iters = 5, t0 = 5, t1 = 50): #TODO：sgd(), n_iters的定义不是迭代次数，因为迭代次数与X_train大小有关，n_iters现在定义为X_train看几圈
    #     assert X_train.shape[0] == y_train.shape[0], "The sample number of X_train and y_train must be the same."
    #     assert n_iters >= 1, "We must iterate all sample for at least one time."
    #     def dJ_sgd(theta, X_b_i, y_i):  # 传入的不是X_b整个矩阵而是X_b的第i个样本，y也变成y_i
    #         return X_b_i.T.dot(X_b_i.dot(theta) - y_i) / len(X_b_i) * 2.
    #
    #     def sgd(X_b, y, initial_theta, n_iters, t0, t1):  # 不需要传入学习率eta，是里面自己计算的
    #
    #         def learning_rate(t):
    #             return t0 / (t + t1)
    #
    #         theta = initial_theta
    #         m = len(X_b)
    #         """为什么不需要下面这一段：这一段的break条件：1. i_iters 超过 n_iters 2. abs()差值足够小
    #         while i_iter <= n_iters:
    #             gradient = dJ(theta, X_b, y)
    #             last_theta = theta
    #             theta = theta + (-1) * eta * gradient
    #             if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon): #因为采用sgd，下降方向是随机的，即便abs(xx-yy)的值很小，也不代表到了最低值
    #                 break
    #             i_iter += 1
    #         """
    #         # 现在break条件： i_iters 超过 n_iters （所以用for loop）
    #         for cur_iter in range(n_iters):
    #             indexes = np.random.permutation(m) #TODO：将index顺序打乱：既保证了全部样本都用到，有保证样本的顺序随机
    #             X_b_new = X_b[indexes]
    #             y_new = y[indexes]
    #             #TODO：去除这一行，因为不能保证每个样本都被选中：rand_i = np.random.randint(m)
    #             for i in range(m):
    #                 gradient = dJ_sgd(theta, X_b_new[i], y_new[i])
    #                 theta = theta + (-1) * learning_rate(cur_iter * m + i) * gradient #TODO: 从learning_rate(cur_iter)改为learning_rate(cur_iter * m + i)，例如当cur_iter=0， i=0时候，是learning_rate(0)
    #
    #         return theta
    #
    #     X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
    #     initial_theta = np.zeros(X_b.shape[1])
    #     self._theta = sgd(X_b, y_train, initial_theta, n_iters, t0, t1)
    #     self.interception_ = self._theta[0]
    #     self.coef_ = self._theta[1:]
    #
    #     return self

    def predict_proba(self, X_predict):
        '''预测训练集X_predict, 返回X_predict的结果概率向量'''
        assert self.coef_ is not None and self.interception_ is not None, "must fit before predict."
        assert len(self.coef_) == X_predict.shape[1], "The feature number of X_train and X_test must be the same."

        X_b = np.hstack([np.ones((X_predict.shape[0], 1)), X_predict]) #X_predict.shape[0] 和 len(X_predict)都行
        return self._sigmoid(X_b.dot(self._theta))


    def predict(self, X_predict):
        '''测试训练集'''
        assert self.coef_ is not None and self.interception_ is not None, "must fit before predict."
        assert len(self.coef_) == X_predict.shape[1], "The feature number of X_train and X_test must be the same."

        prob = self.predict_proba(X_predict)
        return np.array(prob >= 0.5, dtype = 'int') #返回的从boolean变成int：1，0

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "Bessie: myLogisticRegression()."