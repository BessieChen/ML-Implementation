import numpy as np

class myPCA:
    def __init__(self, n_components):
        assert n_components >= 1, "n_components is invalid"
        self.n_components = n_components
        self.components_ = None #通过用户传来的数据，后面函数自己加工，所以命名方式为self.xx_

    def fit(self, X, eta = 0.01, n_iters = 1e4):
        '''获得数据集X的前n个部分'''
        assert self.n_components <= X.shape[1], "n_components must not be greater than the feature number of X."

        def demean(X):
            return X - np.mean(X, axis=0)  # 需要减去的每一列的均值

        def f(w, X):
            return np.sum((X.dot(w) ** 2)) / len(X)

        def df(w, X):
            return X.T.dot(X.dot(w)) * 2. / len(X)

        def direction(w):
            return w / np.linalg.norm(w)  # 求模：np.linalg.norm()

        def first_component(X, initial_w, eta, n_iters=1e4, epsilon=1e-8):  # 就是之前的gradient_descent()
            w = direction(initial_w)
            cur_iter = 0
            while cur_iter <= n_iters:
                gradient = df(w, X)
                last_w = w
                w = w + (+1) * eta * gradient  # 注意w必须是单位向量，所以我们每一步循环需要保证w还是单位向量
                w = direction(w)  # TODO: 为什么不在gradient = df(w, X) 这一行来进行direction()
                if (abs(f(w, X) - f(last_w, X)) < epsilon):
                    break
                cur_iter += 1

            return w

        X_pca = demean(X)
        self.components_ = np.empty([self.n_components, X.shape[1]])
        for i in range(self.n_components):
            initial_w = np.random.random(X_pca.shape[1])
            w = first_component(X_pca, initial_w, eta)
            self.components_[i, :] = w #之前是 res.append(w)， 因为初始res = [], X.shape[1]==1
            X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w

        return self

    def transform(self, X):
        '''将给定的X，映射到各个主成分分量中'''
        assert X.shape[1] == self.components_.shape[1]

        return X.dot(self.components_.T)

    def inverse_transform(self, X):
        '''将给定的X，反向映射回原来的特征空间'''
        assert X.shape[1] == self.components_.shape[0]
        return X.dot(self.components_)


    def __repr__(self):
        return "PCA(n_components = %d)" % self.n_components
