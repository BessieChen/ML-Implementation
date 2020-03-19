import numpy as np

class myStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        '''根据训练数据集 X 获得数据的 mean 和 variance'''
        #assert X.ndim == 2, "The dimension of X must be 2" #一定是二维，但是这个二维的shape随意

        self.mean_ = [np.mean(X[:,i]) for i in range(X.shape[1])]
        self.scale_ = [np.std(X[:, i]) for i in range(X.shape[1])]
        return self

    def transform(self, X):
        '''将X 根据这个StandardSclae进行 均值方差归一化 处理'''
        #assert X.ndim == 2, "The dimension of X must be 2"
        assert self.mean_ is not None and self.std_ is not None, "must fit before transform"

        assert X.shape[1] == len(self.mean_), "The feature number of X must be equal to length of mean"

        res = np.empty(X.shape, dtype = 'float') #即便用户传过来的X是int型，我们也不用担心结果为int了
        for col in range(X.shape[1]): #我们需要是一列一列的填充
            res[:, col] = (X[:, col] - self.mean_[col]) / self.scale_[col]

        return res

if __name__ == "__main__":
    my = myStandardScaler()
    my.fit(np.random.normal(size = (100,2)))