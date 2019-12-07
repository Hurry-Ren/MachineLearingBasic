import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, x):
        '''根据训练数据集x获得数据的均值和方差'''
        assert x.ndim == 2, "The dimension of x must be 2"  # 在这里只处理二维数据
        self.mean_ = np.array([np.mean(x[:, i]) for i in range(x.shape[1])])
        self.scale_ = np.array([np.std(x[:, j]) for j in range(x.shape[1])])
        return self

    def tranform(self, x):
        '''将x根据这个StandardScaler进行均值方差归一化'''
        assert x.ndim == 2, "The dimension of x must be 2"  # 在这里只处理二维数据
        assert self.scale_ is not None and self.mean_ is not None, \
            "must fit before tranform！"
        assert x.shape[1] == len(self.mean_), \
            "the feature number of x must be equal to the mean_ and std_"  # 保证x列的数目和平均值的数目相等

        resX = np.empty(shape=x.shape, dtype=float)
        for col in range(x.shape[1]):
            resX[:, col] = (x[:, col] - self.mean_[col]) / self.scale_[col]
        return resX
