import numpy as np
from metrics import r2_score


class LinearRegression:
    def __init__(self):
        '''初始化LinearRegression模型'''
        self.coef_ = None  # 序数
        self.interception_ = None  # 截距
        self._theta = None

    # 正规方程解
    def fit_normal(self, X_train, y_train):
        '''根据训练数据集X_train, y_train训练LinearRegression模型'''
        # shape()返回的是一维和二维长度
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        # 在X_train的左边插入(len(X_train),1)的一列，该列的值全为1
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)  # inv 求逆  T 转置  dot 点乘
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    # 批量梯度下降法
    def fit_gd(self, X_train, y_train, eta=0.01, n_iters=1e4):
        '''根据训练集X_train和y_train，使用梯度下降法训练LinearRegression模型'''
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        # 损失函数
        def J(theta, X_b, y):  # X_b为X添加‘1’那列后的数组
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(X_b)
            except:
                return float('inf')  # 返回float的最大值，表示损失函数达到了最大值

        # 损失函数的梯度
        def dJ(theta, X_b, y):
            # 该方法是for循环的方式
            # res = np.empty(len(theta))
            # res[0] = np.sum(X_b.dot(theta) - y)
            # for i in range(1, len(theta)):
            #     res[i] = (X_b.dot(theta) - y).dot(X_b[:, i])
            # return res * 2 / len(X_b)
            return X_b.T.dot(X_b.dot(theta) - y) * 2. / len(X_b)  # 这是向量化的方法，更快捷高效

        # 梯度下降法
        def gradient_descent(X_b, y, initial_theta, eta, n_iters, epsilon=1e-8):  # n_iters设置循环的次数，不让循环无限的执行
            theta = initial_theta
            i_iters = 0
            while i_iters < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if (abs(J(last_theta, X_b, y) - J(theta, X_b, y)) < epsilon):
                    break
                i_iters += 1
            return theta

        # 对X_train和theta数据的处理
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])  # n+1维的向量，数量为特征数+1
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)
        self.interception_ = self._theta[0]  # 截距
        self.coef_ = self._theta[1:]
        return self

    # 随机梯度下降法
    def fit_sgd(self, X_train, y_train, n_iters=5, t0=5, t1=50):  # n_iters对样本查询的遍数
        '''根据训练集X_train和y_train，使用随机梯度下降法训练LinearRegression模型'''
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        assert n_iters >= 1, "n_iters must be >= 1"

        # 损失函数的梯度
        def dJ_sgd(theta, X_b_i, y_i):
            return X_b_i.T.dot(X_b_i.dot(theta) - y_i) * 2.

        # 梯度下降法
        def sgd(X_b, y, initial_theta, n_iters=5, t0=5, t1=50):
            def learning_rate(t):  # 计算学习率（步长）
                return t0 / (t + t1)

            theta = initial_theta
            m = len(X_b)
            for cur_iter in range(n_iters):  # 设置数据查询的次数为总数据的5倍
                indexes = np.random.permutation(m)  # 对数组乱序
                X_b_permutation = X_b[indexes]
                y_permutation = y[indexes]
                for i in range(m):
                    gradient = dJ_sgd(theta, X_b_permutation[i], y_permutation[i])
                    theta = theta - learning_rate(cur_iter * m + i) * gradient
            return theta

        # 对X_train和theta数据的处理
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])  # n+1维的向量，数量为特征数+1
        self._theta = sgd(X_b, y_train, initial_theta, n_iters, t0, t1)
        self.interception_ = self._theta[0]  # 截距
        self.coef_ = self._theta[1:]
        return self

    def predict(self, X_predict):
        '''给定待预测数据X_predict，返回表示X_predict的结果变量'''
        assert self.interception_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])  # 在X_predict的左边插入一列
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        '''根据测试数据集X_test和y_test确定当前模型的准确度'''
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"
