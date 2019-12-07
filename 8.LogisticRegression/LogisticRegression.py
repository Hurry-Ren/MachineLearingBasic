import numpy as np
from metrics import accuracy_score


class LogisticRegression:
    def __init__(self):
        '''初始化Logistic Regression模型'''
        self.coef_ = None  # 序数
        self.interception_ = None  # 截距
        self._theta = None

    def _sigmoid(self, t):
        return 1. / (1. + np.exp(-t))

    # 批量梯度下降法
    def fit(self, X_train, y_train, eta=0.01, n_iters=1e4):
        '''根据训练集X_train和y_train，使用梯度下降法训练Logistic Regression模型'''
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        # 损失函数
        def J(theta, X_b, y):  # X_b为X添加‘1’那列后的数组
            y_hat = self._sigmoid(X_b.dot(theta))
            try:
                return - np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / len(y)
            except:
                return float('inf')  # 返回float的最大值，表示损失函数达到了最大值

        # 损失函数的梯度
        def dJ(theta, X_b, y):
            return X_b.T.dot(self._sigmoid(X_b.dot(theta)) - y) / len(X_b)  # 这是向量化的方法，更快捷高效

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

    def predict_proba(self, X_predict):
        '''给定待预测数据X_predict，返回表示X_predict的结果概率变量'''
        assert self.interception_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])  # 在X_predict的左边插入一列
        return self._sigmoid(X_b.dot(self._theta))

    def predict(self, X_predict):
        '''给定待预测数据X_predict，返回表示X_predict的结果变量'''
        assert self.interception_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"
        proba = self.predict_proba(X_predict)
        return np.array(proba >= 0.5, dtype='int')

    def score(self, X_test, y_test):
        '''根据测试数据集X_test和y_test确定当前模型的准确度'''
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "LogisticRegression()"
