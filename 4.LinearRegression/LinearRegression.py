import numpy as np
from metrics import r2_score

class LinearRegression:
    def __init__(self):
        '''初始化LinearRegression模型'''
        self.coef_ = None    #序数
        self.interception_ = None   #截距
        self._theta = None

    def fit_normal(self , X_train , y_train):
        '''根据训练数据集X_train, y_train训练LinearRegression模型'''
        #shape()返回的是一维和二维长度
        assert X_train.shape[0] == y_train.shape[0],\
            "the size of X_train must be equal to the size of y_train"
        # 在X_train的左边插入(len(X_train),1)的一列，该列的值全为1
        X_b = np.hstack([np.ones((len(X_train), 1)),X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)#inv 求逆  T 转置  dot 点乘
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def predict(self,X_predict):
        '''给定待预测数据X_predict，返回表示X_predict的结果变量'''
        assert self.interception_ is not None and self.coef_ is not None,\
            "must fit before predict!"
        assert X_predict.shape[1] ==len(self.coef_),\
            "the feature number of X_predict must be equal to X_train"
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])  # 在X_predict的左边插入一列
        return X_b.dot(self._theta)

    def score(self , X_test , y_test):
        '''根据测试数据集X_test和y_test确定当前模型的准确度'''
        y_predict = self.predict(X_test)
        return r2_score(y_test,y_predict)

    def __repr__(self):
        return "LinearRegression()"