import numpy as np
from metrics import r2_score

class SimpleLinearRegression1:
    def __init__(self):
        '''初始化Simple Linear Regression模型'''
        self.a_ = None
        self.b_ = None

    def fit(self,x_train,y_train):
        '''根据数据集x_train,y_train训练Simple Linear Regression模型'''
        # 设置断言，保证用户传入数据是合法的
        assert x_train.ndim == 1 ,\
            "Simple Linear Regressor can only solve single feature training data."
        assert len(x_train) == len(y_train) ,\
            "the size of x_train must be equal to the size of y_train"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        top = 0.0   #分子
        bottom = 0.0    #分母
        for x,y in zip(x_train,y_train):
            top += (x *(y - y_mean))
            bottom += (x * (x - x_mean))
        self.a_ = top / bottom
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self , x_predict):
        '''给定待预测数据集x_predict，返回表示x_predict的结果向量'''
        assert x_predict.ndim == 1 ,\
            "Simple Linear Regressor can only solve single feature training data."
        assert self.a_ is not None and self.b_ is not None ,\
            "must fit before predict!"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        '''给定单个待预测数据x_single,返回x_single的预测结果值'''
        return self.a_ * x_single + self.b_

    def __repr__(self): #字符串输出
        return "SimpleLinearRegression1()"

'''
向量化运算，减少运算时间（与for循环相比）
'''
class SimpleLinearRegression2:
    def __init__(self):
        '''初始化Simple Linear Regression模型'''
        self.a_ = None
        self.b_ = None

    def fit(self,x_train,y_train):
        '''根据数据集x_train,y_train训练Simple Linear Regression模型'''
        # 设置断言，保证用户传入数据是合法的
        assert x_train.ndim == 1 ,\
            "Simple Linear Regressor can only solve single feature training data."
        assert len(x_train) == len(y_train) ,\
            "the size of x_train must be equal to the size of y_train"

        x_mean = np.mean(x_train)   #求平均数
        y_mean = np.mean(y_train)

        top = 0.0   #分子
        bottom = 0.0    #分母
        #向量化计算
        top = x_train.dot(y_train - y_mean)
        bottom = x_train.dot(x_train - x_mean)

        self.a_ = top / bottom
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self , x_predict):
        '''给定待预测数据集x_predict，返回表示x_predict的结果向量'''
        assert x_predict.ndim == 1 ,\
            "Simple Linear Regressor can only solve single feature training data."
        assert self.a_ is not None and self.b_ is not None ,\
            "must fit before predict!"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        '''给定单个待预测数据x_single,返回x_single的预测结果值'''
        return self.a_ * x_single + self.b_

    def score(self,x_test,y_test):
        '''根据测试集x_test和y_test 确定当前模型的准确度'''
        y_predict = self.predict(x_test)
        return r2_score(y_test , y_predict)

    def __repr__(self): #字符串输出
        return "SimpleLinearRegression2()"
