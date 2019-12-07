import numpy as np
from math import sqrt

def accuracy_score(y_test, y_predict):
    '''计算y_true和y_predict之间的准确率'''
    assert y_test.shape[0] == y_predict.shape[0], \
        "the size of y_true must be equal to the size of y_predict"
    return sum(y_test == y_predict) / len(y_test)

#MSE 均方误差
def mean_squared_error(y_predict , y_test ):
    '''计算y_predict和y_test之间的MSE'''
    assert len(y_predict) == len(y_test),\
        "the size of y_predict must be equal to the size of y_test"
    return np.sum((y_predict - y_test)**2) /len(y_test)

#RMSE 均方根误差
def root_mean_squared_error(y_predict , y_test ):
    '''计算y_predict和y_test之间的RMSE'''
    return sqrt(mean_squared_error(y_predict , y_test ))

#MAE 平均绝对值误差
def mean_absolute_error(y_predict , y_test ):
    ''''计算y_predict和y_test之间的RMSE'''
    assert len(y_predict) == len(y_test), \
        "the size of y_predict must be equal to the size of y_test"
    return np.sum(np.absolute(y_predict - y_test )) / len(y_test)

#R2 Square R-方
def r2_score(y_test , y_predict):
    '''计算y_test和y_predict之间的R Squared'''
    return 1 - mean_squared_error(y_predict , y_test) / np.var(y_test)
