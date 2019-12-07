import numpy as np
from math import sqrt

def accuracy_score(y_test, y_predict):
    """计算y_true和y_predict之间的准确率"""
    assert y_test.shape[0] == y_predict.shape[0], \
        "the size of y_true must be equal to the size of y_predict"
    return sum(y_test == y_predict) / len(y_test)

# MSE
def mean_squared_error(y_predict, y_test):
    '''计算y_predict和y_test之间的MSE'''
    assert len(y_predict) == len(y_test), \
        "the size of y_predict must be equal to the size of y_test"
    return np.sum((y_predict - y_test) ** 2) / len(y_test)

# RMSE
def root_mean_squared_error(y_predict, y_test):
    '''计算y_predict和y_test之间的RMSE'''
    return sqrt(mean_squared_error(y_predict, y_test))

# MAE
def mean_absolute_error(y_predict, y_test):
    ''''计算y_predict和y_test之间的RMSE'''
    assert len(y_predict) == len(y_test), \
        "the size of y_predict must be equal to the size of y_test"
    return np.sum(np.absolute(y_predict - y_test)) / len(y_test)

# R2 Square
def r2_score(y_test, y_predict):
    '''计算y_test和y_predict之间的R Squared'''
    return 1 - mean_squared_error(y_predict, y_test) / np.var(y_test)

#TN
def TN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 0))

#FP
def FP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 1))

#FN
def FN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 0))

#TP
def TP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 1))

#混淆矩阵
def confusion_matrix(y_true, y_predict):
    return np.array([
        [TN(y_true, y_predict), FP(y_true, y_predict)],
        [FN(y_true, y_predict), TP(y_true, y_predict)]
    ])

#精确率
def precison_score(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fp = FP(y_true, y_predict)
    try:
        return tp / (tp + fp)
    except:
        return 0.

#召回率
def recall_score(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fn = FN(y_true, y_predict)
    try:
        return tp / (tp + fn)
    except:
        return 0.

#F1 Score
def f1_score(y_true, y_predict):
    precision = precison_score(y_true, y_predict)
    recall = recall_score(y_true, y_predict)
    try:
        return (2 * precision * recall) / (precision + recall)
    except:
        return 0.

#TPR
def TPR(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fn = FN(y_true, y_predict)
    try:
        return tp / (tp + fn)
    except:
        return 0.

#FPR
def FPR(y_true, y_predict):
    fp = FP(y_true, y_predict)
    tn = TN(y_true, y_predict)
    try:
        return fp / (fp + tn)
    except:
        return 0.
