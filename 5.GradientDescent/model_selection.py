import numpy as np


def train_test_split(X, y, test_ratio=0.2, seed=None):
    """将数据 X 和 y 按照test_ratio分割成X_train, X_test, y_train, y_test"""
    assert X.shape[0] == y.shape[0], \
        "the size of X must be equal to the size of y"
    assert 0.0 <= test_ratio <= 1.0, \
        "test_ration must be valid"

    if seed: #若希望两次随机产生的结果相同，则指定seed种子
        np.random.seed(seed)

    shuffled_indexes = np.random.permutation(len(X)) #进行索引乱序处理

    test_size = int(len(X) * test_ratio) #测试用例个数
    test_indexes = shuffled_indexes[:test_size] #测试数据集
    train_indexes = shuffled_indexes[test_size:] #训练数据集

    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, X_test, y_train, y_test