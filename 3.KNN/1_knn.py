import numpy as np
from math import sqrt
from collections import Counter

def kNN_classify(k, X_train, y_train, x): #k 为邻近的多少个点；x 为将要预测的点

    assert 1 <= k <= X_train.shape[0],"k must be valid"
    assert X_train.shape[0] == y_train.shape[0],\
        "the size of X_train must equal to the size of y_train"
    assert X_train.shape[1] == x.shape[0],\
        "the feature number of x must be equal to X_train"
    #计算距离，返回距离从小到大的索引
    distances = [sqrt(np.sum((x_train - x) **2)) for x_train in X_train]
    nearest = np.argsort(distances)
    #取最接近的k个值，然后统计这k个值的标记的数目
    topK_y = [y_train[i] for i in nearest[:k]]
    votes = Counter(topK_y)

    return votes.most_common(1)[0][0] #返回最接近的那个点的标记
