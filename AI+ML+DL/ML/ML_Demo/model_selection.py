import numpy as np
from sklearn.model_selection import train_test_split


# 进行训练集和测试集的分离
# X 样本集, y 标签集, test_ratio分割率, seed 随机化种子
def my_train_test_split(X, y, test_ratio=0.2, seed=None):
    assert X.shape[0] == y.shape[0], \
        "the size of X must be equal to the size of y"
    assert 0.0 <= test_ratio <= 1.0, \
        "test_ration must be valid"
    if seed:
        np.random.seed(seed)
    # 进行索引随机化，permutation(x)可以获得 0~x 的随机排列
    shuffle_indexes = np.random.permutation(len(X))

    # 设置测试数据据的比例和大小
    test_size = int(len(X) * test_ratio)

    # 获得测试数据集和训练数据集的索引
    test_indexes = shuffle_indexes[:test_size]
    train_indexes = shuffle_indexes[test_size:]

    # 利用 Fancy Indexing 快速构建测试集和训练集
    X_train = X[train_indexes]
    y_train = y[train_indexes]
    # 利用 Fancy Indexing 快速构建测试集和训练集
    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, y_train, X_test, y_test
