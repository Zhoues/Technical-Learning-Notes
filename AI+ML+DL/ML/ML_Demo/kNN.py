import numpy as np
from math import sqrt
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier


# 自己实现的
def my_kNN_classifier(k, X_train, y_train, x):
    assert 1 <= k <= X_train.shape[0], "k must be valid"
    assert X_train.shape[0] == y_train.shape[0], \
        "the size of X_train must equal to the size of y_train"
    assert X_train.shape[1] == x.shape[0], \
        "the feature number of x must be equal to X_train"

    # 计算出 x 向量距离 X_train 中各个数据点的距离
    distances = [sqrt(np.sum((x - x_train) ** 2)) for x_train in X_train]
    # 对距离进行排序并得到其索引位置
    nearest = np.argsort(distances)
    # 找到 k 近邻的种类
    topK_y = [y_train[i] for i in nearest[:k]]
    # 进行种类统计
    votes = Counter(topK_y)
    # 返回统计数最多的一个种类
    return votes.most_common(1)[0][0]


# sklearn自带的
def sklearn_kNN_classifier(k, X_train, y_train, x):
    assert 1 <= k <= X_train.shape[0], "k must be valid"
    assert X_train.shape[0] == y_train.shape[0], \
        "the size of X_train must equal to the size of y_train"
    assert X_train.shape[1] == x.shape[1], \
        "the feature number of x must be equal to X_train"
    # 创建一个 kNN算法的分类器
    kNN_classifier = KNeighborsClassifier(n_neighbors=k)
    # 开始进行拟合
    kNN_classifier.fit(X_train, y_train)
    # # 由于 sklearn 中对于需要预测的向量为二维，所有需要升维
    # x_predict = x.reshape(1, -1)
    return kNN_classifier.predict(x)
