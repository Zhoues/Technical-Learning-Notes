from math import sqrt

import numpy as np
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

# 加载数据
boston = load_boston()

# 暂时只使用波士顿房价数据
X = boston.data
y = boston.target

# 去除不确定的点
max_y = np.max(y)
X = X[y < max_y]
y = y[y < max_y]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

# 创建kNN线性回归模型
knn_reg = KNeighborsRegressor()

# 进行拟合(利用最小二乘法计算a,b)
knn_reg.fit(X_train, y_train)

print(knn_reg.score(X_test, y_test))
