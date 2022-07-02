from math import sqrt

import numpy as np
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# 加载数据
from my_sklearn.SimpleLinearRegression import SimpleLinearRegression

boston = load_boston()

# 暂时只使用波士顿房价数据的 RM(房间的数量) 这一列
x = boston.data[:, 5]
y = boston.target

# 去除不确定的点
max_y = np.max(y)
x = x[y < max_y]
y = y[y < max_y]

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y)

# 创建简单线性回归模型
reg = SimpleLinearRegression()

# 进行拟合(利用最小二乘法计算a,b)
reg.fit(x_train, y_train)

# 得到预测结果
y_predict = reg.predict(x_test)

# 绘制拟合图像
# plt.scatter(x_train, y_train)
# plt.plot(x_train, reg.predict(x_train), color="red")
# plt.show()

# R Square
R_Square_test = r2_score(y_test, y_predict)
print(R_Square_test)
