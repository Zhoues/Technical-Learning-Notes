import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x = np.random.uniform(-3, 3, size=100)
# 调整为列向量
X = x.reshape(-1, 1)
y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100)

# 进行多项式升维处理,degree表示升至的维度
poly = PolynomialFeatures(degree=2)
# 拟合
poly.fit(X)
# 调整至指定维数
X2 = poly.transform(X)

# 采用线性回归
reg = LinearRegression()

reg.fit(X2, y)

print(reg.coef_)
