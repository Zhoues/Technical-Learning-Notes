import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

x = np.random.uniform(-3, 3, size=100)
# 调整为列向量
X = x.reshape(-1, 1)
y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100)

# 定义一个管道，该管道内部为多项式回归执行的步骤
poly_reg = Pipeline([
    ("poly", PolynomialFeatures(degree=2)),
    ("std_scaler", StandardScaler()),
    ("lin_reg", LinearRegression())
])

# 进行拟合
poly_reg.fit(X, y)

# 进行预测
y_predict = poly_reg.predict(X)

# 进行绘图
plt.scatter(x, y)
plt.plot(np.sort(x), y_predict[np.argsort(x)])
plt.show()
