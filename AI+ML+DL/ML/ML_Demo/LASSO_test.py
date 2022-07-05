import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

x = np.random.uniform(-3, 3, size=100)
# 调整为列向量
X = x.reshape(-1, 1)
y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100)

# 定义一个岭回归
Lasso_reg = Pipeline([
    ("poly", PolynomialFeatures(degree=20)),
    ("std_scaler", StandardScaler()),
    ("Lasso_reg", Lasso(alpha=0.01))
])

# 进行拟合
Lasso_reg.fit(X, y)

# 进行预测
y_predict = Lasso_reg.predict(X)

# 查看方差
print(mean_squared_error(y, y_predict))

# 进行绘图
plt.scatter(x, y)
plt.plot(np.sort(x), y_predict[np.argsort(x)])
plt.show()
