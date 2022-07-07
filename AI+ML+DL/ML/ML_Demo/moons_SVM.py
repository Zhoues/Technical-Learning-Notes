from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import LinearSVC

X, y = make_moons(noise=0.15, random_state=666)

# 定义一个管道，该管道内部为SVM执行的步骤
poly_reg = Pipeline([
    # 生成多项式
    ("poly", PolynomialFeatures(degree=20)),
    # 标准化
    ("std_scaler", StandardScaler()),
    # 设置模型正则化的参数
    ("lin_reg", LinearSVC(C=1))
])

poly_reg.fit(X, y)

plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()
