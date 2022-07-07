import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

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

# 定义一个管道，该管道内部为SVM执行的步骤
poly_reg = Pipeline([
    # 标准化
    ("std_scaler", StandardScaler()),
    # 设置模型正则化的参数
    ("lin_reg", LinearSVR(epsilon=0.1))
])

poly_reg.fit(X_train, y_train)
print(poly_reg.score(X_test, y_test))
