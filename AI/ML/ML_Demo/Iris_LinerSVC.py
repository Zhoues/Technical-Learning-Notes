import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# 加载数据

iris = load_iris()

# 获取样本矩阵和标签向量
X = iris.data
y = iris.target

X = X[y < 2, :2]
y = y[y < 2]

standardScaler = StandardScaler()
standardScaler.fit(X)
X_std = standardScaler.transform(X)

# 引入线性 SVM 分类器 LinearSVC (C表示正则化参数)
svc = LinearSVC(C=1e9)
svc.fit(X_std, y)
