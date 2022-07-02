import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# 加载数据
iris = load_iris()

# 获取样本矩阵和标签向量
X = iris.data
y = iris.target

# 进行训练集和测试集的分离
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 创建一个 kNN算法的分类器
kNN_classifier = KNeighborsClassifier(n_neighbors=3)
# 开始进行拟合
kNN_classifier.fit(X_train, y_train)
# 进行预测
y_predict = kNN_classifier.predict(X_test)
# 计算准确率
shot_ratio = accuracy_score(y_test, y_predict)

print(shot_ratio)
print(y_predict)
print(y_test)
