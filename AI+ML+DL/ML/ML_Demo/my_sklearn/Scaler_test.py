import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载数据
iris = load_iris()

# 获取样本矩阵和标签向量
X = iris.data
y = iris.target

# 进行训练集和测试集的分离
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 预处理：进行均值方差归一化
# 创建 Scaler 对象
standardScaler = StandardScaler()
# 进行拟合(传递值)
standardScaler.fit(X_train)

# 查看各个维度的均值
print(standardScaler.mean_)

# 查看各个维度的方差
print(standardScaler.scale_)

# 进行归一化处理训练集
X_train = standardScaler.transform(X_train)

# 利用训练集的均值方差归一化的Scaler对测试集进行归一化
X_test_standard = standardScaler.transform(X_test)

# 创建一个 kNN算法的分类器
kNN_classifier = KNeighborsClassifier()
# 开始进行拟合
kNN_classifier.fit(X_train, y_train)

# 计算准确率
shot_ratio = kNN_classifier.score(X_test_standard, y_test)

print(shot_ratio)
