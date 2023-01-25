import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# 加载数据
digits = load_digits()

# 获取样本矩阵和标签向量
X = digits.data
y = digits.target

# 进行训练集和测试集的分离
X_train, X_test, y_train, y_test = train_test_split(X, y)

# # 利用 explained_variance_ratio_ 查看降维的情况以此来选择降维数
# # 查看所有维度的降维覆盖率
# pca = PCA(n_components=X_train.shape[1])
# pca.fit(X_train)
# print(pca.explained_variance_ratio_)
#
# # 画出覆盖率曲线图
# plt.plot([i for i in range(X_train.shape[1])],
#          [np.sum(pca.explained_variance_ratio_[:i + 1])
#           for i in range(X_train.shape[1])])
# plt.show()

# 初始化 PCA 对象，其中 n_components 的值表示最后降维后的主成分个数
# pca = PCA(n_components=50)
# 初始化 PCA 对象, 其中的小数表示方差覆盖的比率
pca = PCA(0.99)
# print(pca.n_components_)

# 进行拟合(先求出前n-2个主成分，然后把n维转化为2维，获得降维矩阵)
# 1. 梯度上升求一个主成分(demean均值归零，求方差最大)
# 2. 把数据在该主成分上的分量去掉(向量相减)
# 3. 直到剩余 n_components 个维度，获得降维矩阵
pca.fit(X_train)

# 获取降维结果(利用输入矩阵和降维矩阵进行点成)
X_train_reduction = pca.transform(X_train)

# 获得测试数据的降维结果
X_test_reduction = pca.transform(X_test)

# 创建一个 kNN算法的分类器
kNN_classifier = KNeighborsClassifier()

# 开始进行拟合
kNN_classifier.fit(X_train_reduction, y_train)

# 查看降维之后方差的覆盖率(列表中数据之和越大越好)


# 计算准确率
print(kNN_classifier.score(X_test_reduction, y_test))
