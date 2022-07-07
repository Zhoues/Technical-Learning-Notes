from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier

# 加载数据
iris = load_iris()

# 获取样本矩阵和标签向量
X = iris.data
y = iris.target

# 进行训练集和测试集的分离
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

log_reg = LogisticRegression()

# 转变为 OvO 多标签分类模式
ovo = OneVsOneClassifier(log_reg)

ovo.fit(X_train, y_train)
print(ovo.score(X_test, y_test))
