from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

X, y = make_moons(n_samples=500, noise=0.3, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y)

# 使用bagging完成集成学习
# (n_estimators为集成模型的个数,max_samples为每一个子模型看多少个数据,
# bootstrap选定是放回取样还是不放回取样,oob_score表示使用没有使用的数据当作测试集)
bagging_clf = BaggingClassifier(DecisionTreeClassifier(),
                                n_estimators=500, max_samples=100,
                                bootstrap=True, oob_score=True)

bagging_clf.fit(X, y)
# 使用 oob_score_ 参数来进行计算得分
print(bagging_clf.oob_score_)

# plt.scatter(X[y == 0, 0], X[y == 0, 1])
# plt.scatter(X[y == 1, 0], X[y == 1, 1])
# plt.show()
