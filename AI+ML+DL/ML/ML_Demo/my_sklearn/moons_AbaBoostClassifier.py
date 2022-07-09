from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

X, y = make_moons(n_samples=500, noise=0.3, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y)

# 使用AdaBoosting完成集成学习
# (n_estimators为集成模型的个数)
bagging_clf = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=500)

bagging_clf.fit(X_train, y_train)
print(bagging_clf.score(X_test, y_test))

