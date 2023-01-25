from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

X, y = make_moons(n_samples=500, noise=0.3, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y)

# 使用GradientBoosting完成集成学习
# 由于GradientBoosting只能使用决策树，所以基本参数和决策树一致
# (n_estimators为集成模型的个数)
gd_clf = GradientBoostingClassifier(max_depth=2, n_estimators=500)

gd_clf.fit(X_train, y_train)
print(gd_clf.score(X_test, y_test))
