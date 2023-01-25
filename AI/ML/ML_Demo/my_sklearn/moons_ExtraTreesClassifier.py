from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=500, noise=0.3, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y)

# 使用随机森林完成集成学习
# (n_estimators 为集成模型的个数，oob_score表示使用没有使用的数据当作测试集,bootstrap选定是放回取样还是不放回取样)
rf_clf = ExtraTreesClassifier(n_estimators=500, random_state=666,
                              oob_score=True, bootstrap=True)

rf_clf.fit(X, y)
# 使用 oob_score_ 参数来进行计算得分
print(rf_clf.oob_score_)
