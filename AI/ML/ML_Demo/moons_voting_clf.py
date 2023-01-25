from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

X, y = make_moons(n_samples=500, noise=0.01, random_state=43)

X_train, X_test, y_train, y_test = train_test_split(X, y)

# 使用集成学习
voting_clf = VotingClassifier([
    ('log_clf', LogisticRegression()),
    ('svm_clf', SVC()),
    ('dt_clf', DecisionTreeClassifier())

], voting='hard')

voting_clf.fit(X_train, y_train)
print(voting_clf.score(X_test, y_test))
# plt.scatter(X[y == 0, 0], X[y == 0, 1])
# plt.scatter(X[y == 1, 0], X[y == 1, 1])
# plt.show()
