import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# 加载数据
digits = load_digits()

# 获取样本矩阵和标签向量
X = digits.data
y = digits.target

# 进行训练集和测试集的分离
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

# 使用交叉验证(其实和网格搜索中的GridSearchCV很接近)
best_score, best_p, best_k = 0, 0, 0
for k in range(2, 11):
    for p in range(1, 6):
        knn_clf = KNeighborsClassifier(weights="distance", n_neighbors=k, p=p)
        # 进行交叉验证(cv表示分组的数量)
        scores = cross_val_score(knn_clf, X_train, y_train, cv=5)
        # 取平均值作为最后结果
        score = np.mean(scores)
        if score > best_score:
            best_score, best_p, best_k = score, p, k
print(best_score, best_p, best_k)
