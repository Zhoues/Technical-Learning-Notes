import numpy as np
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

# 加载数据
boston = load_boston()

# 获取样本矩阵和标签向量
X = boston.data
y = boston.target

# 进行训练集和测试集的分离
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 定义超参数的网格
param_grid = [
    {
        'weights': ['uniform'],
        'n_neighbors': [i for i in range(1, 11)]
    },
    {
        'weights': ['distance'],
        'n_neighbors': [i for i in range(1, 11)],
        'p': [i for i in range(1, 6)]
    }
]

# 定义一个分类器
knn_reg = KNeighborsRegressor()

# 定义一个网格搜索器(knn_clf-原始分类器， param_grid-网格超参数, n_jobs-CPU运行颗数, verbose-输出信息详细度)
grid_search = GridSearchCV(knn_reg, param_grid, n_jobs=-1, verbose=2)

# 进行超参数拟合
grid_search.fit(X_train, y_train)

# 获得网格搜索的最佳分类器
print(grid_search.best_estimator_)

# 获取网格搜索的最佳准确率
print(grid_search.best_score_)

# 获取网格搜索的最佳超参数
print(grid_search.best_params_)
