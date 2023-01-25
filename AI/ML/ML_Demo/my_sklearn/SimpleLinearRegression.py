import numpy as np


class SimpleLinearRegression:
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        assert x_train.ndim == 1, \
            "Simple Linear Regressor can only solve single feature training data"
        assert len(x_train) == len(y_train), \
            "the size of x_train must be equal to the size of y_train"

        # 获取平均值
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        # 利用最小二乘法的公式求出参数 a_ 和 b_ (采用向量化运算)
        num = (x_train - x_mean).dot(y_train - y_mean)
        d = (x_train - x_mean).dot(x_train - x_mean)
        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def predict(self, x_predict):
        return np.array(self.a_ * x_predict + self.b_)
