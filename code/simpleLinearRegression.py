import numpy as np


class SimpleLinearRegression:
    def __init__(self):
        self.a_ = 0.0
        self.b_ = 0.0

    def fit(self, x_train, y_train):
        """使用给定的训练数据，来对线性回归模型中的参数进行确定"""
        assert x_train.ndim == 1
        assert len(x_train) == len(y_train)
        num = 0.0
        d = 0.0
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        temp = x_train - x_mean
        num = temp.dot(y_train - y_mean)
        d = temp.dot(temp)
        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        assert x_predict.ndim == 1
        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single_predict):
        return x_single_predict * self.a_ + self.b_



