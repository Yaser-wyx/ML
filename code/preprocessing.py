import numpy as np
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split


# 数据归一化预处理
class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scaler_ = None

    def fit(self, X):
        """根据传进来的训练数据集X，获取数据的均值以及方差"""
        assert X.ndim == 2
        self.mean_ = np.array([np.mean(X[:, i]) for i in range(X.shape[1])])
        self.scaler_ = np.array([np.std(X[:, i]) for i in range(X.shape[1])])
        return self

    def transform(self, X):
        """将X进行均值方差归一化"""
        assert X.ndim == 2
        assert self.mean_ is not None and self.scaler_ is not None
        assert X.shape[1] == len(self.mean_)
        resX = np.empty(X.shape, dtype=float)
        for col in range(X.shape[1]):
            resX[:, col] = (X[:, col] - self.mean_[col]) / self.scaler_[col]
        return resX


iris = load_iris()
X = iris.data
Y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=666)
standardScaler = StandardScaler()
standardScaler.fit(X_train)

X_test_standard = standardScaler.transform(X_test)
print(X_test_standard)
