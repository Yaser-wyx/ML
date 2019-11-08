import numpy as np
from sklearn.metrics import r2_score


class LinearRegression:
    def __init__(self):
        self.coef_ = None  # 系数
        self.intercept_ = None  # 截距
        self._theta = None

    def fit_normal(self, X_train, y_train):
        """使用正规方程来训练模型"""
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def fit_gd(self, X_train, y_train, eta=0.1, n_iters=10000):
        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
            except:
                return float('inf')

        def dJ(theta, X_b, y):
            return X_b.T.dot(X_b.dot(theta) - y) * (2. / len(X_b))

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=10000, epsilon=1e-8):
            theta = initial_theta
            for i in range(0, n_iters):
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon:
                    break
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)
        self.coef_ = self._theta[1:]
        self.intercept_ = self._theta[0]
        return self

    # 随机梯度下降
    def fit_gsd(self, X_train, y_train, n_iters=5, t0=5, t1=50):
        # 求单个样本的梯度值
        def dJ(X_b_i, y_i, theta):
            return X_b_i*(X_b_i.dot(theta) - y_i) * 2

        def learning_rate(b):
            return t0 / (b + t1)

        def gsd(X_b, y, n_iters, initial_theta):
            theta = initial_theta
            m = len(X_b)
            for cur_iter in range(n_iters):
                # 随机化原始数据集
                indexes = np.random.permutation(m)
                X_b_random = X_b[indexes]
                y_random = y[indexes]
                for i in range(m):
                    theta = theta - learning_rate(i + cur_iter * m) * dJ(X_b_random[i], y_random[i], theta)
            return theta
        # 扩展原始数据的系数
        X_b = np.hstack([np.ones(len(X_train)).reshape(-1, 1), X_train])
        # 初始化系数
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gsd(X_b, y_train, n_iters, initial_theta)
        self.coef_ = self._theta[1]
        self.intercept_ = self._theta[0]
        return self

    def predict(self, X_predict):
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)
