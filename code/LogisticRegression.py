import numpy as np
from sklearn.metrics import accuracy_score


class LogisticRegression:
    def __init__(self):
        self.coef_ = None  # 系数
        self.intercept_ = None  # 截距
        self._theta = None

    def sigmoid(self, t):
        return 1 / (1 + np.exp(-t))

    def fit(self, X_train, y_train, eta=0.01, n_iters=10000):
        def dJ(theta, X_b, y):
            return X_b.T.dot(self.sigmoid(X_b.dot(theta) )- y) / len(X_b)

        def J(theta, X_b, y):
            y_hat = self.sigmoid(X_b.dot(theta))
            try:
                return - np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / len(y)
            except:
                return float('inf')
        def gradient_desc(X_b, initial_theta, y, eta, n_iters, epsilon=1e-8):
            theta = initial_theta
            for i in range(n_iters):
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_desc(X_b, initial_theta, y_train, eta, n_iters)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def predict_prob(self, X_test):
        X_b = np.hstack([np.ones((len(X_test), 1)), X_test])
        # 使用sigmoid函数对预测的结果值进行映射到0~1之间
        return self.sigmoid(X_b.dot(self._theta))

    def predict(self, X_test):
        prob = self.predict_prob(X_test)
        # 获取概率值
        return np.array(prob >= 0.5, dtype='int')  # 将大于0.5的值映射为1

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)
