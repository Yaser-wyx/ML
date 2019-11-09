import numpy as np
import matplotlib.pyplot as plt


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None

    # 使用传入的参数来计算其主成分
    def fit(self, X, eta=0.01, n_iters=10000):
        def df(w, X):
            return X.T.dot(X.dot(w)) * (2 / len(X))

        def demean(X):
            return X - np.mean(X, axis=0)

        def direction(w):
            return w / np.linalg.norm(w)

        def first_component(initial_w, X, eta, n_iters):
            w = direction(initial_w)
            for i in range(n_iters):
                gradient = df(w, X)
                w = direction(w + eta * gradient)
            return w

        # 初始化数据
        self.components_ = np.empty((self.n_components, X.shape[1]))
        X_pca = demean(X)
        for i in range(self.n_components):
            initial_w = np.random.random(X.shape[1])
            w = first_component(initial_w, X_pca, eta, n_iters)
            self.components_[i] = w
            X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w
        return self

    def transform(self, X):
        return X.dot(self.components_.T)

    def invert_transform(self, X):
        return X.dot(self.components_)


size = 1000
np.random.seed(666)
X = np.empty((size, 2))
X[:, 0] = np.random.uniform(0., 100., size=size)
X[:, 1] = 0.68 * X[:, 0] + 3.57 + np.random.normal(0, 5, size=size)

pca = PCA(1)
pca.fit(X)
x_pca = pca.transform(X)

plt.scatter(X[:, 0], X[:, 1])
x_pca_invert = pca.invert_transform(x_pca)
plt.scatter(x_pca_invert[:, 0], x_pca_invert[:, 1], color='r', alpha=0.5)
plt.show()
