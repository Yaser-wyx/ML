import numpy as np
import matplotlib.pyplot as plt


def J(theta, X_b, y):
    return np.sum((X_b.dot(theta) - y) ** 2) / len(theta)


def dJ(theta, X_b, y):
    res = np.empty(theta.shape)
    for i in range(0, len(theta)):
        res[i] = (X_b.dot(theta) - y).dot(X_b[:, i])
    return res * (2 / len(X_b))


def gradient_descent(X_b, y, initial_theta, eta, n_iters=10000, epsilon=1e-8):
    theta = initial_theta
    for i in range(0, n_iters):
        gradient = dJ(theta, X_b, y)
        last_theta = theta
        theta = theta - eta * gradient
        if abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon:
            break
    return theta


np.random.seed(666)
x = 2 * np.random.random(size=100)
y = x * 3.0 + 4.0 + np.random.normal(size=100)
X = x.reshape(-1, 1)

X_b = np.hstack([np.ones((len(X), 1)), X])
initial_theta = np.zeros(X_b.shape[1])
eta = 0.1
# print(len(initial_theta),len(X_b))
theta = gradient_descent(X_b, y, initial_theta, eta)
print(theta)
