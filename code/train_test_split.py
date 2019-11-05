import numpy as np
import math
from sklearn.datasets import load_iris


def train_test_split(X, y, ratio=0.2):
    # 随机生成指定范围的数据
    shuffle_indexes = np.random.permutation(len(X))
    test_size = math.floor(len(X) * ratio)  # 测试数据集大小
    test_indexes = shuffle_indexes[:test_size]
    train_indexes = shuffle_indexes[test_size:]
    # 测试数据
    x_test = X[test_indexes]
    y_test = y[test_indexes]
    # 训练数据
    x_train = X[train_indexes]
    y_train = y[train_indexes]
    return x_train, x_test, y_train, y_test
