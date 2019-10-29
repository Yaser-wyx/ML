import numpy as np
def accuracy_score(y_true,y_predict):
    assert y_predict.shape[0] == y_true.shape[0]
    return sum(y_predict==y_true)/len(y_true)