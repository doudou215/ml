import numpy as np
import pandas as pd


def loadData(filename):
    data = pd.read_csv(filename, sep='\s+', header=None)
    data = data.values
    row, col = data.shape
    data = np.concatenate((np.ones((row, 1)), data), axis=1)
    X = data[:, :-1]
    y = data[:, col:col+1]
    return X, y

def pla1(X, y, w):
    num = 0
    prepos = 0
    while(True):
        ypred = np.sign(np.dot(X, w.T))
        ypred[np.where(ypred == 0)] = -1#np.where() return the indexs which satisfy the condition
        index = np.where(ypred != y)[0]
        if not index.any():
            break
        if not index[index >= prepos].any():
            prepos = 0
        pos = index[index >= prepos][0]
        prepos = pos
        w += y[pos]*X[pos, :]
        num += 1
    return num, w


X, y = loadData('data.txt')
w = np.zeros((1, X.shape[1]))
ret, w = pla1(X, y, w)
print(w )
print(ret)




