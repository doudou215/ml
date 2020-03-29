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

def pla1(X, y, w, n):
    num = 0
    prepos = 0
    while(True):
        ypred = np.sign(np.dot(X, w.T))
        ypred[np.where(ypred == 0)] = -1#np.where() return the indexs which satisfy the condition
        index = np.where(ypred != y)[0]#return row of where ypred!= y
        if not index.any():
            break
        if not index[index >= prepos].any():
            prepos = 0#return the valuegg
        pos = index[index >= prepos][0]
        prepos = pos
        w +=n*y[pos]*X[pos, :]
        num += 1
    return num, w


X, y = loadData('data.txt')
ret = 0
for i in range(2000):
    row = np.random.permutation(X.shape[0])
    Xp = X[row, :]
    yp = y[row, :]
    w = np.zeros((1, X.shape[1]))
    num, w = pla1(Xp, yp, w, 0.5)
    ret += num
print(w)
print(ret/2000)




