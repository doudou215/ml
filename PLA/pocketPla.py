import numpy as np
import pandas as pd


def loadData(filename):
    data = pd.read_csv(filename, sep='\s+', header=None)
    data = data.values
    row, col = data.shape
    data = np.concatenate((np.ones((row, 1)), data), axis=1)
    X = data[:, :-1]
    y = data[:, col:col + 1]
    return X, y


def mistake(ypred, y):
    row, col = y.shape
    return np.sum(ypred != y) / row


def pocket(X, y, w, n, iterations):
    wbest = np.zeros(w.shape)
    errold = 2
    for i in range(iterations):
        ypred = np.sign(np.dot(X, w.T))
        ypred[np.where(ypred == 0)] = -1  # np.where() return the indexs which satisfy the condition
        #print(ypred)
        errnew = mistake(ypred, y)
        #print(errnew)
        if errnew < errold:
            wbest = w.copy()
            #print(wbest)
            errold = errnew
        index = np.where(ypred != y)[0]  # return row of where ypred!= y
        if not index.any():
            break
        # if not index[index >= prepos].any():
        # prepos = 0  # index[index >= prepos] return the element in index which value >= prepos
        pos = index[np.random.permutation(len(index))][0]  # the first one
        w += n * y[pos] * X[pos, :]
    return wbest


trainX, trainY = loadData('data1.txt')
#print(trainX)
#print(trainX.shape[0])
testX, testY = loadData('testData.txt')
total = 0
for i in range(2000):
    randpos = np.random.permutation(500)
    #print(randpos)
    X = trainX[randpos, :]
    #print(X)
    Y = trainY[randpos, :]
    #print(Y)
    w = np.zeros((1, trainX.shape[1]))
    wbest = pocket(X, Y, w, 1, 50)
    ypred = np.sign(np.dot(testX, wbest.T))
    ypred[np.where(ypred == 0)] = -1
    err = mistake(ypred, testY)
    #print(err)
    total += err
print(total/2000)

