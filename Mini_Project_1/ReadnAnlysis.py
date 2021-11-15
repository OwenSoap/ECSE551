import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path1 = 'data/hepatitis.csv'
path2 = 'data/bankrupcy.csv'

with open(path1, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    hepaData = np.array(list(reader)).astype(float)

#print(hepaData)
x = np.delete(hepaData, -1, 1)
#print(x)
y = hepaData[:, [-1]]
#print(y)
class hepatitis():
    def __init__(self,):
        pass

    def fit(self, x, y, alpha, epsilon, iters):
        i = 0
        error = 0
        error_old = 0
        w = np.zeros(len(x))
        wT = np.transpose(w)

        while (abs(error - error_old) >= epsilon & i <= iters):
            i += 1
            wTx = np.dot(x, wT)
            Py1 = (1 / (1 + np.exp(-wTx)))
            delta = np.sum(x * (y - Py1))
            error_old = error
            error = -1*np.sum(np.multiply(y, np.log(Py1)) + np.multiply((1 - y), np.log((1 - Py1))))
            wT = wT - (alpha * delta)

        return wT