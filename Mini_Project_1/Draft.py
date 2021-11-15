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

    def fit(self, x, y, alpha, epsilon, stop):
        xT = np.transpose(x)
        xTx = np.dot(xT, x)
        xTxinv = np.linalg.inv(xTx)
        xTy = np.dot(xT, y)
        w = np.dot(np.dot(xTxinv, xT), y)
        xTxw = np.dot(xTx, w)

        w_before = w
        delta = 2 * (xTxw -xTy)
        w_future = w_before - alpha * delta

        while (abs(w_future - w_before) >= epsilon):
            w_before = w_future
            xTxw_before = np.dot(xTx, w_before)
            delta = 2*(xTxw_before, xTy)
            w_future = w_before - (alpha * delta)

        return w_future
