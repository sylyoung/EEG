import torch
import numpy as np
import sys
import pandas as pd



x1 = np.load('x1.npy')
x2 = np.load('x2.npy')

print(x1.shape)
print(x2.shape)



'''
a = np.array([[74.618,75.069,74.618,74.792,73.021]
,[86.042,85.347,85.556,84.167,86.007]
,[80.347,80.,79.583,78.75,79.514]
,[84.583,83.889,84.479,85.764,84.965]
,[88.125,86.875,87.708,87.708,87.674]
,[80.278,81.528,80.486,80.903,79.653]
,[75.069,77.847,76.389,76.701,76.875]
,[73.889,74.41,73.576,73.021,74.583]
,[88.09,87.639,89.097,88.16,88.16,]
,[85.556,87.083,86.215,85.903,86.563]])


print(a)

scores_arr = a

print('all scores', scores_arr)
print('all avgs', np.average(scores_arr, 1).round(3))
print('sbj stds', np.std(scores_arr, 1).round(3))
print('all avg', np.average(np.average(scores_arr, 0)).round(3))
print('all std', np.std(np.average(scores_arr, 0)).round(3))
'''