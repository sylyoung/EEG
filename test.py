import torch
import numpy as np
import sys
import pandas as pd
import torch as tr
import torch.nn as nn
import scipy
import numpy as np
import torch as tr
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Function
from typing import Optional, Sequence
from torch.autograd import Variable
import time
from models.FC import FC
from scipy.linalg import fractional_matrix_power


def EA(x):
    """
    Parameters
    ----------
    x : numpy array
        data of shape (num_samples, num_channels, num_time_samples)

    Returns
    ----------
    XEA : numpy array
        data of shape (num_samples, num_channels, num_time_samples)
    """
    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)
    sqrtRefEA = fractional_matrix_power(refEA, -0.5)
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])
    return XEA


def EA_online(x, R, sample_num):
    """
    Parameters
    ----------
    x : numpy array
        sample of shape (num_channels, num_time_samples)
    R : numpy array
        current reference matrix (num_channels, num_channels)

    Returns
    ----------
    refEA : numpy array
        data of shape (num_channels, num_channels)
    """

    cov = np.cov(x)
    refEA = (R * sample_num + cov) / (sample_num + 1)
    return refEA


if __name__ == '__main__':

    x = np.load('./data/BNCI2015003/X.npy')
    print(x.shape)
    print(x[0, 0])

    print(np.average([82.639,52.778,95.139,70.833,51.389,75.694,62.5,93.056,82.639]))

    '''
    num_sources = 10
    num_trials = 100
    train_x = []
    for i in range(num_sources):
        source = np.random.rand(num_trials, 22, 256)
        aligned = EA(source)
        train_x.append(aligned)
    train_x = np.concatenate(train_x)
    print(train_x.shape)

    num_test_trials = 50
    test_sample_num = 0
    test_batch_size = 8
    test_batch = np.random.rand(test_batch_size, 22, 256)
    R = 0
    for test_id in range(num_test_trials):
        test_trial = np.random.rand(22, 256)
        R = EA_online(test_trial, R, test_sample_num)
        test_sample_num += 1
        test_batch_aligned = []
        for i in range(test_batch_size):
            test_batch_aligned.append(np.dot(R, test_batch[i]))

            print(test_batch_aligned[i].shape)
    '''






