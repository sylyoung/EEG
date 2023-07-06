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
from flopth import flopth
from models.FC import FC
from scipy.linalg import fractional_matrix_power
from models.EEGNet import EEGNet


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


    a = [[1, 1, 0], [0, 0, 2], [0, 0, -1]]
    print(np.exp(a))



    R = 0



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
    for i in range(5):
        start_time = time.time()

        inputs = torch.randn(13, 2561)
        R = EA_online(inputs, R, i)
        sqrtRefEA = fractional_matrix_power(R, -0.5)
        inputs = np.dot(sqrtRefEA, inputs)

    EA_time = time.time()
    print('EA finished in ms:', np.round((EA_time - start_time) * 1000, 3))

    # declare Model object
    my_model = EEGNet(n_classes=2,
                       Chans=22,
                       Samples=1001,
                       kernLenght=125,
                       F1=4,
                       D=2,
                       F2=8,
                       dropoutRate=0.25,
                       norm_rate=0.5)

    # Use input size
    flops, params = flopth(my_model, in_size=((1, 22, 1001),))
    print(flops, params)

    # Or use input tensors
    dummy_inputs = torch.rand(8, 1, 22, 1001)
    flops, params = flopth(my_model, inputs=(dummy_inputs,))
    print(flops, params)



