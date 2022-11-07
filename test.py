from __future__ import print_function
import numpy as np
import torch
import braindecode
from torchvision.datasets import CIFAR100, CIFAR10
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from main import similarity_score

if __name__ == '__main__':

    data = np.load('/Users/Riccardo/Workspace/HUST-BCI/repos/EEG/data/BNCI2014001_inter_feature0.npz')
    lst = data.files
    for item in lst:
        print(item)
        print(data[item].shape)

    '''
    criterion = torch.nn.CrossEntropyLoss()

    fitted_loss = []
    y = torch.Tensor([1]).to(torch.long)
    outputs = torch.Tensor([[1, 1]])
    loss = criterion(outputs, y)
    print(loss.item())
    '''
    '''

    train_data = CIFAR100(download=False, root="/mnt/data2/sylyoung/Image/CIFAR100/")

    print(type(train_data))

    b = []
    for i in range(len(a)):
        if i < 200:
            print(a[i][1])
        if i % 5 <= 3:
            b.append(i)
    o_d = torch.utils.data.dataset.Subset(train_data, b)
    print(type(o_d))
    print(len(o_d))
    '''