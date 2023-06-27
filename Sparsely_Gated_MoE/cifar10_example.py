# This file is based on the `https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html`.

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100

from moe import MoE


def compute_mean_std(dataset):
    """compute the mean and std of torchvision dataset
    Args:
        torchvision dataset name

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    if dataset == 'CIFAR10':
        cifar_trainset = CIFAR10(root='/mnt4/zwwang/data/Image/' + dataset + '/', train=True, download=False,
                                 transform=transforms.ToTensor())
    elif dataset == 'CIFAR100':
        cifar_trainset = CIFAR100(root='/mnt4/zwwang/data/Image/' + dataset + '/', train=True, download=False,
                                  transform=transforms.ToTensor())

    imgs = [item[0] for item in cifar_trainset]  # item[0] and item[1] are image and its label
    imgs = torch.stack(imgs, dim=0).numpy()

    # calculate mean over each channel (r,g,b)
    mean_r = imgs[:, 0, :, :].mean()
    mean_g = imgs[:, 1, :, :].mean()
    mean_b = imgs[:, 2, :, :].mean()
    mean = [mean_r, mean_g, mean_b]

    # calculate std over each channel (r,g,b)
    std_r = imgs[:, 0, :, :].std()
    std_g = imgs[:, 1, :, :].std()
    std_b = imgs[:, 2, :, :].std()
    std = std_r, std_g, std_b

    return mean, std


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":

    mean, std = compute_mean_std('CIFAR10')
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=mean, std=std)])
    dataset = 'CIFAR10'
    trainset = torchvision.datasets.CIFAR10(root='/mnt4/zwwang/data/Image/' + dataset + '/', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='/mnt4/zwwang/data/Image/' + dataset + '/', train=False,
                                           download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # if torch.cuda.is_available():
    #     device = torch.device('cuda')
    # else:
    #     device = torch.device('cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    df1 = pd.DataFrame(
        columns=['Num_experts', 'Accuracy'])
    for e in range(10, 16, 5):
        #net = MoE(input_size=3072, output_size=10, num_experts=e, hidden_size=256, noisy_gating=True, k=3)
        net = MoE(input_size=3072, output_size=10, num_experts=e, noisy_gating=True)
        #net = Net()
        net = net.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

        net.train()
        for epoch in range(10):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                inputs = inputs.view(inputs.shape[0], -1)
                outputs, aux_loss = net(inputs)
                # outputs = net(inputs)
                loss = criterion(outputs, labels)
                total_loss = loss + aux_loss
                # total_loss = loss
                total_loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 100 == 99:  # print every 100 mini-batches
                    # print('[%d, %5d] loss: %.3f' %
                    #       (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0

        print('Finished Training')
        path = './cifar_net.pth'
        torch.save(net.state_dict(), path)

        correct = 0
        total = 0
        net.eval()
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs, _ = net(images.view(images.shape[0], -1))
                # outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        result = np.round(100 * correct / total, 3)
        print('Accuracy of the network on the 10000 test images: %.3f %%' % result)
        df1 = df1.append(
            {'Num_experts': e, 'Accuracy': result}, ignore_index=True)
    # df1.to_excel('num_experts.xlsx', sheet_name='sheet_info')

    # ax = df1.plot(title='num_experts', x='Num_experts', y='Accuracy')
    # ax.set_xlabel('Num_experts')
    # ax.set_ylabel('Accuracy')
    # plt.show()

    # MoE yields a test accuracy of around 46 %
    # CNN accuracy is 63.3 %
