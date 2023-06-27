"""
@author: Siyang Li
@date: Nov.2 2022
@email: lsyyoungll@gmail.com

Image Classification
"""

import os
import sys
import random
import time

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import warnings
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader, random_split
from skimage import io
from sklearn.metrics import accuracy_score, f1_score
from datetime import date
from tqdm import tqdm
from torchvision.models import resnet50, vit_b_16, efficientnet_b0
from torchvision.datasets import CIFAR100, CIFAR10
from torchsummary import summary


class FC_classifier(nn.Module):

    def __init__(self, fc_num=0, out_chann=0):
        super(FC_classifier, self).__init__()

        # FC Layer
        self.fc = nn.Linear(fc_num, out_chann)

    def forward(self, x):
        x = self.fc(x)
        return x


class CustomImagesDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_file = pd.read_csv(csv_file, header=None, sep=',')
        self.transform = transform

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.csv_file.iloc[idx, 0]

        image = io.imread(img_path)

        # apply transform
        if self.transform:
            image = self.transform(image)

        # convert label to tensor
        label = int(self.csv_file.iloc[idx, 1])
        label = torch.Tensor([label]).to(torch.long)

        return image, label


def compute_mean_std(path, dataset=None, img_size=None):
    """compute the mean and std of torchvision dataset
    Args:
        torchvision dataset name

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    if path is not None:
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.ConvertImageDtype(torch.float32),
             transforms.Resize([img_size, img_size])])
        dataset = CustomImagesDataset(path, transform=transform)
        imgs = [item[0] for item in dataset]  # item[0] and item[1] are image and its label
        imgs = torch.stack(imgs, dim=0).numpy()
    else:
        imgs = [item[0] for item in dataset]  # item[0] and item[1] are image and its label
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


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def image_classification(config):
    device, dataset, num_classes, batch_size, savelog, lr, num_iter, num_workers, exp_id, exp_time, model_arc, \
    mean, std, pretrained, train_all, validation, randaug, labelsmooth, train_paths, test_paths = \
        config['device'], config['dataset'], config['num_classes'], config['batch_size'], config['savelog'], \
        config['lr'], config['num_iter'], config['num_workers'], config['exp_id'], config['exp_time'], \
        config['model_arc'], config['mean'], config['std'], config['pretrained'], config['train_all'], \
        config['validation'], config['randaug'], config['labelsmooth'], config['train_paths'], config['test_paths']

    if model_arc == 'vit':
        feature = vit_b_16(pretrained=pretrained)
        del feature.heads
        if not train_all:
            for para in feature.parameters():
                para.requires_grad = False
        feature.to(device)
        clf = FC_classifier(768, num_classes)
        for para in clf.parameters():
            para.requires_grad = True
        clf.to(device)
        model = [feature, clf]
    elif model_arc == 'effnet':
        model = efficientnet_b0(pretrained=pretrained)
        model_dict = model.state_dict()
        new_state_dict = {}
        for k, v in model_dict.items():
            if not k.startswith('classifier.'):
                new_state_dict[k] = v
        model_dict.update(new_state_dict)
        model.load_state_dict(model_dict)
        if not train_all:
            for para in model.parameters():
                para.requires_grad = False
        model.classifier = nn.Sequential(
            nn.Linear(1280, num_classes),
        )
        for para in model.classifier.parameters():
            para.requires_grad = True
        summary(model, input_size=(3, 224, 224), batch_size=batch_size, device='cpu')
        model.to(device)
    elif model_arc == 'resnet':
        model = resnet50(pretrained=pretrained)
        if not train_all:
            for para in model.parameters():
                para.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        for para in model.fc.parameters():
            para.requires_grad = True
        model.to(device)

    img_size = 224

    if randaug:
        train_transform = transforms.Compose(
            [transforms.RandAugment(),
             transforms.ToTensor(),
             transforms.Resize([img_size, img_size]),
             transforms.Normalize(mean=mean, std=std)])
    else:
        train_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize([img_size, img_size]),
             transforms.Normalize(mean=mean, std=std)])

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize([img_size, img_size]),
         transforms.Normalize(mean=mean, std=std)])

    if dataset == 'CIFAR100':
        train_data = CIFAR100(download=False, root="/mnt/data2/sylyoung/Image/CIFAR100/", transform=train_transform)
        test_data = CIFAR100(root="/mnt/data2/sylyoung/Image/CIFAR100/", train=False, transform=test_transform)
    elif dataset == 'CIFAR10':
        train_data = CIFAR10(download=False, root="/mnt/data2/sylyoung/Image/CIFAR10/", transform=train_transform)
        test_data = CIFAR10(root="/mnt/data2/sylyoung/Image/CIFAR10/", train=False, transform=test_transform)
    elif dataset == 'Office31':
        train_data = CustomImagesDataset(train_paths[0], transform=train_transform)
        test_data = CustomImagesDataset(test_paths[0], transform=test_transform)

    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    if validation:
        splitted_datasets = random_split(train_data,
                                         [int(0.8 * len(train_data)), len(train_data) - int(0.8 * len(train_data))],
                                         generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(splitted_datasets[0], batch_size=batch_size, num_workers=num_workers, shuffle=True)
        valid_loader = DataLoader(splitted_datasets[1], batch_size, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)


    def train(model, iterator, optimizer, criterion, savelog=False, epoch=None, exp_time=None):

        print('training...')

        epoch_loss = 0

        correct = 0
        total = 0

        if type(model) is list:
            model[0].train()
            model[1].train()
        else:
            model.train()

        for batch in tqdm(iterator):
            optimizer.zero_grad()

            data, labels = batch
            data, labels = data.to(device), labels.to(device)

            if type(model) is list:
                predictions = model[1](model[0](data))
            else:
                predictions = model(data)

            _, predicted = torch.max(predictions.cpu().data, 1)

            loss = criterion(predictions, labels.reshape(-1, ))

            correct += len(np.where(predicted.reshape(-1, 1) == labels.detach().cpu().reshape(-1, 1))[0])
            total += predicted.shape[0]

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss, avg_acc = epoch_loss / len(iterator), np.round(100 * correct / total, 3)

        if savelog:
            if not os.path.isdir('./runs'):
                os.mkdir('./runs')
            if not os.path.isdir('./runs/' + dataset):
                os.mkdir('./runs/' + dataset)
            with open('./runs/' + dataset + '/log_train_' + exp_time + exp_id, 'a') as logfile:
                logfile.write('Epoch ' + str(epoch) + ': Train Loss ' + str(round(avg_loss, 3)) + ', Train Acc ' + str(
                    round(avg_acc, 3)))
                logfile.write('\n')

        print('#' * 50)

        return avg_loss, avg_acc

    def evaluate(model, iterator, criterion, savelog=False, phase=None, epoch=None, exp_time=None):

        print('evaluating...')

        epoch_loss = 0

        if type(model) is list:
            model[0].eval()
            model[1].eval()
        else:
            model.eval()

        predicted_all = []
        labels_all = []

        with torch.no_grad():
            for batch in tqdm(iterator):
                data, labels = batch
                data, labels = data.to(device), labels.to(device)

                if type(model) is list:
                    predictions = model[1](model[0](data))
                else:
                    predictions = model(data)
                _, predicted = torch.max(predictions.cpu().data, 1)

                loss = criterion(predictions, labels.reshape(-1, ))

                predicted_all.append(predicted)
                labels_all.append(labels.detach().cpu())

                epoch_loss += loss.item()

        predicted_all = np.concatenate(predicted_all, axis=0)
        labels_all = np.concatenate(labels_all, axis=0)
        predicted_all = predicted_all.reshape(-1, 1)
        labels_all = labels_all.reshape(-1, 1)
        f1_arr = f1_score(labels_all, predicted_all, average=None)
        epoch_acc = accuracy_score(labels_all, predicted_all)

        avg_loss = epoch_loss / len(iterator)

        if savelog and phase == 'Valid':
            if not os.path.isdir('./runs'):
                os.mkdir('./runs')
            if not os.path.isdir('./runs/' + dataset):
                os.mkdir('./runs/' + dataset)
            with open('./runs/' + dataset + '/log_train_' + exp_time + exp_id, 'a') as logfile:
                logfile.write('Epoch ' + str(epoch) + ': Valid Loss ' + str(round(avg_loss, 3)) + ', Valid Acc ' +
                              str(round(epoch_acc, 3)) + ', Valid F1 ' + str(round(np.average(f1_arr), 3)))
                logfile.write('\n')

        if savelog and phase == 'Test':
            if not os.path.isdir('./runs'):
                os.mkdir('./runs')
            if not os.path.isdir('./runs/' + dataset):
                os.mkdir('./runs/' + dataset)
            with open('./runs/' + dataset + '/log_test_' + exp_time + exp_id, 'a') as logfile:
                logfile.write('Test Loss ' + str(round(avg_loss, 3)) + ', Test Acc ' + str(round(epoch_acc, 3))
                              + ', Test F1 ' + str(round(np.average(f1_arr), 3)))
                logfile.write('\n')

        return avg_loss, epoch_acc, f1_arr

    if type(model) is list:
        optimizer = optim.Adam(list(model[0].parameters()) + list(model[1].parameters()), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    if labelsmooth != 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=labelsmooth)
    else:
        criterion = nn.CrossEntropyLoss()
    criterion_eval = nn.CrossEntropyLoss()

    best_valid_loss = float('inf')

    # training phase
    not_improving_cnt = 0
    for epoch in range(num_iter):
        print('epoch : ' + str(epoch + 1) + ' of ' + str(num_iter))

        start_time = time.time()

        train_loss, train_acc = train(model, train_loader, optimizer, criterion,
                                      savelog=savelog, epoch=(epoch + 1), exp_time=exp_time)

        print(f'\tTrain Loss/Accuracy:  {train_loss:.5f} {train_acc:.5f}')

        # early stopping (model save) with validation set
        if validation:
            valid_loss, valid_acc, f1_arr = evaluate(model, valid_loader, criterion_eval,
                                                     savelog=savelog, phase='Valid', epoch=(epoch + 1),
                                                     exp_time=exp_time)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                print('Current epoch is best epoch for valid set, saving models...')
                if type(model) is list:
                    torch.save(model[0].state_dict(),
                               './runs/' + dataset + '/model_feature_best_' + exp_time + exp_id + '.pt')
                    torch.save(model[1].state_dict(),
                               './runs/' + dataset + '/model_clf_best_' + exp_time + exp_id + '.pt')
                else:
                    torch.save(model.state_dict(), './runs/' + dataset + '/model_best_' + exp_time + exp_id + '.pt')
                not_improving_cnt = 0
            else:
                not_improving_cnt += 1
            if not_improving_cnt == 5:
                break
            print(f'\t Val. Loss/Accuracy/F1: {valid_loss:.5f} {valid_acc:.5f} {np.average(f1_arr):.5f}')
        else:
            #if epoch + 1 == num_iter: # TODO
            if epoch + 1 % 5 or epoch + 1 == num_iter:
                if type(model) is list:
                    torch.save(model[0].state_dict(), './runs/' + dataset + '/model_feature_epoch_' + str(
                        epoch + 1) + '_' + exp_time + exp_id + '.pt')
                    torch.save(model[1].state_dict(), './runs/' + dataset + '/model_clf_epoch_' + str(
                        epoch + 1) + '_' + exp_time + exp_id + '.pt')
                else:
                    torch.save(model.state_dict(), './runs/' + dataset + '/model_epoch_' + str(
                        epoch + 1) + '_' + exp_time + exp_id + '.pt')

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if not os.path.isdir('./runs'):
            os.mkdir('./runs')
        if not os.path.isdir('./runs/' + dataset):
            os.mkdir('./runs/' + dataset)

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')

    # test phase
    # load best if early stopping with validation set
    if validation:
        if type(model) is list:
            model[0].load_state_dict(
                torch.load('./runs/' + dataset + '/model_feature_best_' + exp_time + exp_id + '.pt'))
            model[1].load_state_dict(
                torch.load('./runs/' + dataset + '/model_clf_best_' + exp_time + exp_id + '.pt'))
        else:
            model.load_state_dict(
                torch.load('./runs/' + dataset + '/model_best_' + exp_time + exp_id + '.pt'))

    test_loss, test_acc, f1_arr = evaluate(model, test_loader, criterion, savelog=savelog, phase='Test',
                                           exp_time=exp_time)
    print(f'Test Loss/Accuracy/F1: {test_loss:.3f} {test_acc:.3f} {np.average(f1_arr):.3f}')


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    config = {}

    config['exp_id'] = '_10'  # experiment ID
    config['exp_time'] = str(date.today())  # experiment time
    config['dataset'] = 'Office31'  # dataset name string
    config['model_arc'] = 'resnet'  # model architecture
    config['validation'] = False  # use validation set
    config['test'] = False  # whether to test only
    config['savelog'] = True  # whether to save running outputs (loss/acc)

    # number of classes
    if config['dataset'] == 'CIFAR100':
        config['num_classes'] = 100
    elif config['dataset'] == 'CIFAR10':
        config['num_classes'] = 10
    elif config['dataset'] == 'Office31':
        config['num_classes'] = 31

    config['lr'] = 0.00001  # learning rate
    config['num_iter'] = 20  # number of max iterations
    config['batch_size'] = 16  # training batch size
    config['num_workers'] = 8  # number of parallel workers
    config['pretrained'] = True  # use ImageNet pretrained weights
    config['train_all'] = True  # train all weights
    config['randaug'] = False  # apply randaugment
    config['labelsmooth'] = 0  # apply label smoothing with specified alpha

    if config['dataset'] == 'CIFAR10':
        dataset_torch = CIFAR10(root='/mnt/data2/sylyoung/Image/' + config['dataset'] + '/', train=True, download=False, transform=transforms.ToTensor())
    elif config['dataset'] == 'CIFAR100':
        dataset_torch = CIFAR100(root='/mnt/data2/sylyoung/Image/' + config['dataset'] + '/', train=True, download=False, transform=transforms.ToTensor())

    if 'CIFAR' in config['dataset']:
        mean, std = compute_mean_std(None, dataset_torch)  # training set mean and std for normalization
    else:
        if config['dataset'] == 'Office31':
            train_path = '/mnt/data2/sylyoung/Image/' + config['dataset'] + '/OFFICE31/' + 'A.csv'
            test_path = '/mnt/data2/sylyoung/Image/' + config['dataset'] + '/OFFICE31/' + 'D.csv'
            img_size = 224
        mean, std = compute_mean_std(train_path, img_size=img_size)  # training set mean and std for normalization

        config['train_paths'] = [train_path]
        config['test_paths'] = [test_path]

    config['mean'] = mean
    config['std'] = std

    cuda_device_id = str(sys.argv[1])
    try:
        device = torch.device('cuda:' + cuda_device_id)
        print('using GPU')
    except:
        try:
            device = torch.device('cuda:' + str(0))
            print('using GPU id 0')
        except:
            device = torch.device('cpu')
            print('using CPU')

    config['device'] = device

    print(config)

    if not config['test']:
        with open('./runs/image_configs', 'a') as logfile:
            for key, value in config.items():
                logfile.write(str(key) + ': ' + str(value))
                logfile.write('\n')
            logfile.write('\n')
    else:
        config['num_iter'] = 0

    image_classification(config)

