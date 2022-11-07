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


def compute_mean_std(dataset):
    """compute the mean and std of torchvision dataset
    Args:
        torchvision dataset name

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    if dataset == 'CIFAR10':
        cifar_trainset = CIFAR10(root='/mnt/data2/sylyoung/Image/' + dataset + '/', train=True, download=False, transform=transforms.ToTensor())
    elif dataset == 'CIFAR100':
        cifar_trainset = CIFAR100(root='/mnt/data2/sylyoung/Image/' + dataset + '/', train=True, download=False, transform=transforms.ToTensor())

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


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def image_classification(config):
    device, dataset, num_classes, batch_size, savelog, lr, num_iter, num_workers, exp_id, exp_time, model_arc, \
    mean, std, pretrained, train_all, validation, randaug, labelsmooth, retrain, retrain_load_id = \
        config['device'], config['dataset'], config['num_classes'], config['batch_size'], config['savelog'], \
        config['lr'], config['num_iter'], config['num_workers'], config['exp_id'], config['exp_time'], \
        config['model_arc'], config['mean'], config['std'], config['pretrained'], config['train_all'], \
        config['validation'], config['randaug'], config['labelsmooth'], config['retrain'], config['retrain_load_id']

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
            [transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
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

    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    if validation:
        splitted_datasets = random_split(train_data,
                                         [int(0.8 * len(train_data)), len(train_data) - int(0.8 * len(train_data))],
                                         generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(splitted_datasets[0], batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                  shuffle=True)
        valid_loader = DataLoader(splitted_datasets[1], batch_size, num_workers=num_workers, pin_memory=True)
    # retraining and elimiate bad training data based on pretrained model loss rankings
    elif retrain:
        train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, pin_memory=False,
                                  shuffle=False)
        if model_arc == 'vit':
            feature_cp = vit_b_16(pretrained=False)
            del feature_cp.heads
            if not train_all:
                for para in feature_cp.parameters():
                    para.requires_grad = False
            feature_cp.to(device)
            clf_cp = FC_classifier(768, num_classes)
            for para in clf_cp.parameters():
                para.requires_grad = True
            clf_cp.to(device)
            model_cp = [feature_cp, clf_cp]

        if type(model_cp) is list:
            model_cp[0].load_state_dict(
                torch.load('./runs/' + dataset + '/model_feature_epoch_19_' + '2022-11-03' + retrain_load_id + '.pt'))
            model_cp[1].load_state_dict(
                torch.load('./runs/' + dataset + '/model_clf_epoch_19_' + '2022-11-03' + retrain_load_id + '.pt'))
        else:
            model_cp.load_state_dict(
                torch.load('./runs/' + dataset + '/model_best_' + '2022-11-03' + retrain_load_id + '.pt'))

        criterion_retrain = torch.nn.CrossEntropyLoss(reduction='none')

        fitted_loss = []
        for batch in tqdm(train_loader):
            data, labels = batch
            data, labels = data.to(device), labels.to(device)

            if type(model_cp) is list:
                predictions = model_cp[1](model_cp[0](data))
            else:
                predictions = model_cp(data)

            _, predicted = torch.max(predictions.cpu().data, 1)

            loss_arr = criterion_retrain(predictions, labels.reshape(-1, ))
            for loss in loss_arr:
                loss = loss.cpu().item()
                fitted_loss.append(loss)

        sorted_loss = np.sort(fitted_loss)

        # threhold is 90%
        #threshold = sorted_loss[int(len(sorted_loss) * 0.9)]
        threshold = 0.693147

        loss_good_indices = []
        for l in range(len(fitted_loss)):
            if fitted_loss[l] < threshold:
                loss_good_indices.append(l)
        train_data_trim = torch.utils.data.dataset.Subset(train_data, loss_good_indices)

        print('train data trim len:', len(train_data_trim))
        train_loader = DataLoader(train_data_trim, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                  shuffle=True)
        validation = False
    else:
        train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                  shuffle=True)

    def train(model, iterator, optimizer, criterion, savelog=False, epoch=None, exp_time=None):

        print('training...')

        epoch_loss = 0
        epoch_acc = 0

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
            acc = accuracy_score(predicted, labels.detach().cpu())

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc

        avg_loss, avg_acc = epoch_loss / len(iterator), epoch_acc / len(iterator)

        if savelog:
            if not os.path.isdir('./runs'):
                os.mkdir('./runs')
            if not os.path.isdir('./runs/' + dataset):
                os.mkdir('./runs/' + dataset)
            with open('./runs/' + dataset + '/log_train_' + exp_time + exp_id, 'a') as logfile:
                logfile.write('Epoch ' + str(epoch) + ': Train Loss ' + str(round(avg_loss, 5)) + ', Train Acc ' + str(
                    round(avg_acc, 5)))
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
                logfile.write('Epoch ' + str(epoch) + ': Valid Loss ' + str(round(avg_loss, 5)) + ', Valid Acc ' +
                              str(round(epoch_acc, 5)) + ', Valid F1 ' + str(round(np.average(f1_arr), 5)))
                logfile.write('\n')

        if savelog and phase == 'Test':
            if not os.path.isdir('./runs'):
                os.mkdir('./runs')
            if not os.path.isdir('./runs/' + dataset):
                os.mkdir('./runs/' + dataset)
            with open('./runs/' + dataset + '/log_test_' + exp_time + exp_id, 'a') as logfile:
                logfile.write('Test Loss ' + str(round(avg_loss, 5)) + ', Test Acc ' + str(round(epoch_acc, 5))
                              + ', Test F1 ' + str(round(np.average(f1_arr), 5)))
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
    print(f'Test Loss/Accuracy/F1: {test_loss:.5f} {test_acc:.5f} {np.average(f1_arr):.5f}')

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    config = {}

    config['exp_id'] = '_8'  # experiment ID
    config['exp_time'] = str(date.today())  # experiment time
    config['dataset'] = 'CIFAR100'  # dataset name string
    config['model_arc'] = 'vit'  # model architecture
    config['validation'] = False  # use validation set
    config['test'] = False  # whether to test only
    config['savelog'] = True  # whether to save running outputs (loss/acc)

    # number of classes
    if config['dataset'] == 'CIFAR100':
        config['num_classes'] = 100
    elif config['dataset'] == 'CIFAR10':
        config['num_classes'] = 10

    config['lr'] = 0.00001  # learning rate
    config['num_iter'] = 42  # number of max iterations
    config['batch_size'] = 64  # training batch size
    config['num_workers'] = 8  # number of parallel workers
    config['pretrained'] = True  # use ImageNet pretrained weights
    config['train_all'] = True  # train all weights
    config['randaug'] = True  # apply randaugment
    config['labelsmooth'] = 0  # apply label smoothing with specified alpha
    config['retrain'] = True  # apply retraining
    config['retrain_load_id'] = '_5' # retraining load experiment id

    mean, std = compute_mean_std(config['dataset'])  # training set mean and std for normalization
    config['mean'] = mean
    config['std'] = std
    config['device'] = torch.device('cuda:' + str(3))  # cuda device id

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

