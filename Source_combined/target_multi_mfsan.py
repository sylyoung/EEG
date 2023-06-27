# -*- coding: utf-8 -*-
import argparse
import time

import torch as tr
import torch.nn.functional as F
import numpy as np
import copy
import math
import random
import os
from os import walk
from scipy.io import loadmat
from torch.utils.data import Dataset
from utils import network
from utils.dataloader import read_seed_all, read_mi_all, read_seizure_all, data_normalize
from utils.network import MSMDAERNet
from utils.CsvRecord import CsvRecord
from utils.LogRecord import LogRecord


def setup_seed(seed):
    tr.manual_seed(seed)
    tr.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    tr.backends.cudnn.deterministic = True


class CustomDataset(Dataset):
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        data = tr.Tensor(self.Data[index])
        label = tr.LongTensor(self.Label[index])
        return data, label


def train(model, args, source_loaders, target_loader):
    source_iters = []
    for i in range(len(source_loaders)):
        source_iters.append(iter(source_loaders[i]))
    target_iter = iter(target_loader)

    for i in range(1, args.iteration + 1):
        model.train()
        LEARNING_RATE = args.lr / math.pow((1 + 10 * (i - 1) / (args.iteration)), 0.75)
        # if (i - 1) % 100 == 0:
        #     print("\nLearning rate: ", np.round(LEARNING_RATE, 5))

        optimizer = tr.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=args.momentum)

        for j in range(len(source_iters)):
            try:
                source_data, source_label = next(source_iters[j])
            except:
                source_iters[j] = iter(source_loaders[j])
                source_data, source_label = next(source_iters[j])

            try:
                target_data, _ = next(target_iter)
            except:
                target_iter = iter(target_loader)
                target_data, _ = next(target_iter)

            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()

            optimizer.zero_grad()
            cls_loss, mmd_loss, l1_loss = model(source_data, num_src=len(source_iters), data_tgt=target_data,
                                                label_src=source_label, mark=j)
            gamma = 2 / (1 + math.exp(-10 * i / args.iteration)) - 1

            # raw MFSAN
            loss = cls_loss + gamma * (mmd_loss + l1_loss)

            loss.backward()
            optimizer.step()

            # if i % args.log_interval == 0:
            #     print('Train S' + str(j) + ', iter: {}/{} loss: {:.5f}'.format(i, args.iteration, loss.item()))

        if i % (args.log_interval * 20) == 0:
            acc_t_te, sen_t_te, spec_t_te, auc_t_te = cal_acc(model, source_loaders, target_loader)
            log_str = 'Task: {}, Iter:{}/{}, acc = {:.2f}%, sen = {:.2f}%, spec = {:.2f}%, auc = {:.2f}%'.format(args.uda, i, args.iteration, acc_t_te, sen_t_te, spec_t_te, auc_t_te)
            args.log.record(log_str)
            print(log_str)

    return auc_t_te


def cross_subject(data, label, subject_id, args):
    train_idxs = list(range(args.N))
    del train_idxs[subject_id]
    test_idx = subject_id
    target_data, target_label = data[test_idx], label[test_idx]
    source_data = np.array(data, dtype=object)[train_idxs].tolist()
    source_label = np.array(label, dtype=object)[train_idxs].tolist()
    print('each sub data shape', target_data.shape)

    del label
    del data

    source_loaders = []
    for j in range(len(source_data)):
        source_loaders.append(tr.utils.data.DataLoader(dataset=CustomDataset(source_data[j], source_label[j]),
                                                       batch_size=args.batch_size,
                                                       shuffle=True, drop_last=True))

    target_loader = tr.utils.data.DataLoader(dataset=CustomDataset(target_data, target_label),
                                             batch_size=args.batch_size,
                                             shuffle=True, drop_last=True)

    if args.bottleneck == 50:
        backbone_net, _ = network.backbone_net(args, 100, return_type='y')
    if args.bottleneck == 64:
        backbone_net, _ = network.backbone_net(args, 128, return_type='y')
    model = MSMDAERNet(backbone_net, num_src=len(train_idxs), num_class=args.class_num)
    test_acc = train(model.cuda(), args, source_loaders, target_loader)

    return test_acc


def cal_acc(model, source_loaders, target_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with tr.no_grad():
        for data, target in target_loader:
            data = data.cuda()
            target = target.cuda()
            preds = model(data, len(source_loaders))
            for i in range(len(preds)):
                preds[i] = F.softmax(preds[i], dim=1)
            pred = sum(preds) / len(preds)
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target.squeeze()).item()
            pred = pred.data.max(1)[1]
            correct += pred.eq(target.data.squeeze()).cpu().sum()

        test_loss /= len(target_loader.dataset)
        test_acc = 100. * correct / len(target_loader.dataset)

    return test_acc


if __name__ == '__main__':
    # 论文中bz是256，但是128好像好一些
    # 后续对比下backbone在LN2_ReLU，和这里的MLP3_LeakyReLU的差异

    setup_seed(2020)

    data_name_list = ['001-2014', '001-2014_2', 'SEED', 'SEED4']
    data_name = data_name_list[1]
    if data_name == '001-2014': N, chn, class_num, trial_num = 9, 22, 4, 288  # 001-2014
    if data_name == '001-2014_2': N, chn, class_num, trial_num = 9, 22, 2, 144  # 001-2014_2
    if data_name == 'SEED': N, chn, class_num, trial_num = 15, 62, 3, 3394
    if data_name == 'SEED4': N, chn, class_num, trial_num = 15, 62, 4, 851

    args = argparse.Namespace(N=N, chn=chn, class_num=class_num, trial_num=trial_num,
                              lr=0.01, momentum=0.9, log_interval=10,
                              bottleneck=64, layer='wn')

    # data preparation
    args.backbone = 'Net_ln2'
    args.app = 'no'
    args.method = 'MFSAN'
    args.data = data_name
    print('Model:', args.method, ' Data:', args.data)

    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    args.data_env = 'gpu' if tr.cuda.device_count() != 0 else 'local'

    if args.data in ['SEED', 'SEED4', 'seizure']:
        args.batch_size = 128
        args.epoch = 200
        args.norm_type = 'zscore'
        args.input_dim = 1332
        data, label = read_seizure_all(args)
        args.validation = 'random'
    else:
        args.batch_size = 8
        args.epoch = 100
        args.norm_type = 'none'
        args.aug = 1
        args.cov_type = 'oas'
        args.input_dim = int(args.chn * (args.chn + 1) / 2)
        data, label = read_mi_all(args)
        args.validation = 'last'

    args.SEED = 2020
    tr.manual_seed(args.SEED)
    tr.cuda.manual_seed(args.SEED)
    np.random.seed(args.SEED)
    random.seed(args.SEED)
    tr.backends.cudnn.deterministic = True

    # training settings
    args.iteration = math.ceil(args.epoch * args.trial_num / args.batch_size)
    print('BS: {}, epoch: {}, Iter: {}'.format(args.batch_size, args.epoch, args.iteration))
    print(args)

    args.local_dir = r'/Users/wenz/code/PyPrj/TL/TL_BCI/MSDT/'
    args.result_dir = 'results/target/'
    my_log = LogRecord(args)
    my_log.log_init()
    my_log.record('=' * 50 + '\n' + os.path.basename(__file__) + '\n' + '=' * 50)

    print('Normalization type: ', args.norm_type)
    data_norm = copy.deepcopy(data)
    label_norm = copy.deepcopy(label)
    if args.norm_type != 'none':
        for i in range(len(data_norm)):
            data_norm[i] = data_normalize(data_norm[i], args.norm_type)

    # cross-subject, (SEED session 0, 1-14 as sources, 15 as target)
    sub_acc_all = np.zeros(N)
    duration_all = np.zeros(N)
    for idt in range(N):
        source_str = 'Except_S' + str(idt + 1)
        target_str = 'S' + str(idt + 1)
        args.uda = source_str + '_2_' + target_str
        info_str = '\n========================== Transfer to ' + target_str + ' =========================='
        print(info_str)
        my_log.record(info_str)
        args.log = my_log

        t1 = time.time()
        acc_sub = cross_subject(data_norm, label_norm, idt, args)
        sub_acc_all[idt] = float(acc_sub)
        duration_all[idt] = time.time() - t1
        print(f'Sub:{idt:2d}, [{duration_all[idt]:5.2f}], Acc: {sub_acc_all[idt]:.4f}')
        print('\n' + '=' * 50, np.round(float(acc_sub), 2), '=' * 50 + '\n')
    print('Sub acc: ', np.round(sub_acc_all, 3))
    print('Avg acc: ', np.round(np.mean(sub_acc_all), 3))
    print('Avg duration: ', np.round(np.mean(duration_all), 3))

    acc_sub_str = str(np.round(sub_acc_all, 3).tolist())
    acc_mean_str = str(np.round(np.mean(sub_acc_all), 3).tolist())
    args.log.record("\n==========================================")
    args.log.record(acc_sub_str)
    args.log.record(acc_mean_str)

    # record sub acc to csv
    # args.file_str = os.path.basename(__file__).split('.')[0]
    # csv_log = CsvRecord(args)
    # csv_log.init()
    # csv_log.record(sub_acc_all)

    # SEED-IV MSMDA
    # [59.69, 45.12, 46.30, 38.78, 46.30, 26.44, 58.05, 44.77, 46.53, 56.17, 56.64, 56.99, 45.71, 20.92, 42.42]
    # mean: 46.06 std:  10.80

    # SEED-IV MFSAN
    # [41.48, 23.85, 52.17, 30.79, 21.15, 22.68, 45.59, 30.67, 46.89, 61.34, 44.65, 41.48, 49.47, 46.53, 43.83]
    # Avg acc 40.17 std:  11.38

    # SEED MFSAN
    # [66.59, 50.82, 34.47, 60.93, 49.12, 46.88, 64.41, 53.68, 55.72, 45.11, 65.88, 55.42, 58.37, 50.62, 58.72]
    # Avg acc:  54.45 std:  8.39

    # MI2-2 MFSAN Net_CFE
    # [83.433 58.234 94.246 74.504 57.54  67.46  63.69  91.964 78.869]
    # Avg acc:  74.438

    # MI2-2 MFSAN Net_ln2
    # [85.119 53.968 94.246 71.825 56.845 69.841 61.905 93.254 72.619]
    # Avg acc:  73.291
