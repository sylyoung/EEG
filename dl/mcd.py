# -*- coding: utf-8 -*-
# @Time    : 2023/04/28
# @Author  : Siyang Li
# @File    : mcd.py
import numpy as np
import argparse
import os
import torch as tr
import torch.nn as nn
import torch.optim as optim
import os.path as osp
import pandas as pd
import torch.nn.functional as F

from utils import network, loss
from utils.CsvRecord import CsvRecord
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_combine_tar, read_seed_combine_tar
from utils.utils import lr_scheduler_full, fix_random_seed, cal_acc_comb, data_loader, cal_bca_comb
from utils.loss import CELabelSmooth_raw, CDANE, Entropy, RandomLayer
from utils.network import calc_coeff
from utils.loss import ClassConfusionLoss

import gc
import torch


def train_target(args):
    X_src, y_src, X_tar, y_tar = read_mi_combine_tar(args)
    dset_loaders = data_loader(X_src, y_src, X_tar, y_tar, args)

    netF, netC1 = network.backbone_net(args, return_type='xy')
    _, netC2 = network.backbone_net(args, return_type='xy')
    netF, netC1, netC2 = netF.cuda(), netC1.cuda(), netC2.cuda()
    base_network = nn.Sequential(netF, netC1)  # randomly choose one classifier for prediction

    args.max_iter = args.max_epoch * len(dset_loaders["source"])

    criterion = nn.CrossEntropyLoss()
    if args.paradigm == 'ERP':
        loss_weights = []
        ar_unique, cnts_class = np.unique(y_src, return_counts=True)
        print("labels:", ar_unique)
        print("Counts:", cnts_class)
        loss_weights.append(1.0)
        loss_weights.append(cnts_class[0] / cnts_class[1])
        print(loss_weights)
        loss_weights = torch.Tensor(loss_weights).cuda()
        criterion = nn.CrossEntropyLoss(weight=loss_weights)

    optimizer_f = optim.Adam(netF.parameters(), lr=args.lr)
    optimizer_c = optim.Adam(list(netC1.parameters()) + list(netC2.parameters()), lr=args.lr)

    max_iter = args.max_epoch * len(dset_loaders["source"])
    interval_iter = max_iter // args.max_epoch
    args.max_iter = max_iter
    iter_num = 0
    netF.train()
    netC1.train()
    netC2.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = iter_source.next()
        except:
            iter_source = iter(dset_loaders["source"])
            inputs_source, labels_source = iter_source.next()

        try:
            inputs_target, _ = iter_target.next()
        except:
            iter_target = iter(dset_loaders["target"])
            inputs_target, _ = iter_target.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1

        inputs_source, inputs_source, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()

        x = torch.cat((inputs_source, inputs_source), dim=0)
        assert x.requires_grad is False

        args.trade_off = 1
        args.trade_off_entropy = 0.03

        # Step A train all networks to minimize loss on source domain
        optimizer_f.zero_grad()
        optimizer_c.zero_grad()

        g = netF(x)
        _, y_1 = netC1(g)
        _, y_2 = netC2(g)
        y1_s, y1_t = y_1.chunk(2, dim=0)
        y2_s, y2_t = y_2.chunk(2, dim=0)

        y1_t, y2_t = F.softmax(y1_t, dim=1), F.softmax(y2_t, dim=1)
        loss = F.cross_entropy(y1_s, labels_source) + F.cross_entropy(y2_s, labels_source) + \
               (-torch.mean(torch.log(torch.mean(y1_t, 0) + 1e-6)) + -torch.mean(torch.log(torch.mean(y2_t, 0) + 1e-6))) * args.trade_off_entropy
        loss.backward()
        optimizer_f.step()
        optimizer_c.step()

        # Step B train classifier to maximize discrepancy
        optimizer_f.zero_grad()
        optimizer_c.zero_grad()

        g = netF(x)
        _, y_1 = netC1(g)
        _, y_2 = netC2(g)
        y1_s, y1_t = y_1.chunk(2, dim=0)
        y2_s, y2_t = y_2.chunk(2, dim=0)
        y1_t, y2_t = F.softmax(y1_t, dim=1), F.softmax(y2_t, dim=1)
        loss = F.cross_entropy(y1_s, labels_source) + F.cross_entropy(y2_s, labels_source) + \
               (-torch.mean(torch.log(torch.mean(y1_t, 0) + 1e-6)) + -torch.mean(torch.log(torch.mean(y2_t, 0) + 1e-6))) * args.trade_off_entropy - \
               torch.mean(torch.abs(y1_t - y2_t)) * args.trade_off
        #print('loss:', loss, ', ', F.cross_entropy(y1_s, labels_source), F.cross_entropy(y2_s, labels_source), -torch.mean(torch.log(torch.mean(y1_t, 0) + 1e-6)), -torch.mean(torch.log(torch.mean(y2_t, 0) + 1e-6)),  torch.mean(torch.abs(y1_t - y2_t)))
        loss.backward()
        optimizer_f.step()

        # Step C train genrator to minimize discrepancy
        for k in range(args.class_num):
            optimizer_f.zero_grad()
            g = netF(x)
            _, y_1 = netC1(g)
            _, y_2 = netC2(g)
            y1_s, y1_t = y_1.chunk(2, dim=0)
            y2_s, y2_t = y_2.chunk(2, dim=0)
            y1_t, y2_t = F.softmax(y1_t, dim=1), F.softmax(y2_t, dim=1)
            mcd_loss = torch.mean(torch.abs(y1_t - y2_t)) * args.trade_off
            #print('mcd loss:', mcd_loss)
            mcd_loss.backward()
            optimizer_f.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            base_network.eval()

            if args.paradigm == 'MI':
                acc_t_te, _ = cal_acc_comb(dset_loaders["Target"], base_network)
                log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%'.format(args.task_str, int(iter_num // len(dset_loaders["source"])), int(max_iter // len(dset_loaders["source"])), acc_t_te)
                args.log.record(log_str)
                print(log_str)
            elif args.paradigm == 'ERP':
                acc_t_te, _ = cal_bca_comb(dset_loaders["Target"], base_network)
                log_str = 'Task: {}, Iter:{}/{}; BCA = {:.2f}%'.format(args.task_str, int(iter_num // len(dset_loaders["source"])), int(max_iter // len(dset_loaders["source"])), acc_t_te)
                args.log.record(log_str)
                print(log_str)

            base_network.train()

    gc.collect()
    torch.cuda.empty_cache()

    return acc_t_te


if __name__ == '__main__':

    data_name_list = ['BNCI2014001', 'BNCI2014002', 'BNCI2015001']
    # data_name_list = ['BNCI2014008', 'BNCI2014009', 'BNCI2015003']
    #data_name_list = ['BNCI2014001', 'BNCI2014002', 'BNCI2015001', 'BNCI2014008', 'BNCI2014009', 'BNCI2015003']

    dct = pd.DataFrame(
        columns=['dataset', 'avg', 'std', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11',
                 's12', 's13'])

    for data_name in data_name_list:

        if data_name == 'BNCI2014001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 2, 1001, 250, 144, 248
        if data_name == 'BNCI2014002': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 14, 15, 2, 2561, 512, 100, 640
        if data_name == 'MI1': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 7, 59, 2, 300, 200, 100, 72
        if data_name == 'BNCI2015001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 12, 13, 2, 2561, 512, 200, 640
        if data_name == 'BNCI2014008': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'ERP', 8, 8, 2, 206, 256, 4200, 48
        if data_name == 'BNCI2014009': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'ERP', 10, 16, 2, 206, 256, 1728, 48
        if data_name == 'BNCI2015003': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'ERP', 10, 8, 2, 206, 256, 2520, 48

        args = argparse.Namespace(feature_deep_dim=feature_deep_dim, lr=0.001, trial_num=trial_num, layer='wn',
                                  time_sample_num=time_sample_num, sample_rate=sample_rate,
                                  N=N, chn=chn, class_num=class_num, smooth=0, paradigm=paradigm)

        args.data = data_name
        args.method = 'MCD'
        args.backbone = 'EEGNet'
        args.feature = False

        import sys

        align = sys.argv[1]
        if align == 'True':
            args.align = True
        elif align == 'False':
            args.align = False

        args.batch_size = 32
        args.max_epoch = 100
        args.validation = 'None'
        args.eval_epoch = args.max_epoch / 10

        device_id = str(sys.argv[2])
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        args.data_env = 'gpu' if tr.cuda.device_count() != 0 else 'local'
        total_acc = []

        for s in [1, 2, 3, 4, 5]:
            args.SEED = s

            fix_random_seed(args.SEED)
            tr.backends.cudnn.deterministic = True

            args.data = data_name
            print(args.data)
            print(args.method)
            print(args.SEED)
            print(args)

            args.local_dir = './data/' + str(data_name) + '/'
            args.result_dir = './logs/'
            my_log = LogRecord(args)
            my_log.log_init()
            my_log.record('=' * 50 + '\n' + os.path.basename(__file__) + '\n' + '=' * 50)

            sub_acc_all = np.zeros(N)
            for idt in range(N):
                args.idt = idt
                source_str = 'Except_S' + str(idt + 1)
                target_str = 'S' + str(idt + 1)
                args.task_str = source_str + '_2_' + target_str
                info_str = '\n========================== Transfer to ' + target_str + ' =========================='
                print(info_str)
                my_log.record(info_str)
                args.log = my_log

                args.src = ['S' + str(i + 1) for i in range(N)]
                args.src.remove(target_str)
                #args.output_dir_src = osp.join(args.output_src, source_str)

                sub_acc_all[idt] = train_target(args)
            print('Sub acc: ', np.round(sub_acc_all, 3))
            print('Avg acc: ', np.round(np.mean(sub_acc_all), 3))
            total_acc.append(sub_acc_all)

            acc_sub_str = str(np.round(sub_acc_all, 3).tolist())
            acc_mean_str = str(np.round(np.mean(sub_acc_all), 3).tolist())
            args.log.record("\n==========================================")
            args.log.record(acc_sub_str)
            args.log.record(acc_mean_str)

            # record sub acc to csv
            #args.file_str = os.path.basename(__file__).split('.')[0]
            #csv_log = CsvRecord(args)
            #csv_log.init()
            #csv_log.record(sub_acc_all)

        args.log.record('\n' + '#' * 20 + 'final results' + '#' * 20)

        print(str(total_acc))

        args.log.record(str(total_acc))

        subject_mean = np.round(np.average(total_acc, axis=0), 5)
        total_mean = np.round(np.average(np.average(total_acc)), 5)
        total_std = np.round(np.std(np.average(total_acc, axis=1)), 5)

        print(subject_mean)
        print(total_mean)
        print(total_std)

        args.log.record(str(subject_mean))
        args.log.record(str(total_mean))
        args.log.record(str(total_std))

        result_dct = {'dataset': data_name, 'avg': total_mean, 'std': total_std}
        for i in range(len(subject_mean)):
            result_dct['s' + str(i)] = subject_mean[i]

        dct = dct.append(result_dct, ignore_index=True)

    dct.to_csv('./logs/' + str(args.method) + ".csv")