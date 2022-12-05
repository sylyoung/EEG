# -*- coding: utf-8 -*-
# @Time    : 2021/12/20 13:58
# @Author  : wenzhang
# @File    : target_adapt_combine_uda.py
import time

import numpy as np
import argparse
import os
import torch as tr
import torch.nn as nn
import torch.optim as optim
from utils import network, loss
from utils.CsvRecord import CsvRecord
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_combine_tar, read_seed_combine_tar
from utils.utils import lr_scheduler_full, fix_random_seed, cal_acc_comb, data_loader, cal_bca_comb
from utils.loss import ClassConfusionLoss, CELabelSmooth_raw

import gc
import torch
from imblearn.over_sampling import RandomOverSampler

def train_target(args):
    X_src, y_src, X_tar, y_tar = read_mi_combine_tar(args)
    print('X_src, y_src, X_tar, y_tar:', X_src.shape, y_src.shape, X_tar.shape, y_tar.shape)

    def apply_randup(train_x, train_y):
        shape0, shape1, shape2 = train_x.shape
        sampler = RandomOverSampler()
        print('before Random Upsampling:', train_x.shape, train_y.shape)
        train_x, train_y = sampler.fit_resample(train_x.reshape(shape0, -1), train_y)
        train_x = train_x.reshape(-1, shape1, shape2)
        print('after Random Upsampling:', train_x.shape, train_y.shape)
        return train_x, train_y

    X_src, y_src = apply_randup(X_src, y_src)
    args.trial_num = len(X_src) // (args.N - 1)

    dset_loaders = data_loader(X_src, y_src, X_tar, y_tar, args)

    netF, netC = network.backbone_net(args, return_type='xy')
    netF, netC = netF.cuda(), netC.cuda()
    base_network = nn.Sequential(netF, netC)

    criterion = nn.CrossEntropyLoss()

    optimizer_f = optim.SGD(netF.parameters(), lr=args.lr)
    optimizer_c = optim.SGD(netC.parameters(), lr=args.lr)

    max_iter = args.max_epoch * len(dset_loaders["source"])
    print('max_iter:', max_iter)
    interval_iter = max_iter // args.max_epoch
    args.max_iter = max_iter
    iter_num = 0
    base_network.train()

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
        lr_scheduler_full(optimizer_f, init_lr=args.lr, iter_num=iter_num, max_iter=args.max_iter)
        lr_scheduler_full(optimizer_c, init_lr=args.lr, iter_num=iter_num, max_iter=args.max_iter)

        features_source, outputs_source = base_network(inputs_source)
        features_target, outputs_target = base_network(inputs_target)

        # new version img loss
        # p = float(iter_num) / max_iter
        # alpha = 2. / (1. + np.exp(-10 * p)) - 1
        args.loss_trade_off = 1.0
        args.t_mcc = 2
        transfer_loss = ClassConfusionLoss(t=args.t_mcc)(outputs_target)
        classifier_loss = criterion(outputs_source, labels_source)
        total_loss = args.loss_trade_off * transfer_loss + classifier_loss

        optimizer_f.zero_grad()
        optimizer_c.zero_grad()
        total_loss.backward()
        optimizer_f.step()
        optimizer_c.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            base_network.eval()

            if args.paradigm == 'MI':
                acc_t_te = cal_acc_comb(dset_loaders["Target"], base_network)
                log_str = 'UP: Task: {}, Iter:{}/{}; Acc = {:.2f}%'.format(args.task_str, int(iter_num // len(dset_loaders["source"])), int(max_iter // len(dset_loaders["source"])), acc_t_te)
                args.log.record(log_str)
                print(log_str)
            elif args.paradigm == 'ERP':
                acc_t_te = cal_bca_comb(dset_loaders["Target"], base_network)
                log_str = 'UP: Task: {}, Iter:{}/{}; BCA = {:.2f}%'.format(args.task_str, int(iter_num // len(dset_loaders["source"])), int(max_iter // len(dset_loaders["source"])), acc_t_te)
                args.log.record(log_str)
                print(log_str)

            base_network.train()

    gc.collect()
    torch.cuda.empty_cache()

    return acc_t_te


if __name__ == '__main__':

    #data_name_list = ['BNCI2014001', 'BNCI2014002', 'MI1', 'BNCI2015001']
    # data_name_list = ['BNCI2014008', 'BNCI2014009', 'BNCI2015003']
    data_name_list = ['BNCI2015003']

    for data_name in data_name_list:

        if data_name == 'BNCI2014001': paradigm, N, chn, class_num, time_sample_num, trial_num, sample_rate, feature_deep_dim = 'MI', 9, 22, 2, 1001, 144, 250, 248
        if data_name == 'BNCI2014002': paradigm, N, chn, class_num, time_sample_num, trial_num, sample_rate, feature_deep_dim = 'MI', 14, 15, 2, 2561, 100, 512, 640
        if data_name == 'MI1': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 7, 59, 2, 300, 200, 100, 72
        if data_name == 'BNCI2015001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 12, 13, 2, 2561, 512, 200, 640
        if data_name == 'BNCI2014008': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'ERP', 8, 8, 2, 206, 256, 4200, 48
        if data_name == 'BNCI2014009': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'ERP', 10, 16, 2, 206, 256, 1728, 48
        if data_name == 'BNCI2015003': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'ERP', 10, 8, 2, 206, 256, 2520, 48

        args = argparse.Namespace(feature_deep_dim=feature_deep_dim, lr=0.0001, lr_decay1=0.1, lr_decay2=0.1,
                                  epsilon=1e-05, trial_num=trial_num,
                                  time_sample_num=time_sample_num, sample_rate=sample_rate,
                                  N=N, chn=chn, class_num=class_num, smooth=0, paradigm=paradigm)

        args.data = data_name
        args.method = 'MCC'
        args.backbone = 'EEGNet'

        import sys
        align = sys.argv[1]
        if align == 'True':
            args.align = True
        elif align == 'False':
            args.align = False

        args.batch_size = 256  # 8
        args.max_epoch = 100  # 100
        args.validation = 'last'
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
            duration_all = np.zeros(N)
            for idt in range(N):
                args.idt = idt
                source_str = 'Except_S' + str(idt + 1)
                target_str = 'S' + str(idt + 1)
                args.task_str = source_str + '_2_' + target_str
                info_str = '\n========================== Transfer to ' + target_str + ' =========================='
                print(info_str)
                my_log.record(info_str)
                args.log = my_log

                t1 = time.time()
                sub_acc_all[idt] = train_target(args)
                duration_all[idt] = time.time() - t1
                print(f'Sub:{idt:2d}, [{duration_all[idt]:5.2f}], Acc: {sub_acc_all[idt]:.4f}')
            print('Sub acc: ', np.round(sub_acc_all, 3))
            print('Avg acc: ', np.round(np.mean(sub_acc_all), 3))
            print('Avg duration: ', np.round(np.mean(duration_all), 3))
            total_acc.append(sub_acc_all)

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