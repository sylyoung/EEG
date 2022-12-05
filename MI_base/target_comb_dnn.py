# -*- coding: utf-8 -*-
# @Time    : 2021/12/20 13:58
# @Author  : wenzhang
# @File    : target_adapt_dnn.py
import numpy as np
import argparse
import os
import torch as tr
import torch.nn as nn
import torch.optim as optim

from utils import network
from utils.CsvRecord import CsvRecord
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_combine_tar, read_seed_combine_tar
from utils.utils import lr_scheduler_full, fix_random_seed, cal_acc_comb, data_loader


def train_target(args):
    if args.data in ['SEED', 'SEED4']:
        X_src, y_src, X_tar, y_tar = read_seed_combine_tar(args)
    else:
        X_src, y_src, X_tar, y_tar = read_mi_combine_tar(args)

    dset_loaders = data_loader(X_src, y_src, X_tar, y_tar, args)

    if args.bottleneck == 50:
        netF, netC = network.backbone_net(args, 100, return_type='xy')
    if args.bottleneck == 64:
        netF, netC = network.backbone_net(args, 128, return_type='xy')
    base_network = nn.Sequential(netF, netC)

    optimizer_f = optim.SGD(netF.parameters(), lr=args.lr * 0.1)
    optimizer_c = optim.SGD(netC.parameters(), lr=args.lr)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    args.max_iter = max_iter
    iter_num = 0
    base_network.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = source_loader_iter.next()
        except:
            source_loader_iter = iter(dset_loaders["source"])
            inputs_source, labels_source = source_loader_iter.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler_full(optimizer_f, init_lr=args.lr * 0.1, iter_num=iter_num, max_iter=args.max_iter)
        lr_scheduler_full(optimizer_c, init_lr=args.lr, iter_num=iter_num, max_iter=args.max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        features_source, outputs_source = base_network(inputs_source)

        # CE loss
        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)

        optimizer_f.zero_grad()
        optimizer_c.zero_grad()
        classifier_loss.backward()
        optimizer_f.step()
        optimizer_c.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            base_network.eval()

            acc_s_te = cal_acc_comb(dset_loaders["source_te"], base_network)
            acc_t_te = cal_acc_comb(dset_loaders["Target"], base_network)
            log_str = 'Task: {}, Iter:{}/{}; Val_acc = {:.2f}%; Test_Acc = {:.2f}%'.format(args.task_str, iter_num,
                                                                                           max_iter, acc_s_te, acc_t_te)
            args.log.record(log_str)
            print(log_str)
            base_network.train()

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                acc_tar_src_best = acc_t_te

    return acc_tar_src_best


if __name__ == '__main__':

    data_name_list = ['001-2014', '001-2014_2', 'SEED', 'SEED4']
    # data_idx = 1

    for dt in range(2, 3):
        data_idx = dt

        data_name = data_name_list[data_idx]
        if data_name == '001-2014': N, chn, class_num, trial_num = 9, 22, 4, 288  # 001-2014
        if data_name == '001-2014_2': N, chn, class_num, trial_num = 9, 22, 2, 144  # 001-2014_2
        if data_name == 'SEED': N, chn, class_num, trial_num = 15, 62, 3, 3394
        if data_name == 'SEED4': N, chn, class_num, trial_num = 15, 62, 4, 851

        args = argparse.Namespace(bottleneck=50, lr=0.01, lr_decay1=0.1, lr_decay2=1.0,
                                  epsilon=1e-05, layer='wn', cov_type='oas', trial=trial_num,
                                  N=N, chn=chn, class_num=class_num, smooth=0)

        args.data = data_name
        args.method = 'DNN'
        args.backbone = 'Net_ln2'
        if args.data in ['SEED', 'SEED4']:
            args.batch_size = 32  # 32
            args.max_epoch = 10  # 10
            args.input_dim = 310
            args.norm = 'zscore'
            args.validation = 'random'
        else:
            args.batch_size = 8  # 8
            args.max_epoch = 10  # 10
            args.input_dim = int(args.chn * (args.chn + 1) / 2)
            args.validation = 'last'

        args.eval_epoch = args.max_epoch / 10

        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        args.data_env = 'gpu' if tr.cuda.device_count() != 0 else 'local'
        args.SEED = 2020
        fix_random_seed(args.SEED)
        tr.backends.cudnn.deterministic = True

        args.data = data_name
        print(args.data)
        print(args.method)
        print(args)

        args.local_dir = r'/Users/wenz/code/PyPrj/TL/TL_BCI/MSDT/'
        args.result_dir = 'results/target/'
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
            sub_acc_all[idt] = train_target(args)
        print('Sub acc: ', np.round(sub_acc_all, 3))
        print('Avg acc: ', np.round(np.mean(sub_acc_all), 3))

        acc_sub_str = str(np.round(sub_acc_all, 3).tolist())
        acc_mean_str = str(np.round(np.mean(sub_acc_all), 3).tolist())
        args.log.record("\n==========================================")
        args.log.record(acc_sub_str)
        args.log.record(acc_mean_str)

        # record sub acc to csv
        args.file_str = os.path.basename(__file__).split('.')[0]
        csv_log = CsvRecord(args)
        csv_log.init()
        csv_log.record(sub_acc_all)

        # batch size太大了不好，8>4>16
        # 数据对齐操作对于MSDA几乎没有啥效果，可以不用再提了
        # 对于数据combine训练的模型是有效果的

        # SEED，source only, LN2最好，必须加zscore，不然结果很差
        # Sub acc:  [72.717 70.949 72.481 82.94  69.711 71.803 77.431 70.566 68.945 72.746
        #  76.724 62.375 67.089 71.656 77.549]
        # Avg acc:  72.379
