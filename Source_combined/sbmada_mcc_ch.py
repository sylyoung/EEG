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
from os import walk
from scipy.io import loadmat

from utils.dataloader import read_mi_combine_tar, read_seed_combine_tar, read_ch_combine_sbmada
from utils.utils import lr_scheduler_full, fix_random_seed, cal_acc_comb, data_loader
from utils.loss import ClassConfusionLoss, CELabelSmooth_raw, lmmd


def train_target(args):
    if args.data in ['SEED', 'SEED4']:
        X_src, y_src, X_tar, y_tar = read_seed_combine_tar(args)
    elif args.data in ['seizure']:
        X_src, y_src, X_tar, y_tar, X_sb, y_sb, trade_off = read_ch_combine_sbmada(args)
    else:
        X_src, y_src, X_tar, y_tar = read_mi_combine_tar(args)
    dset_loaders = data_loader(X_src, y_src, X_tar, y_tar, args)
    dset_loaders_sb = data_loader(X_src, y_src, X_sb, y_sb, args)

    if args.bottleneck == 50:
        netF, netC = network.backbone_net(args, 100, return_type='xy')
    if args.bottleneck == 64:
        netF, netC = network.backbone_net(args, 128, return_type='xy')
    base_network = nn.Sequential(netF, netC)

    max_len = max(len(dset_loaders["source"]), len(dset_loaders["target"]))
    args.max_iter = args.max_epoch * max_len
    optimizer_f = optim.SGD(netF.parameters(), lr=args.lr * 0.1)
    optimizer_c = optim.SGD(netC.parameters(), lr=args.lr)
    auc_init = 0
    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source"])
    interval_iter = max_iter // 10
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

        try:
            inputs_sb, labels_sb = iter_sb.next()
        except:
            iter_sb = iter(dset_loaders_sb["target"])
            inputs_sb, labels_sb = iter_sb.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler_full(optimizer_f, init_lr=args.lr * 0.1, iter_num=iter_num, max_iter=args.max_iter)
        lr_scheduler_full(optimizer_c, init_lr=args.lr, iter_num=iter_num, max_iter=args.max_iter)

        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
        inputs_sb, labels_sb = inputs_sb.cuda(), labels_sb.cuda()
        features_source, outputs_source = base_network(inputs_source)
        features_target, outputs_target = base_network(inputs_target)
        features_sb, outputs_sb = base_network(inputs_sb)

        # new version img loss
        # p = float(iter_num) / max_iter
        # alpha = 2. / (1. + np.exp(-10 * p)) - 1
        args.lamda1 = 0.1
        args.lamda2 = 15
        args.t_mcc = 4  # modify from 2
        transfer_loss = ClassConfusionLoss(t=args.t_mcc)(outputs_target)
        classifier_loss = CELabelSmooth_raw(num_classes=args.class_num, epsilon=args.smooth)(outputs_source,
                                                                                             labels_source)
        # class_weight = tr.tensor([1., 3], dtype=tr.float32).cuda()  # class imbalance
        # classifier_loss = nn.CrossEntropyLoss(weight=class_weight)(outputs_source, labels_source)

        # src best loss
        args.sbloss_trade_off = trade_off
        sb_loss = lmmd(source=features_source, target=features_sb, s_label=labels_source, t_label=labels_sb)

        total_loss = args.lamda1 * transfer_loss + classifier_loss + args.lamda2 * args.sbloss_trade_off * sb_loss

        optimizer_f.zero_grad()
        optimizer_c.zero_grad()
        total_loss.backward()
        optimizer_f.step()
        optimizer_c.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            base_network.eval()
            acc_s_te, sen_s_te, spec_s_te, auc_s_te = cal_acc_comb(dset_loaders["source_te"], base_network)
            acc_t_te, sen_t_te, spec_t_te, auc_t_te = cal_acc_comb(dset_loaders["Target"], base_network)
            log_str = 'Task: {}, Iter:{}/{}; acc = {:.2f}%, sen = {:.2f}%, spec = {:.2f}%, auc = {:.2f}%'.format(
                args.task_str, iter_num, max_iter, acc_t_te, sen_t_te, spec_t_te, auc_t_te)
            args.log.record(log_str)
            print(log_str)
            print(classifier_loss.item(), transfer_loss.item(), sb_loss.item())
            print('loss_output: ', total_loss.item())  # output the loss

            base_network.train()
            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                auc_tar_src_best = auc_t_te
                sen_tar_src_best = sen_t_te
                spec_tar_src_best = spec_t_te
                acc_tar_src_best = acc_t_te


    return auc_tar_src_best, sen_tar_src_best, spec_tar_src_best, acc_tar_src_best


def get_n_target(target_id):
    domains = next(walk('/home/zwwang/code/Source_combined/data/child_hos/fts_labels/'), (None, None, []))[2]
    for i in range(len(domains)):
        tar = loadmat('/home/zwwang/code/Source_combined/data/child_hos/fts_labels/' + domains[target_id])
        tar_data = tar['data']
        tar_num = tar_data.shape[0]
    return tar_num


if __name__ == '__main__':

    data_name_list = ['seizure']
    data_idx = 0
    data_name = data_name_list[data_idx]
    domain = next(walk('/home/zwwang/code/Source_combined/data/child_hos/fts_labels/'), (None, None, []))[2]
    n_subject = len(domain)

    sub_auc_all = np.zeros(n_subject)
    sub_sen_all = np.zeros(n_subject)
    sub_spec_all = np.zeros(n_subject)
    sub_acc_all = np.zeros(n_subject)
    duration_all = np.zeros(n_subject)
    for idt in range(n_subject):
        n_tar = get_n_target(idt)
        if data_name == 'seizure': N, chn, class_num, trial_num = n_subject, 18, 2, n_tar
        if data_name == '001-2014': N, chn, class_num, trial_num = 9, 22, 4, 288  # 001-2014
        if data_name == '001-2014_2': N, chn, class_num, trial_num = 9, 22, 2, 144  # 001-2014_2
        if data_name == 'SEED': N, chn, class_num, trial_num = 15, 62, 3, 3394
        if data_name == 'SEED4': N, chn, class_num, trial_num = 15, 62, 4, 851

        args = argparse.Namespace(bottleneck=50, lr=0.001, lr_decay1=0.1, lr_decay2=1.0,
                                  epsilon=1e-05, layer='wn', cov_type='oas', trial=trial_num,
                                  N=N, chn=chn, class_num=class_num, smooth=0)

        args.data = data_name
        args.app = 'sbmada'
        args.method = 'MCC'
        args.backbone = 'Net_ln2'
        if args.data in ['SEED', 'SEED4', 'seizure']:
            args.batch_size = 16  # 32
            args.max_epoch = 10  # 10
            args.input_dim = 1332
            args.norm = 'zscore'
            args.validation = 'random'
        else:
            args.batch_size = 8  # 8 对于DANN和CDAN合适的
            args.max_epoch = 10  # 10
            args.input_dim = int(args.chn * (args.chn + 1) / 2)
            args.validation = 'last'
        args.eval_epoch = args.max_epoch / 10

        os.environ["CUDA_VISIBLE_DEVICES"] = '2'
        args.data_env = 'gpu' if tr.cuda.device_count() != 0 else 'local'
        args.SEED = 2020
        fix_random_seed(args.SEED)
        tr.backends.cudnn.deterministic = True

        args.data = data_name
        print(args.data)
        print(args.method)
        print(args)

        args.local_dir = r'C:/wzw/研一下/0_MSDT/Source_combined/'
        args.result_dir = 'results/target/'
        my_log = LogRecord(args)
        my_log.log_init()
        my_log.record('=' * 50 + '\n' + os.path.basename(__file__) + '\n' + '=' * 50)

        args.idt = idt
        source_str = 'Except_S' + domain[idt][1:-4]
        target_str = 'S' + domain[idt][1:-4]
        args.task_str = source_str + '_2_' + target_str
        info_str = '\n========================== Transfer to ' + target_str + ' =========================='
        print(info_str)
        my_log.record(info_str)
        args.log = my_log

        t1 = time.time()
        sub_auc_all[idt], sub_sen_all[idt], sub_spec_all[idt], sub_acc_all[idt] = train_target(args)
        duration_all[idt] = time.time() - t1
        print(f'Sub:{idt:2d}, [{duration_all[idt]:5.2f}], Acc: {sub_auc_all[idt]:.4f}')
    print('Sub auc: ', np.round(sub_auc_all, 3))
    print('Sub sen: ', np.round(sub_sen_all, 3))
    print('Sub spec: ', np.round(sub_spec_all, 3))
    print('Sub acc: ', np.round(sub_acc_all, 3))
    print('Avg auc: ', np.round(np.mean(sub_auc_all), 3))
    print('Avg sen: ', np.round(np.mean(sub_sen_all), 3))
    print('Avg spec: ', np.round(np.mean(sub_spec_all), 3))
    print('Avg acc: ', np.round(np.mean(sub_acc_all), 3))
    print('Avg duration: ', np.round(np.mean(duration_all), 3))

    auc_sub_str = str(np.round(sub_auc_all, 3).tolist())
    auc_mean_str = str(np.round(np.mean(sub_auc_all), 3).tolist())

    sen_sub_str = str(np.round(sub_sen_all, 3).tolist())
    sen_mean_str = str(np.round(np.mean(sub_sen_all), 3).tolist())

    spec_sub_str = str(np.round(sub_spec_all, 3).tolist())
    spec_mean_str = str(np.round(np.mean(sub_spec_all), 3).tolist())

    acc_sub_str = str(np.round(sub_acc_all, 3).tolist())
    acc_mean_str = str(np.round(np.mean(sub_acc_all), 3).tolist())

    args.log.record("\n===================auc====================")
    args.log.record(auc_sub_str)
    args.log.record(auc_mean_str)
    args.log.record("\n===================sen====================")
    args.log.record(sen_sub_str)
    args.log.record(sen_mean_str)
    args.log.record("\n===================spec===================")
    args.log.record(spec_sub_str)
    args.log.record(spec_mean_str)
    args.log.record("\n===================acc====================")
    args.log.record(acc_sub_str)
    args.log.record(acc_mean_str)