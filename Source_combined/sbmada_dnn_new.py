# -*- coding: utf-8 -*-
# @Time    : 2021/12/20 13:58
# @Author  : wenzhang
# @File    : target_adapt_dnn.py
import numpy as np
import argparse
from torch.autograd import Variable
import os
from os import walk
import torch as tr
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat

from utils import network
from utils.CsvRecord import CsvRecord
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_combine_tar, read_seed_combine_tar, read_seizure_combine_sbmada, read_seizure_combine_srcbest_get_1src
from utils.utils import lr_scheduler_full, fix_random_seed, cal_acc_comb, data_loader
from utils.loss import lmmd


def calculate_sbloss(args):
    # src best loss
    X_src, y_src, X_tar, y_tar, X_sb, y_sb, trade_off = read_seizure_combine_sbmada(args)
    dset_loaders_sb = data_loader(X_src, y_src, X_sb, y_sb, args)
    if args.bottleneck == 50:
        netF, netC = network.backbone_net(args, 100, return_type='xy')
    if args.bottleneck == 64:
        netF, netC = network.backbone_net(args, 128, return_type='xy')
    base_network = nn.Sequential(netF, netC)

    sb_loss = 0.0
    for x in range(37):
        X_1src, y_1src = read_seizure_combine_srcbest_get_1src(args, j=x)
        dset_loaders_src1 = data_loader(X_1src, y_1src, None, None, args)
        max_iter1 = args.max_epoch * len(dset_loaders_src1["source"])
        iter_num1 = 0
        sblmmd = 0.0
        while iter_num1 < max_iter1:
            try:
                inputs_src1, labels_src1 = src1_loader_iter.next()
            except:
                src1_loader_iter = iter(dset_loaders_src1["source"])
                inputs_src1, labels_src1 = src1_loader_iter.next()
            try:
                inputs_sb, labels_sb = iter_sb.next()
            except:
                iter_sb = iter(dset_loaders_sb["target"])
                inputs_sb, labels_sb = iter_sb.next()
            if inputs_src1.size(0) == 1:
                continue

            iter_num1 += 1
            inputs_src1, labels_src1 = inputs_src1.cuda(), labels_src1.cuda()
            inputs_sb, labels_sb = inputs_sb.cuda(), labels_sb.cuda()
            features_src1, outputs_src1 = base_network(inputs_src1)
            features_sb, outputs_sb = base_network(inputs_sb)

            # src best loss
            sb_sub_loss = lmmd(source=features_src1, target=features_sb, s_label=labels_src1, t_label=labels_sb)
            sblmmd += sb_sub_loss
        # print('this is the src ' + str(x + 1) + ' sb_loss:', sblmmd)
    sb_loss += sblmmd
    sb_loss = sb_loss / 37.0
    return sb_loss


def train_target(args, sbloss):
    if args.data in ['SEED', 'SEED4']:
        X_src, y_src, X_tar, y_tar = read_seed_combine_tar(args)
    elif args.data in ['seizure']:
        X_src, y_src, X_tar, y_tar, X_sb, y_sb, trade_off = read_seizure_combine_sbmada(args)
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

    auc_init = 0
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
        # sb loss
        sb_loss = sbloss
        args.loss_trade_off = trade_off
        lamda = 1.0
        total_loss = classifier_loss + lamda * args.loss_trade_off * sb_loss
        optimizer_f.zero_grad()
        optimizer_c.zero_grad()
        total_loss.backward()
        optimizer_f.step()
        optimizer_c.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            base_network.eval()

            acc_s_te, sen_s_te, spec_s_te, auc_s_te = cal_acc_comb(dset_loaders["source_te"], base_network)
            acc_t_te, sen_t_te, spec_t_te, auc_t_te = cal_acc_comb(dset_loaders["Target"], base_network)
            log_str = 'Task: {}, Iter:{}/{}; Val_auc = {:.2f}%; Val_acc = {:.2f}%; Test_acc = {:.2f}%, Test_sen = {:.2f}%, Test_spec = {:.2f}%; Test_auc = {:.2f}%'.format(
                args.task_str, iter_num, max_iter, auc_s_te, acc_s_te, acc_t_te, sen_t_te, spec_t_te, auc_t_te)
            args.log.record(log_str)
            print(log_str)
            print('loss_output: ', classifier_loss.item())
            base_network.train()

            if auc_s_te >= auc_init:
                auc_init = auc_s_te
                auc_tar_src_best = auc_t_te
                sen_tar_src_best = sen_t_te
                spec_tar_src_best = spec_t_te
                acc_tar_src_best = acc_t_te
    return auc_tar_src_best, sen_tar_src_best, spec_tar_src_best, acc_tar_src_best


def get_n_target(target_id):
    domains = next(walk('/home/zwwang/code/Source_combined/data/fts_labels/'), (None, None, []))[2]
    for i in range(len(domains)):
        tar = loadmat('/home/zwwang/code/Source_combined/data/fts_labels/' + domains[target_id])
        tar_data = tar['data']
        tar_num = tar_data.shape[0]
    return tar_num


if __name__ == '__main__':

    data_name_list = ['seizure']
    data_idx = 0
    data_name = data_name_list[data_idx]
    domain = next(walk('/home/zwwang/code/Source_combined/data/fts_labels/'), (None, None, []))[2]
    n_subject = len(domain)

    sub_auc_all = np.zeros(n_subject)
    sub_sen_all = np.zeros(n_subject)
    sub_spec_all = np.zeros(n_subject)
    sub_acc_all = np.zeros(n_subject)
    for idt in range(n_subject):
        n_tar = get_n_target(idt)
        if data_name == 'seizure': N, chn, class_num, trial_num = n_subject, 18, 2, n_tar
        if data_name == '001-2014': N, chn, class_num, trial_num = 9, 22, 4, 288  # 001-2014
        if data_name == '001-2014_2': N, chn, class_num, trial_num = 9, 22, 2, 144  # 001-2014_2
        if data_name == 'SEED': N, chn, class_num, trial_num = 15, 62, 3, 3394
        if data_name == 'SEED4': N, chn, class_num, trial_num = 15, 62, 4, 851

        args = argparse.Namespace(bottleneck=50, lr=0.001, lr_decay1=0.1, lr_decay2=1.0,
                                  epsilon=1e-05, layer='wn', cov_type='oas', trial=trial_num,
                                  N=n_subject, chn=chn, class_num=class_num, smooth=0, index=idt)

        args.data = data_name
        args.app = 'sbmada'
        args.method = 'DNN'
        args.backbone = 'Net_ln2'
        if args.data in ['SEED', 'SEED4', 'seizure']:
            args.batch_size = 128  # 32
            args.max_epoch = 10  # 10
            args.input_dim = 1332  # fts dimension
            args.norm = 'zscore'  # TODO
            args.validation = 'random'
        else:
            args.batch_size = 32  # 8 in MI but 32 in seizure
            args.max_epoch = 10  # 10
            args.input_dim = int(args.chn * (args.chn + 1) / 2)  # 253 in MI, but 171 in seizure
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
        # calculate sb loss
        sb_loss = calculate_sbloss(args)
        sub_auc_all[idt], sub_sen_all[idt], sub_spec_all[idt], sub_acc_all[idt] = train_target(args, sbloss=sb_loss)

    print('Sub auc: ', np.round(sub_auc_all, 3))
    print('Sub sen: ', np.round(sub_sen_all, 3))
    print('Sub spec: ', np.round(sub_spec_all, 3))
    print('Sub acc: ', np.round(sub_acc_all, 3))
    print('Avg auc: ', np.round(np.mean(sub_auc_all), 3))
    print('Avg sen: ', np.round(np.mean(sub_sen_all), 3))
    print('Avg spec: ', np.round(np.mean(sub_spec_all), 3))
    print('Avg acc: ', np.round(np.mean(sub_acc_all), 3))

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
