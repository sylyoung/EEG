# -*- coding: utf-8 -*-
# @Time    : 2021/12/22 13:58
# @Author  : wenzhang
# @File    : source_training_combine.py
import numpy as np
import argparse
import os
from os import walk
import torch as tr
import torch.nn as nn
import torch.optim as optim
import os.path as osp
from scipy.io import loadmat

from utils import network, loss
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_combine_tar, read_seed_combine_tar, read_ch_combine_tar
from utils.utils import lr_scheduler_full, fix_random_seed, cal_acc_comb, create_folder, data_loader


def train_source(args):
    if args.data in ['SEED', 'SEED4']:
        X_src, y_src, _, _ = read_seed_combine_tar(args)
    elif args.data in ['seizure']:
        X_src, y_src, X_tar, y_tar = read_ch_combine_tar(args)
    else:
        X_src, y_src, _, _ = read_mi_combine_tar(args)
    # print(X_src.shape, type(X_src))

    dset_loaders = data_loader(X_src, y_src, None, None, args)
    class_num = args.class_num
    if args.bottleneck == 50:
        netF, netC = network.backbone_net(args, 100, return_type='xy')
    if args.bottleneck == 64:
        netF, netC = network.backbone_net(args, 128, return_type='xy')

    base_network = nn.Sequential(netF, netC)

    optimizer_f = optim.SGD(netF.parameters(), lr=args.lr * 0.1)
    optimizer_c = optim.SGD(netC.parameters(), lr=args.lr)

    auc_init = 0
    max_len = len(dset_loaders["source_tr"])
    args.max_iter = args.max_epoch * max_len

    for iter_num in range(1, args.max_iter + 1):
        try:
            inputs_source, labels_source = source_loader_iter.next()
        except:
            source_loader_iter = iter(dset_loaders["source_tr"])
            inputs_source, labels_source = source_loader_iter.next()

        if inputs_source.size(0) == 1:
            continue

        base_network.train()
        lr_scheduler_full(optimizer_f, init_lr=args.lr * 0.1, iter_num=iter_num, max_iter=args.max_iter)
        lr_scheduler_full(optimizer_c, init_lr=args.lr, iter_num=iter_num, max_iter=args.max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        features_source, outputs_source = base_network(inputs_source)
        classifier_loss = loss.CELabelSmooth_raw(reduction='none', num_classes=class_num, epsilon=args.smooth)(
            outputs_source, labels_source)
        # class_weight = tr.tensor([1., 3.406], dtype=tr.float32).cuda()  # class imbalance
        # classifier_loss = nn.CrossEntropyLoss(weight=class_weight)(outputs_source, labels_source)
        optimizer_f.zero_grad()
        optimizer_c.zero_grad()
        classifier_loss.backward()
        optimizer_f.step()
        optimizer_c.step()

        if iter_num == args.max_iter:
            base_network.eval()

            acc_s_te, sen_s_te, spec_s_te, auc_s_te = cal_acc_comb(dset_loaders["source_te"], base_network)
            if auc_s_te >= auc_init:
                auc_init = auc_s_te
                best_netF = netF.state_dict()
                best_netC = netC.state_dict()

            epoch_num = iter_num // max_len
            log_str = 'Task: {}, Epoch:{}/{}; Auc = {:.3f}%; '.format(args.name_src, epoch_num, args.max_epoch,
                                                                      auc_s_te)
            args.log.record(log_str)
            print(log_str)

    tr.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
    tr.save(best_netC, osp.join(args.output_dir_src, "source_C.pt"))

    return auc_s_te, sen_s_te, spec_s_te, acc_s_te  # modify the output


def get_n_src(target_id):
    domains = next(walk('/home/zwwang/code/Source_combined/data/child_hos/fts_labels/'), (None, None, []))[2]
    i = target_id
    src_x = []
    src_y = []
    for j in range(len(domains)):
        if i != j:
            src = loadmat('/home/zwwang/code/Source_combined/data/child_hos/fts_labels/' + domains[j])
            src1 = src['label']
            src_y.append(src1)
    src_label = np.concatenate(src_y, axis=1).squeeze()
    src_num = src_label.shape[0]
    return src_num


if __name__ == '__main__':

    # data_name_list = ['001-2014', '001-2014_2', 'SEED', 'SEED4']
    data_name_list = ['seizure']
    data_idx = 0
    data_name = data_name_list[data_idx]
    domain = next(walk('/home/zwwang/code/Source_combined/data/child_hos/fts_labels/'), (None, None, []))[2]
    n_subject = len(domain)

    sub_auc_all = np.zeros(n_subject)
    sub_sen_all = np.zeros(n_subject)
    sub_spec_all = np.zeros(n_subject)
    sub_acc_all = np.zeros(n_subject)
    for idt in range(n_subject):
        n_src = get_n_src(idt)
        if data_name == 'seizure': N, chn, class_num, trial_num = n_subject, 18, 2, n_src
        if data_name == '001-2014': N, chn, class_num, trial_num = 9, 22, 4, 288  # 001-2014
        if data_name == '001-2014_2': N, chn, class_num, trial_num = 9, 22, 2, 144  # 001-2014_2
        if data_name == 'SEED': N, chn, class_num, trial_num = 15, 62, 3, 3394  # seed
        if data_name == 'SEED4': N, chn, class_num, trial_num = 15, 62, 4, 851  # seed4

        # combine不加smooth效果更好，但是对于shot，遵守其论文设置
        args = argparse.Namespace(bottleneck=50, lr=0.01, lr_decay1=0.1, lr_decay2=1.0,
                                  epsilon=1e-05, layer='wn', cov_type='oas', trial=trial_num,
                                  N=N, smooth=0.1, chn=chn, class_num=class_num)

        args.data = data_name
        args.app = 'src_comb'
        args.method = 'combine'
        args.backbone = 'Net_ln2'
        if args.data in ['SEED', 'SEED4', 'seizure']:
            args.batch_size = 4  # 32
            args.max_epoch = 10  # 10
            args.input_dim = 1332
            args.norm = 'zscore'
            args.validation = 'random'
        else:
            args.batch_size = 8  # 8
            args.max_epoch = 10  # 10
            args.input_dim = int(args.chn * (args.chn + 1) / 2)
            args.validation = 'last'

        # 这个参数设置也会影响结果，很奇怪
        args.eval_epoch = args.max_epoch / 10  # 改变epoch观测间隔

        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        args.data_env = 'gpu' if tr.cuda.device_count() != 0 else 'local'
        args.SEED = 2020
        fix_random_seed(args.SEED)
        tr.backends.cudnn.deterministic = True

        args.output = 'ckps/' + args.data + '/source/'
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
        info_str = '\n========================== Within subject ' + source_str + ' =========================='
        print(info_str)
        my_log.record(info_str)
        args.log = my_log

        args.name_src = source_str
        args.output_dir_src = osp.join(args.output, args.name_src)
        create_folder(args.output_dir_src, args.data_env, args.local_dir)

        sub_auc_all[idt], sub_sen_all[idt], sub_spec_all[idt], sub_acc_all[idt] = train_source(args)
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
