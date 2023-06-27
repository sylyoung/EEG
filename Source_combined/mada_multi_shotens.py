# -*- coding: utf-8 -*-
# @Time    : 2021/12/19 20:58
# @Author  : wenzhang
# @File    : target_adapt_multiple.py
import numpy as np
import pandas as pd
import argparse
import random
import time
import os
from os import walk
from scipy.io import loadmat
import torch as tr
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.onnx
import os.path as osp
from utils import network, loss
from utils.CsvRecord import CsvRecord
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_test, read_seed_test, read_seizure_test
from utils.utils import fix_random_seed, op_copy, cal_acc_multi
from utils.func_utils import update_shot_ens


def data_load(X, y, args):
    dset_loaders = {}
    train_bs = args.batch_size

    sample_idx = tr.from_numpy(np.arange(len(y))).long()
    data_tar = Data.TensorDataset(X, y, sample_idx)

    dset_loaders["target"] = Data.DataLoader(data_tar, batch_size=train_bs, shuffle=True)
    dset_loaders["Target"] = Data.DataLoader(data_tar, batch_size=train_bs * 3, shuffle=False)
    return dset_loaders


def estimate_source_weight(data_tar, netF_list, netC_list, args):
    loss_all = tr.zeros(len(args.src), )
    for s in range(len(args.src)):
        im_loss = tr.zeros(len(data_tar), )
        gentropy_loss = tr.zeros(len(data_tar), )
        loss = tr.zeros(len(data_tar), )
        for n in range(len(data_tar)):
            batch_tar = data_tar[n]  # TODO
            features_test = netF_list[s](batch_tar)
            outputs_test = netC_list[s](features_test)

            softmax_out = nn.Softmax(dim=1)(outputs_test)
            im_loss[n] = tr.mean(loss.Entropy(softmax_out))
            msoftmax = softmax_out.mean(dim=0)
            gentropy_loss[n] = tr.sum(-msoftmax * tr.log(msoftmax + args.epsilon))
            loss[n] = im_loss[n] - gentropy_loss[n]
            loss_all[s] += loss[n]

    weights_domain = loss_all / tr.sum(loss_all)

    return weights_domain

def train_target(args):
    if args.data in ['SEED', 'SEED4']:
        X_tar, y_tar = read_seed_test(args)
    elif args.data in ['seizure']:
        X_tar, y_tar = read_seizure_test(args)
    else:
        X_tar, y_tar = read_mi_test(args)
    dset_loaders = data_load(X_tar, y_tar, args)
    num_src = len(args.src)

    # base network feature extract
    netF_list, netC_list = [], []
    for i in range(num_src):
        if args.bottleneck == 50:
            netF, netC = network.backbone_net(args, 100, return_type='y')
        if args.bottleneck == 64:
            netF, netC = network.backbone_net(args, 128, return_type='y')
        netF_list.append(netF)
        netC_list.append(netC)

    param_group = []
    for i in range(num_src):
        modelpath = args.output_dir_src[i] + '/source_F.pt'
        netF_list[i].load_state_dict(tr.load(modelpath))
        netF_list[i].eval()
        modelpath = args.output_dir_src[i] + '/source_C.pt'
        netC_list[i].load_state_dict(tr.load(modelpath))
        netC_list[i].eval()

        for k, v in netF_list[i].named_parameters():
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        for k, v in netC_list[i].named_parameters():
            v.requires_grad = False

    ###################################################################################
    info_loss = loss.InformationMaximizationLoss().cuda()

    param_single = []
    for k, v in netF_list[i].named_parameters():
        param_single += [{'params': v, 'lr': args.lr * args.lr_decay1}]
    optimizer = optim.SGD(param_single)
    optimizer = op_copy(optimizer)

    t1 = time.time()
    for i in range(len(netF_list)):
        for epoch in range(1, args.max_epoch + 1):
            update_shot_ens(dset_loaders, netF_list[i], netC_list[i], optimizer, info_loss, args)
    acc_t_te, sen_t_te, spec_t_te, auc_t_te = cal_acc_multi(dset_loaders['Target'], netF_list, netC_list, args)
    # weight_src = estimate_source_weight_epoch(dset_loaders['Target'], netF_list, netC_list, args)
    duration = time.time() - t1
    log_str = f'Epoch:{epoch:2d}/{args.max_epoch:2d} [{duration:5.2f}], acc: {acc_t_te:.4f}, sen: {sen_t_te:.4f}, spec: {spec_t_te:.4f}, auc: {auc_t_te:.4f}'
    args.log.record(log_str)
    print(log_str)

    return auc_t_te, sen_t_te, spec_t_te, acc_t_te


def get_n_target(target_id):
    domains = next(walk('/home/zwwang/code/Source_combined/data/fts_labels/'), (None, None, []))[2]
    for i in range(len(domains)):
        tar = loadmat('/home/zwwang/code/Source_combined/data/fts_labels/' + domains[target_id])
        tar_data = tar['data']
        tar_num = tar_data.shape[0]
    return tar_num


if __name__ == '__main__':

    # 这个版本还需要简化，建议后边把MSDF独立出来进行训练和参数策略重制定
    data_name_list = ['seizure']
    data_idx = 0
    data_name = data_name_list[data_idx]
    domain = next(walk('/home/zwwang/code/Source_combined/data/fts_labels/'), (None, None, []))[2]
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

        args = argparse.Namespace(bottleneck=50, lr=0.01, lr_decay1=0.1, lr_decay2=1.0,
                                  cls_par=0.3, ent_par=1.0, epsilon=1e-05, layer='wn',
                                  N=N, chn=chn, class_num=class_num, cov_type='oas', trial=trial_num,
                                  use_weight=0, tempt=1, eps_th=0.9)

        args.data = data_name
        args.app = 'mada'
        args.method = 'shot-ens'
        args.backbone = 'Net_ln2'
        if args.data in ['SEED', 'SEED4', 'seizure']:
            args.batch_size = 32  # 32
            args.max_epoch = 10  # 10
            args.input_dim = 1332
            args.norm = 'zscore'
            args.validation = 'random'
        else:
            args.batch_size = 4  # 4
            args.max_epoch = 10  # 10
            args.input_dim = int(args.chn * (args.chn + 1) / 2)
            args.validation = 'last'

        if args.method == 'shot-ens': args.use_weight = 0

        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        args.data_env = 'gpu' if tr.cuda.device_count() != 0 else 'local'
        args.SEED = 2020
        fix_random_seed(args.SEED)
        tr.backends.cudnn.deterministic = True

        args.data = data_name
        args.output_src = 'ckps/' + args.data + '/source/'
        print(args.data)
        print(args.method)
        print(args)

        args.local_dir = r'C:/wzw/研一下/0_MSDT/Source_combined/'
        args.result_dir = 'results/target/'
        my_log = LogRecord(args)
        my_log.log_init()
        my_log.record('=' * 50 + '\n' + os.path.basename(__file__) + '\n' + '=' * 50)

        args.idt = idt
        target_str = 'S' + domain[idt][1:-4]
        info_str = '\n========================== Transfer to ' + target_str + ' =========================='
        print(info_str)
        my_log.record(info_str)
        args.log = my_log

        # src selection--mada
        src_sel_pd = pd.read_excel('/home/zwwang/code/Source_combined/data/mada_max10_all_with_smote.xlsx',
                                   index_col=0)
        src_sel_numpy = src_sel_pd.to_numpy()
        tar_num = src_sel_numpy[:, 0].shape[0]
        for i in range(tar_num):
            if str(src_sel_numpy[:, 0][i]) == domain[idt][1:-4]:
                d = src_sel_numpy[i][1:31]  # 控制选择的源域数量
        args.src = ['S' + str(d[j]) for j in range(d.shape[0])]  # 选取源域的模型集成，在这里改代码
        print(args.src)

        args.output_dir_src = []
        for i in range(len(args.src)):
            args.output_dir_src.append(osp.join(args.output_src, args.src[i]))
        print(args.output_dir_src)

        t1 = time.time()
        sub_auc_all[idt], sub_sen_all[idt], sub_spec_all[idt], sub_acc_all[idt] = train_target(args)
        duration_all[idt] = time.time() - t1
        print(f'Sub:{idt:2d}, [{duration_all[idt]:5.2f}], Auc: {sub_auc_all[idt]:.4f}')
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