# -*- coding: utf-8 -*-
# @Time    : 2021/12/20 12:00 下午
# @Author  : wenzhang
# @File    : soure_training_multiple.py
import numpy as np
import argparse
import random
import torch.nn as nn
import torch as tr
import torch.optim as optim
import torch.utils.data as Data
import os.path as osp
import os
from scipy.io import loadmat
from os import walk
from imblearn.over_sampling import SMOTE, ADASYN
from utils import network, loss
from utils.loss import CELabelSmooth
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_train, read_seed_train, read_seizure_train, obtain_train_val_source
from utils.utils import create_folder, lr_scheduler, fix_random_seed, op_copy, cal_acc


def data_load(X, y, args):
    dset_loaders = {}
    train_bs = args.batch_size
    tr.manual_seed(args.SEED)

    src_idx = np.arange(len(y.numpy()))
    num_train = int(0.9 * len(src_idx))
    tr.manual_seed(args.SEED)
    id_train, id_val = tr.utils.data.random_split(src_idx, [num_train, len(src_idx) - num_train])

    # SMOTE or ADASYN
    src_data = X[id_train, :].numpy()
    src_label = y[id_train].numpy()
    n0, n1 = 0, 0
    for n in range(src_label.shape[0]):
        if src_label[n] == 0:
            n0 += 1
        else:
            n1 += 1
    rtio = n0 / n1
    # oversample = SMOTE(random_state=42)
    # src_data, src_label = oversample.fit_resample(src_data, src_label)
    # try:
    #     oversample = SMOTE(random_state=42)
    #     src_data, src_label = oversample.fit_resample(src_data, src_label)
    # except ValueError:
    #     oversample = ADASYN(random_state=42)
    #     src_data, src_label = oversample.fit_resample(src_data, src_label)
    # n0, n1 = 0, 0
    # for n in range(src_label.shape[0]):
    #     if src_label[n] == 0:
    #         n0 += 1
    #     else:
    #         n1 += 1
    # print(n0, n1)
    # input('')
    tr_data = tr.from_numpy(src_data).float()
    tr_label = tr.from_numpy(src_label).long()
    source_tr = Data.TensorDataset(tr_data, tr_label)

    dset_loaders['source_tr'] = Data.DataLoader(source_tr, batch_size=train_bs, shuffle=True, drop_last=True)

    source_te = Data.TensorDataset(X[id_val, :], y[id_val])
    dset_loaders['source_te'] = Data.DataLoader(source_te, batch_size=train_bs * 2, shuffle=False, drop_last=True)

    return dset_loaders, rtio


def train_source(args):
    if args.data in ['SEED', 'SEED4']:
        X_src, y_src = read_seed_train(args)
    elif args.data in ['seizure']:
        X_src, y_src = read_seizure_train(args)
    else:
        X_src, y_src = read_mi_train(args)
    dset_loaders, ratio = data_load(X_src, y_src, args)

    if args.bottleneck == 50:
        netF, netC = network.backbone_net(args, 100, return_type='y')
    if args.bottleneck == 64:
        netF, netC = network.backbone_net(args, 128, return_type='y')

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate * 0.1}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    auc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0

    netF.train()
    netC.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = iter_source.next()
        except:
            iter_source = iter(dset_loaders['source_tr'])
            inputs_source, labels_source = iter_source.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()

        # print(inputs_source.shape, labels_source.shape)  # [4, 253] [4]
        labels_source = tr.zeros(args.batch_size, args.class_num).cuda().scatter_(1, labels_source.reshape(-1, 1), 1)
        outputs_source = netC(netF(inputs_source))
        # classifier_loss = CELabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source)
        class_weight = tr.tensor([1., ratio], dtype=tr.float32).cuda()  # class imbalance
        classifier_loss = nn.CrossEntropyLoss(weight=class_weight)(outputs_source, labels_source)
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netC.eval()

            acc_s_te, sen_s_te, spec_s_te, auc_s_te = cal_acc(dset_loaders['source_te'], netF, netC)
            log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%, sen = {:.2f}%, spec = {:.2f}%, Auc = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te, sen_s_te, spec_s_te, auc_s_te)
            args.log.record(log_str)
            print(log_str)
            print('loss_output: ', classifier_loss.item())

            if auc_s_te >= auc_init:  # 返回验证集上最好的auc，保存对应模型
                auc_init = auc_s_te
                best_netF = netF.state_dict()
                best_netC = netC.state_dict()

            netF.train()
            netC.train()

    tr.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
    tr.save(best_netC, osp.join(args.output_dir_src, "source_C.pt"))

    return auc_s_te, sen_s_te, spec_s_te, acc_s_te


def get_n_src(src_id):
    domains = next(walk('/home/zwwang/code/Source_combined/data/fts_labels/'), (None, None, []))[2]
    i = src_id
    src = loadmat('/home/zwwang/code/Source_combined/data/fts_labels/' + domains[i])
    src_label = src['label']
    src_num = src_label.shape[0]
    print(src_num)
    return src_num


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
    for ids in range(n_subject):
        n_src = get_n_src(ids)
        if data_name == 'seizure': N, chn, class_num, trial_num = n_subject, 18, 2, n_src
        if data_name == 'SEED': N, chn, class_num, trial_num = 15, 62, 3, 3394
        if data_name == 'SEED4': N, chn, class_num, trial_num = 15, 62, 4, 851

        args = argparse.Namespace(bottleneck=50, lr=0.01, epsilon=1e-05, layer='wn',
                                  smooth=0, chn=chn, trial=trial_num,
                                  N=N, class_num=class_num, cov_type='oas')
        args.data = data_name
        args.method = 'multiple'
        args.backbone = 'Net_ln2'
        if args.data in ['SEED', 'SEED4', 'seizure']:
            args.batch_size = 4  # 32
            args.max_epoch = 20  # 100
            args.input_dim = 1332
            args.norm = 'zscore'
            args.validation = 'random'

        os.environ["CUDA_VISIBLE_DEVICES"] = '2'
        args.data_env = 'gpu' if tr.cuda.device_count() != 0 else 'local'
        args.SEED = 2020
        fix_random_seed(args.SEED)
        tr.backends.cudnn.deterministic = True

        mdl_path = 'ckps/'
        args.output = mdl_path + args.data + '/source/'
        print(args.data)
        print(args.method)
        print(args)

        args.local_dir = r'C:/wzw/研一下/0_MSDT/Source_combined/'
        args.result_dir = 'results/target/'
        my_log = LogRecord(args)
        my_log.log_init()
        my_log.record('=' * 50 + '\n' + os.path.basename(__file__) + '\n' + '=' * 50)


        args.ids = ids
        source_str = 'S' + domain[ids][1:-4]
        info_str = '\n========================== Within subject ' + source_str + ' =========================='
        print(info_str)
        my_log.record(info_str)
        args.log = my_log

        args.name_src = source_str
        args.output_dir_src = osp.join(args.output, args.name_src)
        create_folder(args.output_dir_src, args.data_env, args.local_dir)

        sub_auc_all[ids], sub_sen_all[ids], sub_spec_all[ids], sub_acc_all[ids] = train_source(args)
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
