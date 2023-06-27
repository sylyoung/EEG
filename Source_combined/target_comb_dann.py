# -*- coding: utf-8 -*-
# @Time    : 2021/12/20 13:58
# @Author  : wenzhang
# @File    : target_adapt_combine_uda.py
import numpy as np
import argparse
import os
import torch as tr
import torch.nn as nn
import torch.optim as optim
import os.path as osp
from os import walk
from scipy.io import loadmat
from utils import network, loss
from utils.CsvRecord import CsvRecord
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_combine_tar, read_seed_combine_tar, read_seizure_combine_tar
from utils.utils import lr_scheduler_full, fix_random_seed, cal_acc_comb, data_loader
from utils.loss import CELabelSmooth_raw, Entropy, ReverseLayerF


def train_target(args):
    if args.data in ['SEED', 'SEED4']:
        X_src, y_src, X_tar, y_tar = read_seed_combine_tar(args)
    elif args.data in ['seizure']:
        X_src, y_src, X_tar, y_tar = read_seizure_combine_tar(args)
    else:
        X_src, y_src, X_tar, y_tar = read_mi_combine_tar(args)
    dset_loaders = data_loader(X_src, y_src, X_tar, y_tar, args)

    if args.bottleneck == 50:
        netF, netC = network.backbone_net(args, 100, return_type='xy')
    if args.bottleneck == 64:
        netF, netC = network.backbone_net(args, 128, return_type='xy')
    base_network = nn.Sequential(netF, netC)

    args.max_iter = args.max_epoch * len(dset_loaders["source"])

    ad_net = network.feat_classifier(type=args.layer, class_num=2, bottleneck_dim=args.bottleneck).cuda()

    optimizer_f = optim.SGD(netF.parameters(), lr=args.lr * 0.1)
    optimizer_c = optim.SGD(netC.parameters(), lr=args.lr)
    optimizer_d = optim.SGD(ad_net.parameters(), lr=args.lr)

    auc_init = 0
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

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler_full(optimizer_f, init_lr=args.lr * 0.1, iter_num=iter_num, max_iter=args.max_iter)
        lr_scheduler_full(optimizer_c, init_lr=args.lr, iter_num=iter_num, max_iter=args.max_iter)
        lr_scheduler_full(optimizer_d, init_lr=args.lr, iter_num=iter_num, max_iter=args.max_iter)

        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
        features_source, outputs_source = base_network(inputs_source)
        features_target, outputs_target = base_network(inputs_target)

        # new version img loss
        p = float(iter_num) / max_iter
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        reverse_source, reverse_target = ReverseLayerF.apply(features_source, alpha), ReverseLayerF.apply(
            features_target,
            alpha)
        domain_output_s = ad_net(reverse_source)
        domain_output_t = ad_net(reverse_target)
        domain_label_s = tr.ones(inputs_source.size()[0]).long().cuda()
        domain_label_t = tr.zeros(inputs_target.size()[0]).long().cuda()
        # print(reverse_source.shape)  # 16*1024
        # print(domain_output_s.shape, domain_label_s.shape)  # 16*2, 16
        classifier_loss = CELabelSmooth_raw(num_classes=args.class_num, epsilon=args.smooth)(outputs_source,
                                                                                             labels_source)
        adv_loss = nn.CrossEntropyLoss()(domain_output_s, domain_label_s) + nn.CrossEntropyLoss()(domain_output_t,
                                                                                                  domain_label_t)
        total_loss = classifier_loss + adv_loss

        optimizer_f.zero_grad()
        optimizer_c.zero_grad()
        optimizer_d.zero_grad()
        total_loss.backward()
        optimizer_f.step()
        optimizer_c.step()
        optimizer_d.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            base_network.eval()
            acc_s_te, sen_s_te, spec_s_te, auc_s_te = cal_acc_comb(dset_loaders["source_te"], base_network)
            acc_t_te, sen_t_te, spec_t_te, auc_t_te = cal_acc_comb(dset_loaders["Target"], base_network)
            log_str = 'Task: {}, Iter:{}/{}; acc = {:.2f}%, sen = {:.2f}%, spec = {:.2f}%, auc = {:.2f}%'.format(
                args.task_str, iter_num, max_iter, acc_t_te, sen_t_te, spec_t_te, auc_t_te)
            args.log.record(log_str)
            print(log_str)

            base_network.train()
            if auc_s_te >= auc_init:
                auc_init = auc_s_te
                auc_tar_src_best = auc_t_te

    return auc_tar_src_best


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
        args.app = 'no'
        args.method = 'DANN'
        args.backbone = 'Net_ln2'
        if args.data in ['SEED', 'SEED4', 'seizure']:
            args.batch_size = 32  # 32
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

        os.environ["CUDA_VISIBLE_DEVICES"] = '1'
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

        sub_auc_all[idt] = train_target(args)
    print('Sub auc: ', np.round(sub_auc_all, 3))
    print('Avg auc: ', np.round(np.mean(sub_auc_all), 3))

    auc_sub_str = str(np.round(sub_auc_all, 3).tolist())
    auc_mean_str = str(np.round(np.mean(sub_auc_all), 3).tolist())
    args.log.record("\n==========================================")
    args.log.record(auc_sub_str)
    args.log.record(auc_mean_str)

    # loss_all = 0.1 * consistency_loss + instance_entropy_loss + batch_entropy_loss + supervised_loss
    # Sub acc:  [84.722 53.472 95.139 76.389 59.028 68.056 72.222 93.75  81.944]
    # Avg acc:  76.08

    # batch size太大了不好，8>4>16
    # 数据对齐操作对于MSDA几乎没有啥效果，可以不用再提了
    # 对于数据combine训练的模型是有效果的

    # SEED，source only 70.80
