# -*- coding: utf-8 -*-
# @Time    : 2021/12/19 20:58
# @Author  : wenzhang
# @File    : target_adapt_multiple.py
import numpy as np
import argparse
import random
import time
import os
import torch as tr
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.onnx
import os.path as osp
from utils import network, loss
from utils.CsvRecord import CsvRecord
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_test, read_seed_test
from utils.utils import fix_random_seed, op_copy, cal_acc_multi
from utils.func_utils import update_decision


def data_load(X, y, args):
    dset_loaders = {}
    train_bs = args.batch_size

    sample_idx = tr.from_numpy(np.arange(len(y))).long()
    data_tar = Data.TensorDataset(X, y, sample_idx)

    dset_loaders["target"] = Data.DataLoader(data_tar, batch_size=train_bs, shuffle=True)
    dset_loaders["Target"] = Data.DataLoader(data_tar, batch_size=train_bs * 3, shuffle=False)
    return dset_loaders


def estimate_source_weight_epoch(batch_tar, netF_list, netC_list, args):
    loss_all = tr.ones(len(args.src), )
    for s in range(len(args.src)):
        features_test = netF_list[s](batch_tar)
        outputs_test = netC_list[s](features_test)

        softmax_out = nn.Softmax(dim=1)(outputs_test)
        im_loss = tr.mean(loss.Entropy(softmax_out))
        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = tr.sum(-msoftmax * tr.log(msoftmax + args.epsilon))
        loss_all[s] = im_loss - gentropy_loss
    weights_domain = loss_all / tr.sum(loss_all)

    return weights_domain


def train_target(args):
    if args.data in ['SEED', 'SEED4']:
        X_tar, y_tar = read_seed_test(args)
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

    netG_list = None
    if args.method == 'decision':
        w = 2 * tr.rand((len(args.src),)) - 1
        netG_list = [network.scalar(w[i]).cuda() for i in range(len(args.src))]

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

        if args.method == 'decision':
            for k, v in netG_list[i].named_parameters():
                param_group += [{'params': v, 'lr': args.lr}]

    ###################################################################################
    imfo_loss = loss.InformationMaximizationLoss().cuda()

    if args.method == 'decision':
        optimizer = optim.SGD(param_group)
        optimizer = op_copy(optimizer)

        for epoch in range(1, args.max_epoch + 1):
            t1 = time.time()
            update_decision(dset_loaders, netF_list, netC_list, netG_list, optimizer, imfo_loss, args)
            test_acc = cal_acc_multi(dset_loaders['Target'], netF_list, netC_list, args, None, netG_list)
            duration = time.time() - t1
            log_str = f'Epoch:{epoch:2d}/{args.max_epoch:2d} [{duration:5.2f}], Acc: {test_acc:.4f}'
            args.log.record(log_str)
            print(log_str)

    return test_acc


if __name__ == '__main__':

    # 这个版本还需要简化，建议后边把MSDF独立出来进行训练和参数策略重制定
    data_name_list = ['001-2014', '001-2014_2', 'SEED', 'SEED4']
    # data_idx = 0

    for dt in range(1, 2):
        data_idx = dt
        data_name = data_name_list[data_idx]
        if data_name == '001-2014': N, chn, class_num, trial_num = 9, 22, 4, 288  # 001-2014
        if data_name == '001-2014_2': N, chn, class_num, trial_num = 9, 22, 2, 144  # 001-2014_2
        if data_name == 'SEED': N, chn, class_num, trial_num = 15, 62, 3, 3394
        if data_name == 'SEED4': N, chn, class_num, trial_num = 15, 62, 4, 851

        args = argparse.Namespace(bottleneck=50, lr=0.01, lr_decay1=0.1, lr_decay2=1.0,
                                  cls_par=0.3, ent_par=1.0, epsilon=1e-05, layer='wn',
                                  N=N, chn=chn, class_num=class_num, cov_type='oas', trial=trial_num,
                                  use_weight=0, tempt=1, eps_th=0.9)

        args.data = data_name
        args.app = 'no'
        args.method = 'decision'
        args.backbone = 'Net_ln2'
        if args.data in ['SEED', 'SEED4']:
            args.batch_size = 32  # 32
            args.max_epoch = 10  # 10
            args.input_dim = 310
            args.norm = 'zscore'
            args.validation = 'random'
        else:
            args.batch_size = 4  # 4
            args.max_epoch = 10  # 10
            args.input_dim = int(args.chn * (args.chn + 1) / 2)
            args.validation = 'last'

        if args.method == 'decision': args.use_weight = 1

        os.environ["CUDA_VISIBLE_DEVICES"] = '3'
        args.data_env = 'gpu' if tr.cuda.device_count() != 0 else 'local'
        args.SEED = 2020
        fix_random_seed(args.SEED)
        tr.backends.cudnn.deterministic = True

        args.data = data_name
        args.output_src = 'ckps/' + args.data + '/source/'
        print(args.data)
        print(args.method)
        print(args)

        args.local_dir = r'/Users/wenz/code/PyPrj/TL/TL_BCI/MSDT/'
        args.result_dir = 'results/target/'
        my_log = LogRecord(args)
        my_log.log_init()
        my_log.record('=' * 50 + '\n' + os.path.basename(__file__) + '\n' + '=' * 50)

        sub_acc_all = np.zeros(N)
        for t in range(N):
            args.idt = t
            target_str = 'S' + str(t + 1)
            info_str = '\n========================== Transfer to ' + target_str + ' =========================='
            print(info_str)
            my_log.record(info_str)
            args.log = my_log

            args.src = ['S' + str(i + 1) for i in range(N)]
            args.src.remove(target_str)

            args.output_dir_src = []
            for i in range(len(args.src)):
                args.output_dir_src.append(osp.join(args.output_src, args.src[i]))
            print(args.output_dir_src)

            sub_acc_all[t] = train_target(args)
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
        # 数据对齐操作几乎没有啥效果，可以不用再提了
        # [68.75, 31.597, 81.597, 38.542, 40.278, 32.986, 68.75, 74.306, 66.319]
        # 55.903

        # MI2-2
        # [84.722, 52.778, 94.444, 70.833, 62.5, 70.139, 72.222, 94.444, 80.556]
        # 75.849

        # MI2-2
        # Sub acc:  [85.417 54.167 96.528 72.917 61.806 70.139 70.833 93.75  81.944]
        # Avg acc:  76.389

