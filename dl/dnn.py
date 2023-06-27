# -*- coding: utf-8 -*-
# @Time    : 2023/01/11
# @Author  : Siyang Li
# @File    : dnn.py
import numpy as np
import argparse
import os
import torch as tr
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from utils import network, loss
from utils.CsvRecord import CsvRecord
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_combine_tar
from utils.utils import fix_random_seed, cal_acc_comb, data_loader, cal_bca_comb

import gc
import torch


def train_target(args):
    X_src, y_src, X_tar, y_tar = read_mi_combine_tar(args)
    print('X_src, y_src, X_tar, y_tar:', X_src.shape, y_src.shape, X_tar.shape, y_tar.shape)
    dset_loaders = data_loader(X_src, y_src, X_tar, y_tar, args)

    netF, netC = network.backbone_net(args, return_type='xy')
    netF, netC = netF.cuda(), netC.cuda()
    base_network = nn.Sequential(netF, netC)

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
    optimizer_c = optim.Adam(netC.parameters(), lr=args.lr)

    max_iter = args.max_epoch * len(dset_loaders["source"])
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

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1

        features_source, outputs_source = base_network(inputs_source)

        classifier_loss = criterion(outputs_source, labels_source)

        optimizer_f.zero_grad()
        optimizer_c.zero_grad()
        classifier_loss.backward()
        optimizer_f.step()
        optimizer_c.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            base_network.eval()

            if args.paradigm == 'MI':
                acc_t_te, _ = cal_acc_comb(dset_loaders["Target"], base_network, args)
                log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%'.format(args.task_str, int(iter_num // len(dset_loaders["source"])), int(max_iter // len(dset_loaders["source"])), acc_t_te)
                args.log.record(log_str)
                print(log_str)
            elif args.paradigm == 'ERP':
                acc_t_te, _ = cal_bca_comb(dset_loaders["Target"], base_network, args)
                log_str = 'Task: {}, Iter:{}/{}; BCA = {:.2f}%'.format(args.task_str, int(iter_num // len(dset_loaders["source"])), int(max_iter // len(dset_loaders["source"])), acc_t_te)
                args.log.record(log_str)
                print(log_str)

            base_network.train()

    print('saving model...')

    torch.save(base_network.state_dict(),
               './runs/' + str(args.data_name) + '/' + str(args.backbone) + '_S' + str(args.idt) + 'withoutEA_seed' + str(args.SEED) + '.ckpt')

    '''
    # plot
    base_network.eval()
    all_features = []
    for i in range(args.N - 1):
        subj_features = []
        iter_source = iter(dset_loaders["Sources"][i])
        for inputs_source, labels_source in iter_source:
            features_source, _ = base_network(inputs_source)
            subj_features.append(features_source)
        subj_features = torch.cat(subj_features)
        all_features.append(subj_features)

    subj_features = []
    iter_target = iter(dset_loaders["Target"])
    for inputs_target, labels_target in iter_target:
        features_target, _ = base_network(inputs_target)
        subj_features.append(features_target)
    subj_features = torch.cat(subj_features)
    all_features.append(subj_features)

    all_features = torch.cat(all_features)
    all_features = all_features.reshape(all_features.shape[0], -1).detach().cpu().numpy()

    X_2d = TSNE(n_components=2, learning_rate=10,
                init='random', random_state=42).fit_transform(all_features)  # 降维到2维

    labels_color = []

    color_arr = ['green', 'dark green', 'yellow', 'dark yellow', 'orange', 'dark orange', 'violet', 'dark violet',
                 'blue', 'dark blue', 'gray', 'black', 'magenta', 'dark magenta', 'pink', 'light pink', 'red', 'dark red']  # https://matplotlib.org/stable/gallery/color/named_colors.html

    for i in range(args.N):
        for j in range(args.trial_num // args.class_num):
            for k in range(args.class_num):
                labels_color.append(color_arr[i])

    palette = []

    for i in range(args.N):
        palette.append(sns.xkcd_rgb[color_arr[i]])

    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], s=3, hue=labels_color, legend=True,
                    linewidth=0, palette=palette)
    # X_2d[:, 0], X_2d[:, 1] 分别表示2维特征的(x1,x2)
    # s 是点的大小

    plt.show()
    plt.savefig(str(args.data_name) + '_targetID' + str(args.idt) + '.png')
    '''
    gc.collect()
    torch.cuda.empty_cache()

    return acc_t_te


if __name__ == '__main__':

    #data_name_list = ['BNCI2014001', 'BNCI2014002', 'BNCI2015001']
    #data_name_list = ['BNCI2014001-4']
    data_name_list = ['BNCI2014008', 'BNCI2014009', 'BNCI2015003']
    #data_name_list = ['BNCI2014001', 'BNCI2014002', 'BNCI2015001', 'BNCI2014008', 'BNCI2014009', 'BNCI2015003']

    dct = pd.DataFrame(columns=['dataset', 'avg', 'std', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13'])

    for data_name in data_name_list:
        if data_name == 'BNCI2014001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 2, 1001, 250, 144, 248
        if data_name == 'BNCI2014002': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 14, 15, 2, 2561, 512, 100, 640
        if data_name == 'MI1': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 7, 59, 2, 300, 200, 100, 72
        if data_name == 'BNCI2015001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 12, 13, 2, 2561, 512, 200, 640
        #if data_name == 'BNCI2014004': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 3, 2, 1126, 250, 200, 640
        if data_name == 'BNCI2014008': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'ERP', 8, 8, 2, 206, 256, 4200, 48
        if data_name == 'BNCI2014009': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'ERP', 10, 16, 2, 206, 256, 1728, 48
        if data_name == 'BNCI2015003': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'ERP', 10, 8, 2, 206, 256, 2520, 48
        if data_name == 'BNCI2014001-4': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 4, 1001, 250, 288, 248
        #if data_name == 'NICU': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'Seizure', 39, 18, 2, 2000, 500, -1, -1

        '''
        import braindecode.models.shallow_fbcsp
        if data_name == 'BNCI2014001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 2, 1001, 250, 144, 2440 #248
        if data_name == 'BNCI2014002': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 14, 15, 2, 2561, 512, 100, 6600 #640
        if data_name == 'MI1': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 7, 59, 2, 300, 200, 100, 72
        if data_name == 'BNCI2015001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 12, 13, 2, 2561, 512, 200, 6600 #640
        if data_name == 'BNCI2014008': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'ERP', 8, 8, 2, 206, 256, 4200, 48
        if data_name == 'BNCI2014009': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'ERP', 10, 16, 2, 206, 256, 1728, 48
        if data_name == 'BNCI2015003': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'ERP', 10, 8, 2, 206, 256, 2520, 48
        if data_name == 'BNCI2014001-4': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 4, 1001, 250, 288, 248
        '''

        args = argparse.Namespace(feature_deep_dim=feature_deep_dim, lr=0.001, trial_num=trial_num,
                                  time_sample_num=time_sample_num, sample_rate=sample_rate,
                                  N=N, chn=chn, class_num=class_num, smooth=0, paradigm=paradigm, data_name=data_name)

        args.data = data_name
        args.method = 'EEGNet-ERP'
        #args.method = 'DNN-ShallowCNN-2015001'
        args.backbone = 'EEGNet'
        #args.backbone = 'ShallowFBCSPNet_feature'
        args.feature = False

        import sys
        align = sys.argv[1]
        if align == 'True':
            args.align = True
        elif align == 'False':
            args.align = False

        args.batch_size = 32
        if paradigm == 'ERP':
            args.batch_size = 256
        args.max_epoch = 100
        args.validation = 'None'
        args.eval_epoch = args.max_epoch / 10

        device_id = str(sys.argv[2])
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        args.data_env = 'gpu' if tr.cuda.device_count() != 0 else 'local'

        total_acc = []

        for s in [1, 2, 3, 4, 5]:
        #for s in [6, 7, 8, 9, 10, 11]:
        #for s in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
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

                sub_acc_all[idt] = train_target(args)
            print('Sub acc: ', np.round(sub_acc_all, 3))
            print('Avg acc: ', np.round(np.mean(sub_acc_all), 3))
            total_acc.append(sub_acc_all)

            acc_sub_str = str(np.round(sub_acc_all, 3).tolist())
            acc_mean_str = str(np.round(np.mean(sub_acc_all), 3).tolist())
            args.log.record("\n==========================================")
            args.log.record(acc_sub_str)
            args.log.record(acc_mean_str)

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