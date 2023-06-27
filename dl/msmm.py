# -*- coding: utf-8 -*-
# @Time    : 2023/01/11
# @Author  : Siyang Li
# @File    : msmm.py
# @Desc    : Multi-Source Multi-Model
import numpy as np
import argparse
import os
import torch as tr
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.nn.functional import softmax
from sklearn.metrics import balanced_accuracy_score, accuracy_score

from utils import network, loss
from utils.CsvRecord import CsvRecord
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_combine_tar, read_seed_combine_tar
from utils.utils import lr_scheduler_full, fix_random_seed, cal_acc_comb, data_loader, cal_bca_comb, cal_metrics_ms, cal_acc_msmm
from utils.loss import MultipleKernelMaximumMeanDiscrepancy, GaussianKernel, mmd, ClassConfusionLoss
from models.FC import FC

import gc
import torch


def train_target(args):
    X_src, y_src, X_tar, y_tar = read_mi_combine_tar(args)
    print('X_src, y_src, X_tar, y_tar:', X_src.shape, y_src.shape, X_tar.shape, y_tar.shape)
    dset_loaders = data_loader(X_src, y_src, X_tar, y_tar, args)

    netF, _ = network.backbone_net(args, return_type='xy')
    netF = netF.cuda()

    netCs = []
    optimizer_cs = []
    for i in range(args.N - 1):
        netC = FC(nn_in=args.feature_deep_dim, nn_out=args.class_num)
        netC = netC.cuda()
        netC.train()
        netCs.append(netC)
        optimizer_c = optim.Adam(netC.parameters(), lr=args.lr)
        optimizer_cs.append(optimizer_c)

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

    optimizer_f = optim.Adam(netF.parameters(), lr=args.lr) #/ (args.N - 1))

    max_iter = args.max_epoch * len(dset_loaders["sources"][0])
    interval_iter = max_iter // args.max_epoch
    args.max_iter = max_iter
    iter_num = 0
    netF.train()
    while iter_num < max_iter:
        iter_target = iter(dset_loaders["target"])

        iter_sources = []
        for i in range(args.N - 1):
            iter_source = iter(dset_loaders["sources"][i])
            iter_sources.append(iter_source)

        for batch_id in range(len(dset_loaders['sources'][0])):
            try:
                inputs_target, _ = iter_target.next()
            except:
                print('ERROR outer')
                sys.exit(0)
            for source_id in range(args.N - 1):
                try:
                    inputs_source, labels_source = iter_sources[source_id].next()
                except:
                    print('ERROR inner')
                    sys.exit(0)

                if inputs_source.size(0) == 1:
                    continue

                features_source = netF(inputs_source)
                features_target = netF(inputs_target)

                outputs_source = netCs[source_id](features_source)
                outputs_target = netCs[source_id](features_target)

                classifier_loss = criterion(outputs_source, labels_source)


                # Deep Alignment Loss
                args.non_linear = False
                mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
                    kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
                    linear=not args.non_linear
                )
                alignment_loss = mkmmd_loss(features_source, features_target)


                '''
                # Class Prediction Loss
                args.prediction_weight = 1.0
                args.t_mcc = 2  # temperature rescaling
                prediction_loss = ClassConfusionLoss(t=args.t_mcc)(outputs_target)
                '''

                '''
                # Multi-Source Discrepancy Loss
                l1_loss = 0
                for i in range(args.N - 1):
                    if i == source_id:
                        continue
                    outputs_other_source = netCs[i](features_source)
                    l1_loss = torch.mean(torch.abs(softmax(outputs_source, dim=1) - softmax(outputs_other_source, dim=1)))
                    l1_loss += l1_loss
                l1_loss /= (args.N - 2)
                discrepancy_loss = l1_loss / (args.N - 2)
                '''

                # Multi-Source Aggregation Loss


                #print(np.round(classifier_loss.item(), 3), np.round(prediction_loss.item(), 3))
                #total_loss = classifier_loss + args.trade_off * mmd_loss + 100 * discrepancy_loss
                #total_loss = classifier_loss + args.prediction_weight * prediction_loss


                total_loss = classifier_loss + alignment_loss
                optimizer_f.zero_grad()
                optimizer_cs[source_id].zero_grad()
                total_loss.backward()
                optimizer_f.step()
                optimizer_cs[source_id].step()

        iter_num += 1

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            for i in range(args.N - 1):
                netCs[i].eval()

            '''
            iter_sources = []
            for i in range(args.N - 1):
                iter_source = iter(dset_loaders["sources"][i])
                iter_sources.append(iter_source)

            source_class_centers = []
            for source_id in range(args.N - 1):
                inputs_source_all, labels_source_all = [], []
                for batch_id in range(len(dset_loaders['sources'][0])):

                    try:
                        inputs_source, labels_source = iter_sources[source_id].next()
                    except:
                        print('ERROR test')
                        sys.exit(0)

                    if inputs_source.size(0) == 1:
                        continue

                    inputs_source_all.append(inputs_source)
                    labels_source_all.append(labels_source)

                inputs_source_all = torch.cat(inputs_source_all)
                labels_source_all = torch.cat(labels_source_all)

                for class_label in range(args.class_num):
                    indices = torch.where(labels_source_all == class_label)
                    inputs_class = inputs_source_all[indices[0]]
                    features_source_class, _ = netF(inputs_class)
                    source_class_center = features_source_class.mean(dim=0)
                    source_class_centers.append(source_class_center)

            source_class_centers = torch.stack(source_class_centers)  # (class_num * source_num, deep_feature_dim)
            
            #acc_t_te = cal_acc_ms_distance(dset_loaders["Target"], base_network, source_class_centers)
            score = cal_acc_msmm(dset_loaders["Target"], netF, netCs, source_class_centers, metrics)

            
            '''
            if args.paradigm == 'MI':
                metrics = accuracy_score
            elif args.paradigm == 'ERP':
                metrics = balanced_accuracy_score

            score = cal_metrics_ms(dset_loaders["Target"], netF, netCs, args.N, args.class_num, metrics)
            log_str = 'Task: {}, Iter:{}/{}; score = {:.2f}%'.format(args.task_str,
                                                                   int(iter_num // len(dset_loaders["sources"][0])),
                                                                   int(max_iter // len(dset_loaders["sources"][0])),
                                                                   score)
            args.log.record(log_str)
            print(log_str)

            netF.train()
            for i in range(args.N - 1):
                netCs[i].train()

    gc.collect()
    torch.cuda.empty_cache()

    return score


if __name__ == '__main__':

    #data_name_list = ['BNCI2014001', 'BNCI2014002', 'BNCI2015001']
    #data_name_list = ['BNCI2014008', 'BNCI2014009', 'BNCI2015003']
    data_name_list = ['BNCI2014001', 'BNCI2014002', 'BNCI2015001', 'BNCI2014008', 'BNCI2014009', 'BNCI2015003']

    dct = pd.DataFrame(
        columns=['dataset', 'avg', 'std', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11',
                 's12', 's13'])

    for data_name in data_name_list:

        if data_name == 'BNCI2014001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 2, 1001, 250, 144, 248
        if data_name == 'BNCI2014002': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 14, 15, 2, 2561, 512, 100, 640
        if data_name == 'MI1': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 7, 59, 2, 300, 200, 100, 72
        if data_name == 'BNCI2015001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 12, 13, 2, 2561, 512, 200, 640
        if data_name == 'BNCI2014008': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'ERP', 8, 8, 2, 206, 256, 4200, 48
        if data_name == 'BNCI2014009': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'ERP', 10, 16, 2, 206, 256, 1728, 48
        if data_name == 'BNCI2015003': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'ERP', 10, 8, 2, 206, 256, 2520, 48

        args = argparse.Namespace(feature_deep_dim=feature_deep_dim, lr=0.001, trial_num=trial_num,
                                  time_sample_num=time_sample_num, sample_rate=sample_rate,
                                  N=N, chn=chn, class_num=class_num, smooth=0, paradigm=paradigm)

        args.data = data_name
        args.method = 'MFSAN'
        args.backbone = 'EEGNet'
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

            # record sub acc to csv
            '''
            args.file_str = os.path.basename(__file__).split('.')[0]
            csv_log = CsvRecord(args)
            csv_log.init()
            csv_log.record(sub_acc_all)
            '''

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