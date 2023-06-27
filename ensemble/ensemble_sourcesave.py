import numpy as np
import pandas as pd
import torch as tr
import torch.nn as nn
import torch
from sklearn import preprocessing

from dl.utils import network, loss
from dl.utils.dataloader import read_mi_combine_tar
from dl.utils.utils import data_loader as d_l

import sys
import os
import random
import argparse
import csv

def time_cut(data, cut_percentage):
    # Time Cutting: cut at a certain percentage of the time. data: (..., ..., time_samples)
    data = data[:, :, :int(data.shape[2] * cut_percentage)]
    return data


def data_loader(dataset):
    '''

    :param dataset: str, dataset name
    :return: X, y, num_subjects, paradigm, sample_rate
    '''

    if dataset == 'MI1':
        data = np.load('./data/' + dataset + '/MI1.npz')
        X = data['data']
        X = X.reshape(-1, X.shape[2], X.shape[3])
        y = data['label']
        y = y.reshape(-1, )
    elif dataset == 'BNCI2014001-4':
        X = np.load('./data/' + 'BNCI2014001' + '/X.npy')
        y = np.load('./data/' + 'BNCI2014001' + '/labels.npy')
    else:
        X = np.load('./data/' + dataset + '/X.npy')
        y = np.load('./data/' + dataset + '/labels.npy')
    print(X.shape, y.shape)

    num_subjects, paradigm, sample_rate = None, None, None

    if dataset == 'BNCI2014001':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

        # only use two classes [left_hand, right_hand]
        indices = []
        for i in range(len(y)):
            if y[i] in ['left_hand', 'right_hand']:
                indices.append(i)
        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014002':
        paradigm = 'MI'
        num_subjects = 14
        sample_rate = 512
        ch_num = 15

        # only use session train, remove session test
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(100) + (160 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

    elif dataset == 'BNCI2015001':
        paradigm = 'MI'
        num_subjects = 12
        sample_rate = 512
        ch_num = 13

        # only use session 1, remove session 2/3
        indices = []
        for i in range(num_subjects):
            if i in [7, 8, 9, 10]:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            elif i == 11:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            else:
                indices.append(np.arange(200) + (400 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014001-4':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

    elif dataset == 'MI1':
        paradigm = 'MI'
        num_subjects = 7
        sample_rate = 100
        ch_num = 59
    elif dataset == 'BNCI2014008':
        paradigm = 'ERP'
        num_subjects = 8
        sample_rate = 256
        ch_num = 8

        # time cut
        X = time_cut(X, cut_percentage=0.8)
    elif dataset == 'BNCI2014009':
        paradigm = 'ERP'
        num_subjects = 10
        sample_rate = 256
        ch_num = 16
    elif dataset == 'BNCI2015003':
        paradigm = 'ERP'
        num_subjects = 10
        sample_rate = 256
        ch_num = 8
    elif dataset == 'PhysionetMI':
        paradigm = 'MI'
        num_subjects = 105
        sample_rate = 160
        ch_num = 64

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    print('data shape:', X.shape, ' labels shape:', y.shape)
    return X, y, num_subjects, paradigm, sample_rate, ch_num


def fix_random_seed(SEED):
    tr.manual_seed(SEED)
    tr.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


if __name__ == '__main__':
    fix_random_seed(42)

    data_name_list = ['BNCI2014001', 'BNCI2014002', 'BNCI2015001']
    #backbone = 'EEGNet'
    backbone = 'ShallowFBCSPNet_feature'
    method = 'SML-train-ShallowCNN'

    align = sys.argv[1]
    if align == 'True':
        align = True
    elif align == 'False':
        align = False

    batch_size = 32

    device_id = str(sys.argv[2])
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id

    for data_name in data_name_list:

        X, y, num_subjects, paradigm, sample_rate, ch_num = data_loader(data_name)

        '''
        if data_name == 'BNCI2014001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 2, 1001, 250, 144, 248
        if data_name == 'BNCI2014002': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 14, 15, 2, 2561, 512, 100, 640
        if data_name == 'MI1': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 7, 59, 2, 300, 200, 100, 72
        if data_name == 'BNCI2015001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 12, 13, 2, 2561, 512, 200, 640
        if data_name == 'BNCI2014008': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'ERP', 8, 8, 2, 206, 256, 4200, 48
        if data_name == 'BNCI2014009': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'ERP', 10, 16, 2, 206, 256, 1728, 48
        if data_name == 'BNCI2015003': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'ERP', 10, 8, 2, 206, 256, 2520, 48
        if data_name == 'BNCI2014001-4': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 4, 1001, 250, 288, 248
        '''
        import braindecode.models.shallow_fbcsp

        if data_name == 'BNCI2014001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 2, 1001, 250, 144, 2440  # 248
        if data_name == 'BNCI2014002': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 14, 15, 2, 2561, 512, 100, 6600  # 640
        if data_name == 'MI1': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 7, 59, 2, 300, 200, 100, 72
        if data_name == 'BNCI2015001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 12, 13, 2, 2561, 512, 200, 6600  # 640
        if data_name == 'BNCI2014008': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'ERP', 8, 8, 2, 206, 256, 4200, 48
        if data_name == 'BNCI2014009': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'ERP', 10, 16, 2, 206, 256, 1728, 48
        if data_name == 'BNCI2015003': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'ERP', 10, 8, 2, 206, 256, 2520, 48
        if data_name == 'BNCI2014001-4': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 4, 1001, 250, 288, 248

        seed_arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        for SEED in seed_arr:
            for target_subject_id in range(N):
                #y_true = []
                y_pred = []

                args = argparse.Namespace(feature_deep_dim=feature_deep_dim, lr=0.001, trial_num=trial_num,
                                          time_sample_num=time_sample_num, sample_rate=sample_rate,
                                          N=N, chn=chn, class_num=class_num, smooth=0, paradigm=paradigm,
                                          data_name=data_name, backbone=backbone, idt=target_subject_id, SEED=SEED,
                                          method=method, data=data_name, batch_size=batch_size, align=align,
                                          feature=False)
                args.validation = 'None'
                args.max_epoch = 0
                args.eval_epoch = args.max_epoch / 10

                args.data_env = 'gpu' if tr.cuda.device_count() != 0 else 'local'

                netF, netC = network.backbone_net(args, return_type='xy')
                if args.data_env == 'gpu':
                    netF, netC = netF.cuda(), netC.cuda()
                base_network = nn.Sequential(netF, netC)

                if args.data_env != 'local':
                    base_network.load_state_dict(
                        torch.load('./runs/' + str(args.data_name) + '/' + str(args.backbone) +
                                   '_S' + str(args.idt) + '_seed' + str(args.SEED) + '.ckpt'))
                    # base_network.load_state_dict(torch.load('./runs/' + str(args.data_name) + '/' + str(args.backbone) +
                    #                                        '_S' + str(args.idt) + '_seed' + str(args.SEED)+ '_Imbalanced' + '.ckpt'))
                else:
                    base_network.load_state_dict(
                        torch.load('./runs/' + str(args.data_name) + '/' + str(args.backbone) +
                                   '_S' + str(args.idt) + '_seed' + str(args.SEED) + '.ckpt',
                                   map_location=torch.device('cpu')))
                    # base_network.load_state_dict(torch.load('./runs/' + str(args.data_name) + '/' + str(args.backbone) +
                    #                                        '_S' + str(args.idt) + '_seed' + str(args.SEED)+ '_Imbalanced' + '.ckpt', map_location=torch.device('cpu')))

                X_src, y_src, X_tar, y_tar = read_mi_combine_tar(args)
                print('X_src, y_src, X_tar, y_tar:', X_src.shape, y_src.shape, X_tar.shape, y_tar.shape)
                dset_loaders = d_l(X_src, y_src, X_tar, y_tar, args)

                base_network.eval()

                iter_source = iter(dset_loaders["source"])
                for i in range(len(iter_source)):
                    inputs_source, labels_source = next(iter_source)

                    _, outputs_source = base_network(inputs_source)

                    softmax_out = nn.Softmax(dim=1)(outputs_source)

                    labels = labels_source.float().cpu()

                    y_pred.append(softmax_out.detach().numpy())
                    #y_true.append(labels.item())

                y_pred = np.array(y_pred).reshape(-1, args.class_num)[:, 1]
                #y_true = np.array(y_true).reshape(-1, )

                path = './logs/' + 'SML-train/' + str(data_name) + str(method) + '_seed_' + str(SEED) + '_testsubj_' + str(target_subject_id) + "_pred.csv"

                with open(path, 'w') as f:
                    print(path)
                    writer = csv.writer(f)
                    writer.writerow(y_pred)