# -*- coding: utf-8 -*-
# @Time    : 2021/12/20 1:06 下午
# @Author  : wenzhang
# @File    : utils.py
import os.path as osp
import os
import numpy as np
import random
import torch as tr
import torch.nn as nn
import torch.utils.data
import torch.utils.data as Data
import moabb
import mne
from torchsampler import ImbalancedDatasetSampler
from sklearn.metrics import balanced_accuracy_score, accuracy_score

from utils.alg_utils import EA

from moabb.datasets import BNCI2014001, BNCI2014002, BNCI2014008, BNCI2014009, BNCI2015003, BNCI2015004, EPFLP300, BNCI2014004, BNCI2015001
from moabb.paradigms import MotorImagery, P300

def dataset_to_file(dataset_name, data_save):
    moabb.set_log_level("ERROR")
    if dataset_name == 'BNCI2014001':
        dataset = BNCI2014001()
        paradigm = MotorImagery(n_classes=4)
        # (5184, 22, 1001) (5184,) 250Hz 9subjects * 4classes * (72+72)trials for 2sessions
    elif dataset_name == 'BNCI2014002':
        dataset = BNCI2014002()
        paradigm = MotorImagery(n_classes=2)
        # (2240, 15, 2561) (2240,) 512Hz 14subjects * 2classes * (50+30)trials * 2sessions(not namely separately)
    elif dataset_name == 'BNCI2014004':
        dataset = BNCI2014004()
        paradigm = MotorImagery(n_classes=2)
        # (6520, 3, 1126) (6520,) 250Hz 9subjects * 2classes * (?)trials * 5sessions
    elif dataset_name == 'BNCI2015001':
        dataset = BNCI2015001()
        paradigm = MotorImagery(n_classes=2)
        # (5600, 13, 2561) (5600,) 512Hz 12subjects * 2 classes * (200 + 200 + (200 for Subj 8/9/10/11)) trials * (2/3)sessions
    elif dataset_name == 'MI1':
        info = None
        return info
        # (1400, 59, 300) (1400,) 100Hz 7subjects * 2classes * 200trials * 1session
    elif dataset_name == 'BNCI2015004':
        dataset = BNCI2015004()
        paradigm = MotorImagery(n_classes=2)
        # [160, 160, 160, 150 (80+70), 160, 160, 150 (80+70), 160, 160]
        # (1420, 30, 1793) (1420,) 256Hz 9subjects * 2classes * (80+80/70)trials * 2sessions
    elif dataset_name == 'BNCI2014008':
        dataset = BNCI2014008()
        paradigm = P300()
        # (33600, 8, 257) (33600,) 256Hz 8subjects 4200 trials * 1session
    elif dataset_name == 'BNCI2014009':
        dataset = BNCI2014009()
        paradigm = P300()
        # (17280, 16, 206) (17280,) 256Hz 10subjects 1728 trials * 3sessions
    elif dataset_name == 'BNCI2015003':
        dataset = BNCI2015003()
        paradigm = P300()
        # (25200, 8, 206) (25200,) 256Hz 10subjects 2520 trials * 1session
    elif dataset_name == 'EPFLP300':
        dataset = EPFLP300()
        paradigm = P300()
        # (25200, 8, 206) (25200,) 256Hz 10subjects 1session
    elif dataset_name == 'ERN':
        ch_names = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
                    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2',
                    'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3',
                    'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'POz', 'P08', 'O1', 'O2']
        info = mne.create_info(ch_names=ch_names, sfreq=200, ch_types=['eeg'] * 56)
        return info
        # (5440, 56, 260) (5440,) 200Hz 16subjects 1session
    # SEED (152730, 62, 5*DE*)  (152730,) 200Hz 15subjects 3sessions

    if data_save:
        print('preparing data...')
        X, labels, meta = paradigm.get_data(dataset=dataset, subjects=dataset.subject_list[:])
        ar_unique, cnts = np.unique(labels, return_counts=True)
        print("labels:", ar_unique)
        print("Counts:", cnts)
        print(X.shape, labels.shape)
        np.save('./data/' + dataset_name + '/X', X)
        np.save('./data/' + dataset_name + '/labels', labels)
        meta.to_csv('./data/' + dataset_name + '/meta.csv')
    else:
        if isinstance(paradigm, MotorImagery):
            X, labels, meta = paradigm.get_data(dataset=dataset, subjects=dataset.subject_list[:], return_epochs=True)
            return X.info
        elif isinstance(paradigm, P300):
            X, labels, meta = paradigm.get_data(dataset=dataset, subjects=dataset.subject_list[:], return_epochs=True)
            return X.info


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def fix_random_seed(SEED):
    tr.manual_seed(SEED)
    tr.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def create_folder(dir_name, data_env, win_root):
    if not osp.exists(dir_name):
        os.system('mkdir -p ' + dir_name)
    if not osp.exists(dir_name):
        if data_env == 'gpu':
            os.mkdir(dir_name)
        elif data_env == 'local':
            os.makedirs(win_root + dir_name)


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def lr_scheduler_full(optimizer, init_lr, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def cal_acc(loader, netF, netC):
    start_test = True
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0].cuda()
            labels = data[1].float()
            outputs = netC(netF(inputs))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = tr.max(all_output, 1)
    accuracy = tr.sum(tr.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    return accuracy * 100


def cal_acc_comb(loader, model, flag=True, fc=None):
    start_test = True
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            if flag:
                _, outputs = model(inputs)
            else:
                if fc is not None:
                    feas, outputs = model(inputs)
                    outputs = fc(feas)
                else:
                    outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = tr.max(all_output, 1)
    pred = tr.squeeze(predict).float()
    true = all_label.cpu()
    acc = accuracy_score(true, pred)

    return acc * 100


def cal_bca_comb(loader, model, flag=True, fc=None):
    start_test = True
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            if flag:
                _, outputs = model(inputs)
            else:
                if fc is not None:
                    feas, outputs = model(inputs)
                    outputs = fc(feas)
                else:
                    outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = tr.max(all_output, 1)
    pred = tr.squeeze(predict).float()
    true = all_label.cpu()
    bca = balanced_accuracy_score(true, pred)

    return bca * 100


def cal_acc_comb_fusion(loader_k, loader_d, model, flag=True, fc=None):
    start_test = True
    with tr.no_grad():
        iter_test_k = iter(loader_k)
        iter_test_d = iter(loader_d)
        for i in range(len(loader_k)):
            k = iter_test_k.next()
            d = iter_test_d.next()
            inputs_k = k[0]
            labels_k = k[1]
            inputs_d = d[0]
            labels_d = d[1]
            inputs_k, inputs_d = inputs_k.cuda(), inputs_d.cuda()
            if flag:
                _, outputs = model((inputs_k, inputs_d))
            else:
                if fc is not None:
                    feas, outputs = model((inputs_k, inputs_d))
                    outputs = fc(feas)
                else:
                    outputs = model((inputs_k, inputs_d))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels_k.float()
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels_k.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = tr.max(all_output, 1)
    pred = tr.squeeze(predict).float()
    true = all_label.cpu()
    acc = accuracy_score(true, pred)

    return acc * 100


def cal_bca_comb_fusion(loader_k, loader_d, model, flag=True, fc=None):
    start_test = True
    with tr.no_grad():
        iter_test_k = iter(loader_k)
        iter_test_d = iter(loader_d)
        for i in range(len(loader_k)):
            k = iter_test_k.next()
            d = iter_test_d.next()
            inputs_k = k[0]
            labels_k = k[1]
            inputs_d = d[0]
            labels_d = d[1]
            inputs_k, inputs_d = inputs_k.cuda(), inputs_d.cuda()
            if flag:
                _, outputs = model((inputs_k, inputs_d))
            else:
                if fc is not None:
                    feas, outputs = model((inputs_k, inputs_d))
                    outputs = fc(feas)
                else:
                    outputs = model((inputs_k, inputs_d))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels_k.float()
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels_k.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = tr.max(all_output, 1)
    pred = tr.squeeze(predict).float()
    true = all_label.cpu()
    acc = balanced_accuracy_score(true, pred)

    return acc * 100


def cal_acc_multi(loader, netF_list, netC_list, args, weight_epoch=None, netG_list=None):
    num_src = len(netF_list)
    for i in range(len(netF_list)): netF_list[i].eval()

    if args.use_weight:
        if args.method == 'msdt':
            domain_weight = weight_epoch.detach()
            # tmp_weight = np.round(tr.squeeze(domain_weight, 0).t().cpu().detach().numpy().flatten(), 3)
            # print('\ntest domain weight: ', tmp_weight)
    else:
        domain_weight = tr.Tensor([1 / num_src] * num_src).reshape([1, num_src, 1]).cuda()

    start_test = True
    with tr.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs, labels = data[0].cuda(), data[1]

            if args.use_weight:
                if args.method == 'decision':
                    weights_all = tr.ones(inputs.shape[0], len(args.src))
                    tmp_output = tr.zeros(len(args.src), inputs.shape[0], args.class_num)
                    for i in range(len(args.src)):
                        tmp_output[i] = netC_list[i](netF_list[i](inputs))
                        weights_all[:, i] = netG_list[i](tmp_output[i]).squeeze()
                    z = tr.sum(weights_all, dim=1) + 1e-16
                    weights_all = tr.transpose(tr.transpose(weights_all, 0, 1) / z, 0, 1)
                    weights_domain = tr.sum(weights_all, dim=0) / tr.sum(weights_all)
                    domain_weight = weights_domain.reshape([1, num_src, 1]).cuda()

            outputs_all = tr.cat([netC_list[i](netF_list[i](inputs)).unsqueeze(1) for i in range(num_src)], 1).cuda()
            preds = tr.softmax(outputs_all, dim=2)
            outputs_all_w = (preds * domain_weight).sum(dim=1).cuda()

            if start_test:
                all_output = outputs_all_w.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs_all_w.float().cpu()), 0)
                all_label = tr.cat((all_label, labels.float()), 0)
    _, predict = tr.max(all_output, 1)
    accuracy = tr.sum(tr.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    for i in range(len(netF_list)): netF_list[i].train()

    return accuracy * 100


def data_alignment(X, num_subjects, args):
    '''
    :param X: np array, EEG data
    :param num_subjects: int, number of total subjects in X
    :return: np array, aligned EEG data
    '''
    # subject-wise EA
    if args.data == 'BNCI2015003' and len(X) < 141:  # check is dataset BNCI2015003 and is downsampled and is not testset
        # upsampling for unequal distributions across subjects, i.e., each subject is upsampled to different num of trials
        print('before EA:', X.shape)
        out = []
        inds = [140, 140, 140, 140, 640, 840, 840, 840, 840, 840]
        inds = np.delete(inds, args.idt)
        for i in range(num_subjects):
            tmp_x = EA(X[np.sum(inds[:i]):np.sum(inds[:i + 1]), :, :])
            out.append(tmp_x)
        X = np.concatenate(out, axis=0)
        print('after EA:', X.shape)
    elif args.data == 'BNCI2015003' and len(X) > 25200:  # check is dataset BNCI2015003 and is upsampled
        # upsampling for unequal distributions across subjects, i.e., each subject is upsampled to different num of trials
        print('before EA:', X.shape)
        out = []
        inds = [4900, 4900, 4900, 4900, 4400, 4200, 4200, 4200, 4200, 4200]
        inds = np.delete(inds, args.idt)
        for i in range(num_subjects):
            tmp_x = EA(X[np.sum(inds[:i]):np.sum(inds[:i + 1]), :, :])
            out.append(tmp_x)
        X = np.concatenate(out, axis=0)
        print('after EA:', X.shape)
    else:
        print('before EA:', X.shape)
        out = []
        for i in range(num_subjects):
            tmp_x = EA(X[X.shape[0] // num_subjects * i:X.shape[0] // num_subjects * (i + 1), :, :])
            out.append(tmp_x)
        X = np.concatenate(out, axis=0)
        print('after EA:', X.shape)
    return X


def data_loader(Xs=None, Ys=None, Xt=None, Yt=None, args=None):
    dset_loaders = {}
    train_bs = args.batch_size


    if args.align and not args.feature:
        Xs = data_alignment(Xs, args.N - 1, args)
        Xt = data_alignment(Xt, 1, args)

    #if Xs != None:
    # 随机打乱会导致训练结果偏高，不影响测试
    src_idx = np.arange(len(Ys))
    if args.validation == 'random':  # for SEED
        num_train = int(0.9 * len(src_idx))
        tr.manual_seed(args.SEED)
        id_train, id_val = tr.utils.data.random_split(src_idx, [num_train, len(src_idx) - num_train])
    if args.validation == 'last':
        if args.paradigm == 'MI':  # for MI
            num_all = args.trial_num
            num_train = int(0.9 * num_all)
            id_train = np.array(src_idx).reshape(-1, num_all)[:, :num_train].reshape(1, -1).flatten()
            id_val = np.array(src_idx).reshape(-1, num_all)[:, num_train:].reshape(1, -1).flatten()

        elif args.paradigm == 'ERP':  # for ERP
            '''
            id_train = []
            id_val = []
            for i in range(args.N - 1):
                subj_Ys = Ys[args.trial_num * i: args.trial_num * (i+1)]
                indices_sorted = np.argsort(subj_Ys)
                ar_unique, cnts_class = np.unique(subj_Ys, return_counts=True)
                inds_train = np.concatenate((indices_sorted[:int(cnts_class[0] * 0.9)], indices_sorted[:int(cnts_class[1] * 0.9)]))
                inds_val = np.concatenate((indices_sorted[int(cnts_class[0] * 0.9):], indices_sorted[int(cnts_class[1] * 0.9):]))
                id_train.append(inds_train + args.trial_num * i)
                id_val.append(inds_val + args.trial_num * i)
            id_train = np.concatenate(id_train).reshape(1, -1).flatten()
            id_val = np.concatenate(id_val).reshape(1, -1).flatten()
            '''
            if args.data == 'BNCI2015003' and len(Xs) < 25200:
                inds = [140, 140, 140, 140, 640, 840, 840, 840, 840, 840]
                inds = np.delete(inds, args.idt)
                id_train, id_val = [], []
                for i in range(args.N - 1):
                    num_all = inds[i]
                    num_train = int(0.9 * num_all)
                    before_inds = int(np.sum(inds[:i]))
                    id_t = (np.arange(num_train, dtype=int) + before_inds)
                    id_v = (np.arange(num_all - num_train, dtype=int) + before_inds + num_train)
                    id_train.append(id_t)
                    id_val.append(id_v)
                id_train = np.concatenate(id_train).reshape(1, -1).flatten()
                id_val = np.concatenate(id_val).reshape(1, -1).flatten()
            elif args.data == 'BNCI2015003' and len(Xs) > 25200:
                inds = [4900, 4900, 4900, 4900, 4400, 4200, 4200, 4200, 4200, 4200]
                inds = np.delete(inds, args.idt)
                id_train, id_val = [], []
                for i in range(args.N - 1):
                    num_all = inds[i]
                    num_train = int(0.9 * num_all)
                    before_inds = int(np.sum(inds[:i]))
                    id_t = (np.arange(num_train, dtype=int) + before_inds)
                    id_v = (np.arange(num_all - num_train, dtype=int) + before_inds + num_train)
                    id_train.append(id_t)
                    id_val.append(id_v)
                id_train = np.concatenate(id_train).reshape(1, -1).flatten()
                id_val = np.concatenate(id_val).reshape(1, -1).flatten()
            else:
                num_all = args.trial_num
                num_train = int(0.9 * num_all)
                id_train = np.array(src_idx).reshape(-1, num_all)[:, :num_train].reshape(1, -1).flatten()
                id_val = np.array(src_idx).reshape(-1, num_all)[:, num_train:].reshape(1, -1).flatten()

            #ar_unique, class_counts = np.unique(Ys, return_counts=True)
            #num_samples = sum(class_counts)
            #labels = Ys  # corresponding labels of samples

            # assuming all subjects have equal number of trials and equal ratio of samples by classes
            #class_weights = [num_samples / class_counts[i] for i in range(len(class_counts))]
            #weights = [class_weights[labels[i]] for i in range(int(num_samples))]
            #sampler_train = Data.WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples * (args.N - 1)), replacement=True)


    #if valid_percentage != None and valid_percentage > 0:
    valid_Xs, valid_Ys = tr.from_numpy(Xs[id_val, :]).to(
        tr.float32), tr.from_numpy(Ys[id_val].reshape(-1,)).to(tr.long)
    #if 'EEGNet' in args.backbone and not args.feature:
    if 'EEGNet' in args.backbone:
        valid_Xs = valid_Xs.unsqueeze_(3).permute(0, 3, 1, 2)
    train_Xs, train_Ys = tr.from_numpy(Xs[id_train, :]).to(
        tr.float32), tr.from_numpy(Ys[id_train].reshape(-1,)).to(tr.long)
    #if 'EEGNet' in args.backbone and not args.feature:
    if 'EEGNet' in args.backbone:
        train_Xs = train_Xs.unsqueeze_(3).permute(0, 3, 1, 2)

    Xs, Ys = tr.from_numpy(Xs).to(
        tr.float32), tr.from_numpy(Ys.reshape(-1, )).to(tr.long)
    if 'EEGNet' in args.backbone:
    #if 'EEGNet' in args.backbone and not args.feature:
        Xs = Xs.unsqueeze_(3).permute(0, 3, 1, 2)

    Xt, Yt = tr.from_numpy(Xt).to(
        tr.float32), tr.from_numpy(Yt.reshape(-1,)).to(tr.long)
    if 'EEGNet' in args.backbone:
    #if 'EEGNet' in args.backbone and not args.feature:
        Xt = Xt.unsqueeze_(3).permute(0, 3, 1, 2)

    try:
        data_src = Data.TensorDataset(Xs.cuda(), Ys.cuda())
        source_tr = Data.TensorDataset(train_Xs.cuda(), train_Ys.cuda())
        source_te = Data.TensorDataset(valid_Xs.cuda(), valid_Ys.cuda())
        #if Xt != None:
        data_tar = Data.TensorDataset(Xt.cuda(), Yt.cuda())
    except Exception:
        data_src = Data.TensorDataset(Xs, Ys)
        source_tr = Data.TensorDataset(train_Xs, train_Ys)
        source_te = Data.TensorDataset(valid_Xs, valid_Ys)
        #if Xt != None:
        data_tar = Data.TensorDataset(Xt, Yt)

    # for DNN
    #if Xs != None:
    dset_loaders["source_tr"] = Data.DataLoader(source_tr, batch_size=train_bs, shuffle=True, drop_last=True)
    dset_loaders["source_te"] = Data.DataLoader(source_te, batch_size=train_bs, shuffle=False, drop_last=False)

    # for DAN/DANN/CDAN/MCC
    #if Xs != None:
    dset_loaders["source"] = Data.DataLoader(data_src, batch_size=train_bs, shuffle=True, drop_last=True)
    #if Xt != None:
    dset_loaders["target"] = Data.DataLoader(data_tar, batch_size=train_bs, shuffle=True, drop_last=True)

    # for generating feature
    #if Xs != None:
    dset_loaders["Source"] = Data.DataLoader(data_src, batch_size=train_bs * 3, shuffle=False, drop_last=False)
    #if Xt != None:
    dset_loaders["Target"] = Data.DataLoader(data_tar, batch_size=train_bs * 3, shuffle=False, drop_last=False)

    return dset_loaders
