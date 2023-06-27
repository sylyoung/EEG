# -*- coding: utf-8 -*-
# @Time    : 2021/12/20 13:58
# @Author  : wenzhang
# @File    : target_adapt_combine.py
import numpy as np
import argparse
import time
import os
import torch as tr
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import os.path as osp
from scipy.spatial.distance import cdist
from os import walk
from scipy.io import loadmat

from utils import network, loss
from utils.CsvRecord import CsvRecord
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_test, read_seed_test, read_seizure_test
from utils.utils import lr_scheduler, fix_random_seed, op_copy, cal_acc


def data_load(X, y, args):
    dset_loaders = {}
    train_bs = args.batch_size

    sample_idx = tr.from_numpy(np.arange(len(y))).long()
    data_tar = Data.TensorDataset(X, y, sample_idx)

    dset_loaders["target"] = Data.DataLoader(data_tar, batch_size=train_bs, shuffle=True)
    dset_loaders["Target"] = Data.DataLoader(data_tar, batch_size=train_bs * 3, shuffle=False)
    return dset_loaders


def train_target(args):
    if args.data in ['SEED', 'SEED4']:
        X_tar, y_tar = read_seed_test(args)
    elif args.data in ['seizure']:
        X_tar, y_tar = read_seizure_test(args)
    else:
        X_tar, y_tar = read_mi_test(args)
    dset_loaders = data_load(X_tar, y_tar, args)

    if args.bottleneck == 50:
        netF, netC = network.backbone_net(args, 100, return_type='y')
    if args.bottleneck == 64:
        netF, netC = network.backbone_net(args, 128, return_type='y')

    modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(tr.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(tr.load(modelpath))
    netC.eval()

    for k, v in netC.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.cuda()
        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netF.eval()
            mem_label = obtain_label(dset_loaders["Target"], netF, netC, args)
            mem_label = tr.from_numpy(mem_label).cuda()
            netF.train()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        features_test = netF(inputs_test)
        outputs_test = netC(features_test)

        # # loss definition
        if args.cls_par > 0:
            pred = mem_label[tar_idx].long()
            classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
            classifier_loss *= args.cls_par
        else:
            classifier_loss = tr.tensor(0.0).cuda()

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = tr.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = tr.sum(msoftmax * tr.log(msoftmax + args.epsilon))
                entropy_loss += gentropy_loss
            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            acc_t_te, sen_t_te, spec_t_te, auc_t_te = cal_acc(dset_loaders["Target"], netF, netC)
            log_str = 'Task: {}, Iter:{}/{}; acc = {:.2f}%, sen = {:.2f}%, spec = {:.2f}%, auc = {:.2f}%'.format(
                args.task_str, iter_num, max_iter, acc_t_te, sen_t_te, spec_t_te, auc_t_te)
            print(log_str)
            netF.train()

    if iter_num == max_iter:
        print('{}, TL Acc = {:.2f}%'.format(args.task_str, auc_t_te))
        return auc_t_te, sen_t_te, spec_t_te, acc_t_te


def obtain_label(loader, netF, netC, args):
    start_test = True
    with tr.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netF(inputs)
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = tr.cat((all_fea, feas.float().cpu()), 0)
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = tr.sum(-all_output * tr.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = tr.max(all_output, 1)

    accuracy = tr.sum(tr.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = tr.cat((all_fea, tr.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / tr.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count > args.threshold)
    labelset = labelset[0]
    # print(labelset)

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):  # SSL
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    # log_str = 'SSL_Acc = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    # print(log_str)

    return pred_label.astype('int')


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

        args = argparse.Namespace(bottleneck=50, lr=0.01, lr_decay1=0.1, lr_decay2=1.0, ent=True,
                                  gent=True, cls_par=0.3, ent_par=1.0, epsilon=1e-05, layer='wn', interval=5,
                                  N=N, chn=chn, class_num=class_num, cov_type='oas', trial=trial_num,
                                  threshold=0, distance='cosine')

        args.data = data_name
        args.app = 'no'
        args.method = 'shot'
        args.backbone = 'Net_ln2'
        if args.data in ['SEED', 'SEED4', 'seizure']:
            args.batch_size = 32
            args.max_epoch = 10
            args.input_dim = 1332
            args.norm = 'zscore'
            args.validation = 'random'
        else:
            args.batch_size = 4
            args.max_epoch = 10
            args.input_dim = int(args.chn * (args.chn + 1) / 2)
            args.validation = 'last'

        os.environ["CUDA_VISIBLE_DEVICES"] = '2'
        args.data_env = 'gpu' if tr.cuda.device_count() != 0 else 'local'
        args.SEED = 2020
        fix_random_seed(args.SEED)
        tr.backends.cudnn.deterministic = True

        args.data = data_name
        args.output_src = 'ckps/' + args.data + '/source/'
        print(args.data)
        print(args)

        args.local_dir = r'C:/wzw/研一下/0_MSDT/Source_combined/'
        args.result_dir = 'results/target/'
        my_log = LogRecord(args)
        my_log.log_init()
        my_log.record('=' * 50 + '\n' + os.path.basename(__file__) + '\n' + '=' * 50)

        args.idt = idt
        source_str = 'Except_S' + domain[idt][1:-4]
        target_str = 'S' + domain[idt][1:-4]
        info_str = '\n========================== Transfer to ' + target_str + ' =========================='
        print(info_str)
        my_log.record(info_str)
        args.log = my_log
        args.task_str = source_str + '_2_' + target_str

        args.src = ['S' + domain[i][1:-4] for i in range(N)]
        args.src.remove(target_str)
        args.output_dir_src = osp.join(args.output_src, source_str)

        sub_auc_all[idt], sub_sen_all[idt], sub_spec_all[idt], sub_acc_all[idt] = train_target(args)
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

    # loss_all = 0.1 * consistency_loss + instance_entropy_loss + batch_entropy_loss + supervised_loss
    # Sub acc:  [84.722 53.472 95.139 76.389 59.028 68.056 72.222 93.75  81.944]
    # Avg acc:  76.08

    # batch size太大了不好，8>4>16
    # 数据对齐操作几乎没有啥效果，可以不用再提了
