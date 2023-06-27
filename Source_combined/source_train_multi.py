# @Time    : 2020/7/21 20:58
# @Author  : wenzhang
# @File    : source_training_mi.py
import numpy as np
import argparse
import random
import torch as tr
import torch.optim as optim
import torch.onnx
import os.path as osp
import os
from os import walk
from scipy.io import loadmat
import torch.utils.data as Data
from utils import network
from utils.loss import CELabelSmooth
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_train, read_seizure_train
from utils.utils import lr_scheduler, op_copy, cal_acc


def data_load(X, y, args):
    dset_loaders = {}
    train_bs = args.batch_size
    tr.manual_seed(args.SEED)
    src_idx = range(y.shape[0])
    num_all = args.trial

    num_train = int(0.9 * num_all)
    id_train = np.array(src_idx).reshape(-1, num_all)[:, :num_train].reshape(1, -1).flatten()
    id_val = np.array(src_idx).reshape(-1, num_all)[:, num_train:].reshape(1, -1).flatten()

    source_tr = Data.TensorDataset(X[id_train, :], y[id_train])
    dset_loaders['source_tr'] = Data.DataLoader(source_tr, batch_size=train_bs, shuffle=True, drop_last=True)

    source_te = Data.TensorDataset(X[id_val, :], y[id_val])
    dset_loaders['source_te'] = Data.DataLoader(source_te, batch_size=train_bs * 2, shuffle=False, drop_last=True)

    return dset_loaders


def train_source(args):
    X_src, y_src = read_seizure_train(args)
    dset_loaders = data_load(X_src, y_src, args)

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
        classifier_loss = CELabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source)

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netC.eval()

            acc_s_te, sen_s_te, spec_s_te, auc_s_te = cal_acc(dset_loaders['source_te'], netF, netC)
            log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%'.format(args.name_src, iter_num, max_iter, auc_s_te)
            args.log.record(log_str)
            print(log_str)

            if auc_s_te >= auc_init:
                auc_init = auc_s_te
                best_netF = netF.state_dict()
                best_netC = netC.state_dict()

            netF.train()
            netC.train()

    torch.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
    torch.save(best_netC, osp.join(args.output_dir_src, "source_C.pt"))

    return auc_s_te


def get_n_src(target_id):
    domains = next(walk('/home/zwwang/code/Source_combined/data/fts_labels/'), (None, None, []))[2]
    i = target_id
    src = loadmat('/home/zwwang/code/Source_combined/data/fts_labels/' + domains[i])
    src_label = src['label']
    src_num = src_label.shape[0]
    return src_num


if __name__ == '__main__':
    # used for ensemble, decision, shot_ens
    data_name_list = ['seizure']
    data_idx = 0
    data_name = data_name_list[data_idx]
    domain = next(walk('/home/zwwang/code/Source_combined/data/fts_labels/'), (None, None, []))[2]
    n_subject = len(domain)
    sub_auc_all = np.zeros(n_subject)
    for ids in range(n_subject):
        n_src = get_n_src(ids)
        if data_name == 'seizure': N, chn, class_num, trial_num = n_subject, 18, 2, n_src
        if data_name == '001-2014': N, chn, class_num, trial_num = 9, 22, 4, 288  # 001-2014
        if data_name == '001-2014_2': N, chn, class_num, trial_num = 9, 22, 2, 144  # 001-2014_2

        args = argparse.Namespace(bottleneck=50, batch_size=32, lr=0.01, epsilon=1e-05, layer='wn',
                                  max_epoch=100, smooth=0.1, chn=chn, trial=trial_num,
                                  N=N, class_num=class_num, cov_type='oas')

        args.aug = 0  # data aug only for MI
        args.data = data_name
        args.method = 'multiple'
        args.backbone = 'Net_ln2'
        args.validation = 'last'

        os.environ["CUDA_VISIBLE_DEVICES"] = '3'
        args.data_env = 'gpu' if tr.cuda.device_count() != 0 else 'local'
        args.SEED = 2020
        torch.manual_seed(args.SEED)
        torch.cuda.manual_seed(args.SEED)
        np.random.seed(args.SEED)
        random.seed(args.SEED)
        torch.backends.cudnn.deterministic = True

        args.input_dim = int(args.chn * (args.chn + 1) / 2)
        mdl_path = 'ckps/'
        args.output = mdl_path + args.data + '/source/'
        print(args.data)

        args.local_dir = r'/Users/wenz/code/PyPrj/TL/TL_BCI/MSDT/'
        args.result_dir = 'results/target/'
        my_log = LogRecord(args)
        my_log.log_init()

        sub_acc_all = []
        for s in range(N):
            args.ids = s
            source_str = 'S' + str(s + 1)
            info_str = '\n========================== Within subject ' + source_str + ' =========================='
            print(info_str)
            my_log.record(info_str)
            args.log = my_log

            args.name_src = source_str
            args.output_dir_src = osp.join(args.output, args.name_src)

            if not osp.exists(args.output_dir_src):
                os.system('mkdir -p ' + args.output_dir_src)
            if not osp.exists(args.output_dir_src):
                if args.data_env == 'gpu':
                    os.mkdir(args.output_dir_src)
                elif args.data_env == 'local':
                    os.makedirs(args.local_dir + args.output_dir_src)

            acc_sub = train_source(args)
            sub_acc_all.append(acc_sub)
        print(np.round(sub_acc_all, 3))
        print(np.round(np.mean(sub_acc_all), 3))

        acc_sub_str = str(np.round(sub_acc_all, 3).tolist())
        acc_mean_str = str(np.round(np.mean(sub_acc_all), 3).tolist())
        args.log.record("\n==========================================")
        args.log.record(acc_sub_str)
        args.log.record(acc_mean_str)



