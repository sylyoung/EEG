# -*- coding: utf-8 -*-
# @Time    : 2022/3/25 13:30
# @Author  : wenzhang
# @File    : mmd.py
import torch
import numpy as np


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])

    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))

    L2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    return sum(kernel_val)


def convert_to_onehot(sca_label, class_num=31):
    return np.eye(class_num)[sca_label]


def cal_basic_weight(lbl_onehot, class_num):
    s_sum = np.sum(lbl_onehot, axis=0).reshape(1, class_num)
    s_sum[s_sum == 0] = 100
    tmp_weight_matrix = lbl_onehot / s_sum

    return tmp_weight_matrix


def cal_weight(s_label, t_label, class_num=31):
    batch_size = s_label.size()[0]
    s_sca_label = s_label.cpu().data.numpy()
    t_sca_label = t_label.cpu().data.max(1)[1].numpy()
    com_class = list(set(s_sca_label) & set(t_sca_label))

    if len(com_class) == 0:
        weight_ss = np.array([0])
        weight_tt = np.array([0])
        weight_st = np.array([0])

        src_class_own = list(np.unique(s_sca_label))
        tar_class_own = list(np.unique(t_sca_label))
        s_vec_label = convert_to_onehot(s_sca_label, class_num=class_num)
        t_vec_label = t_label.cpu().data.numpy()
        s_vec_label_own = s_vec_label[:, src_class_own]
        t_vec_label_own = t_vec_label[:, tar_class_own]

        # mmd_ids for own classes
        tmp_col_num = len(src_class_own) * len(tar_class_own)
        s_vec_label_final = np.zeros([s_vec_label_own.shape[0], tmp_col_num])
        t_vec_label_final = np.zeros([t_vec_label_own.shape[0], tmp_col_num])
        for cs in range(len(src_class_own)):
            idx = (np.arange(cs * len(tar_class_own), (cs + 1) * len(tar_class_own))).tolist()
            s_vec_label_final[:, idx] = np.tile(s_vec_label_own[:, cs].reshape(-1, 1), len(tar_class_own))
            t_vec_label_final[:, idx] = t_vec_label_own

        s_vec_weight_dis = cal_basic_weight(s_vec_label_final, tmp_col_num)
        t_vec_weight_dis = cal_basic_weight(t_vec_label_final, tmp_col_num)

        weight_ss_dis = s_vec_weight_dis @ s_vec_weight_dis.T / tmp_col_num
        weight_tt_dis = t_vec_weight_dis @ t_vec_weight_dis.T / tmp_col_num
        weight_st_dis = s_vec_weight_dis @ t_vec_weight_dis.T / tmp_col_num
        weight_ss_dis = weight_ss_dis.astype('float32')
        weight_tt_dis = weight_tt_dis.astype('float32')
        weight_st_dis = weight_st_dis.astype('float32')

    else:

        # 取交集，batch内同时出现的类别，然后遮蔽S和T没有同时出现的类别
        src_class_own = [c for c in list(np.unique(s_sca_label)) if c not in com_class]
        tar_class_own = [c for c in list(np.unique(t_sca_label)) if c not in com_class]

        # onehot label
        s_vec_label = convert_to_onehot(s_sca_label, class_num=class_num)
        t_vec_label = t_label.cpu().data.numpy()
        s_vec_label_com = s_vec_label[:, com_class]
        s_vec_label_own = s_vec_label[:, src_class_own]
        t_vec_label_com = t_vec_label[:, com_class]
        t_vec_label_own = t_vec_label[:, tar_class_own]

        # 首先条件MMD是在S和T共同的类别集合com_C上进行的，其他不相交的类别为0
        # 这一块加在某个样本x组合上的权重应该包括四部分：
        # 1）1/某类别c概率总和，或者c类样本的总数量，衡量样本x在其对应类别总量上的权重
        # 2）c类样本的总数量 / 对应于com_C的样本总数
        # 3）对应于com_C的样本总数/batch大小，表示batch内关注的样本数权重
        # 4）num_com_C/num_all_C，表示batch内关注的样本数占总类别数的权重，这个加不加看效果

        # for transferability
        class_num_com = len(com_class)
        s_vec_weight = cal_basic_weight(s_vec_label_com, class_num_com)
        t_vec_weight = cal_basic_weight(t_vec_label_com, class_num_com)

        weight_ss = s_vec_weight @ s_vec_weight.T
        weight_tt = t_vec_weight @ t_vec_weight.T
        weight_st = s_vec_weight @ t_vec_weight.T
        # print(s_vec_weight.shape, t_vec_weight.T.shape, weight_ss.shape)
        # (32, 31)(31, 32)(32, 32)

        length = len(com_class)
        weight_ss = (weight_ss / length).astype('float32')
        weight_tt = (weight_tt / length).astype('float32')
        weight_st = (weight_st / length).astype('float32')

        # for discriminability
        if (len(src_class_own) == 0) | len(tar_class_own) == 0:
            weight_ss_dis = np.array([0])
            weight_tt_dis = np.array([0])
            weight_st_dis = np.array([0])
        else:
            # mmd_ids for own classes
            tmp_col_num = len(src_class_own) * len(tar_class_own)
            s_vec_label_own_final = np.zeros([s_vec_label_own.shape[0], tmp_col_num])
            t_vec_label_own_final = np.zeros([t_vec_label_own.shape[0], tmp_col_num])
            for cs in range(len(src_class_own)):
                idx = (np.arange(cs * len(tar_class_own), (cs + 1) * len(tar_class_own))).tolist()
                s_vec_label_own_final[:, idx] = np.tile(s_vec_label_own[:, cs].reshape(-1, 1), len(tar_class_own))
                t_vec_label_own_final[:, idx] = t_vec_label_own

            # mmd_ids for com classes
            tmp_col_num_com = class_num_com * (class_num_com - 1)
            s_vec_label_com_final = np.zeros([s_vec_label_com.shape[0], tmp_col_num_com])
            t_vec_label_com_final = np.zeros([t_vec_label_com.shape[0], tmp_col_num_com])
            for cs in range(len(com_class)):
                idx = (np.arange(cs * (class_num_com - 1), (cs + 1) * (class_num_com - 1))).tolist()
                s_vec_label_com_final[:, idx] = np.tile(s_vec_label_com[:, cs].reshape(-1, 1), class_num_com - 1)
                tmp_t_vec_label_com = t_vec_label_com.copy()
                tmp_t_vec_label_com = np.delete(tmp_t_vec_label_com, cs, axis=1)
                t_vec_label_com_final[:, idx] = tmp_t_vec_label_com

            s_vec_label_final = np.concatenate([s_vec_label_own_final, s_vec_label_com_final], axis=1)
            t_vec_label_final = np.concatenate([t_vec_label_own_final, t_vec_label_com_final], axis=1)
            all_col_num = tmp_col_num + tmp_col_num_com
            s_vec_weight_dis = cal_basic_weight(s_vec_label_final, all_col_num)
            t_vec_weight_dis = cal_basic_weight(t_vec_label_final, all_col_num)

            weight_ss_dis = s_vec_weight_dis @ s_vec_weight_dis.T / all_col_num
            weight_tt_dis = t_vec_weight_dis @ t_vec_weight_dis.T / all_col_num
            weight_st_dis = s_vec_weight_dis @ t_vec_weight_dis.T / all_col_num
            weight_ss_dis = weight_ss_dis.astype('float32')
            weight_tt_dis = weight_tt_dis.astype('float32')
            weight_st_dis = weight_st_dis.astype('float32')

    return [weight_ss, weight_tt, weight_st], [weight_ss_dis, weight_tt_dis, weight_st_dis]


def LMMD_loss(source, target, s_label, t_label, args):
    # default paras
    args.kernel_num = 5
    args.kernel_mul = 2.0
    args.fix_sigma = None
    args.kernel_type = 'rbf'

    kernels = guassian_kernel(source, target, kernel_mul=args.kernel_mul, kernel_num=args.kernel_num,
                              fix_sigma=args.fix_sigma)

    batch_size = source.size()[0]
    f_return0, f_return1 = cal_weight(s_label, t_label, class_num=args.class_num)

    weight_ss, weight_tt, weight_st = f_return0[0], f_return0[1], f_return0[2]
    weight_ss_dis, weight_tt_dis, weight_st_dis = f_return1[0], f_return1[1], f_return1[2]

    weight_ss = torch.from_numpy(weight_ss).cuda()
    weight_tt = torch.from_numpy(weight_tt).cuda()
    weight_st = torch.from_numpy(weight_st).cuda()

    weight_ss_dis = torch.from_numpy(weight_ss_dis).cuda()
    weight_tt_dis = torch.from_numpy(weight_tt_dis).cuda()
    weight_st_dis = torch.from_numpy(weight_st_dis).cuda()

    loss_mmd1 = torch.Tensor([0]).cuda()
    loss_mmd2 = torch.Tensor([0]).cuda()
    if torch.sum(torch.isnan(sum(kernels))):
        return loss_mmd1, loss_mmd2

    SS = kernels[:batch_size, :batch_size]
    TT = kernels[batch_size:, batch_size:]
    ST = kernels[:batch_size, batch_size:]

    loss_mmd1 += torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
    loss_mmd2 += torch.sum(weight_ss_dis * SS + weight_tt_dis * TT - 2 * weight_st_dis * ST)

    return loss_mmd1, loss_mmd2


if __name__ == '__main__':
    import argparse

    args = argparse.Namespace(dset='office')
    args.class_num = 31

    # loss_mmd = LMMD_loss(source, target, s_label, t_label, args)
