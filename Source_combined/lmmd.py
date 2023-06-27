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


def cal_weight(s_label, t_label, class_num=31):
    batch_size = s_label.size()[0]
    s_sca_label = s_label.cpu().data.numpy()
    s_vec_label = convert_to_onehot(s_sca_label, class_num=class_num)
    s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)
    s_sum[s_sum == 0] = 100
    s_vec_label = s_vec_label / s_sum

    t_sca_label = t_label.cpu().data.max(1)[1].numpy()
    t_vec_label = t_label.cpu().data.numpy()
    t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
    t_sum[t_sum == 0] = 100
    t_vec_label = t_vec_label / t_sum

    # 取交集，batch内同时出现的类别，然后遮蔽S和T没有同时出现的类别
    index = list(set(s_sca_label) & set(t_sca_label))
    mask_arr = np.zeros((batch_size, class_num))
    mask_arr[:, index] = 1
    t_vec_label = t_vec_label * mask_arr
    s_vec_label = s_vec_label * mask_arr

    weight_ss = s_vec_label @ s_vec_label.T
    weight_tt = t_vec_label @ t_vec_label.T
    weight_st = s_vec_label @ t_vec_label.T
    # print(s_vec_label.shape, s_vec_label.T.shape, weight_ss.shape)
    # (32, 31)(31, 32)(32, 32)

    # 首先条件MMD是在S和T共同的类别集合com_C上进行的，其他不相交的类别为0
    # 这一块加在某个样本x组合上的权重应该包括三部分：
    # 1）1/某类别c概率总和，或者c类样本的总数量，衡量样本x在其对应类别总量上的权重
    # 2）对应于com_C的样本总数/batch大小，表示batch内关注的样本数权重
    # 3）num_com_C/num_all_C，表示batch内关注的样本数占总类别数的权重，这个加不加看效果
    length = len(index)
    if length != 0:
        weight_ss = weight_ss / length
        weight_tt = weight_tt / length
        weight_st = weight_st / length
    else:
        weight_ss = np.array([0])
        weight_tt = np.array([0])
        weight_st = np.array([0])
    return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')


def LMMD_loss(source, target, s_label, t_label, args):
    # default paras
    args.kernel_num = 5
    args.kernel_mul = 2.0
    args.fix_sigma = None
    args.kernel_type = 'rbf'

    kernels = guassian_kernel(source, target, kernel_mul=args.kernel_mul, kernel_num=args.kernel_num,
                              fix_sigma=args.fix_sigma)

    batch_size = source.size()[0]
    weight_ss, weight_tt, weight_st = cal_weight(s_label, t_label, class_num=args.class_num)
    weight_ss = torch.from_numpy(weight_ss).cuda()
    weight_tt = torch.from_numpy(weight_tt).cuda()
    weight_st = torch.from_numpy(weight_st).cuda()

    loss = torch.Tensor([0]).cuda()
    if torch.sum(torch.isnan(sum(kernels))):
        return loss
    SS = kernels[:batch_size, :batch_size]
    TT = kernels[batch_size:, batch_size:]
    ST = kernels[:batch_size, batch_size:]

    loss += torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
    return loss


if __name__ == '__main__':
    import argparse

    args = argparse.Namespace(dset='office')
    args.class_num = 31

    # loss_mmd = LMMD_loss(source, target, s_label, t_label, args)
