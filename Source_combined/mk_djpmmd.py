# -*- coding: utf-8 -*-
# @Time    : 2020/7/4 19:18
# @Author  : wenzhang
# @File    : djp_mmd.py
import torch as tr
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb
from torch.autograd import Function
from typing import Optional, Sequence


def _primal_kernel(Xs, Xt):
    Z = tr.cat((Xs.T, Xt.T), 1)  # Xs / Xt: batch_size * k
    return Z


def _linear_kernel(Xs, Xt):
    Z = tr.cat((Xs, Xt), 0)  # Xs / Xt: batch_size * k
    K = tr.mm(Z, Z.T)
    return K


def _rbf_kernel(Xs, Xt, sigma):
    Z = tr.cat((Xs, Xt), 0)
    ZZT = tr.mm(Z, Z.T)
    diag_ZZT = tr.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.T
    K = tr.exp(-exponent / (2 * sigma ** 2))
    return K


# functions to compute the marginal MMD with rbf kernel
def rbf_mmd(Xs, Xt, args):
    args.sigma = 1

    K = _rbf_kernel(Xs, Xt, args.sigma)
    m = Xs.size(0)  # assume Xs, Xt are same shape
    e = tr.cat((1 / m * tr.ones(m, 1), -1 / m * tr.ones(m, 1)), 0)
    M = e * e.T
    tmp = tr.mm(tr.mm(K.cpu(), M.cpu()), K.T.cpu())
    loss = tr.trace(tmp).cuda()
    return loss


# functions to compute rbf kernel JMMD
def rbf_jmmd(Xs, Xt, Ys, Yt0, args):
    C = args.class_num
    args.sigma = 1

    K = _rbf_kernel(Xs, Xt, args.sigma)
    n = K.size(0)
    m = Xs.size(0)  # assume Xs, Xt are same shape
    e = tr.cat((1 / m * tr.ones(m, 1), -1 / m * tr.ones(m, 1)), 0)
    M = e * e.T * C
    for c in range(C):
        e = tr.zeros(n, 1)
        if len(Ys[Ys == c]) == 0:
            e[:m][Ys == c] = 0
        else:
            e[:m][Ys == c] = 1 / len(Ys[Ys == c])

        if len(Yt0[Yt0 == c]) == 0:
            e[m:][Yt0 == c] = 0
        else:
            e[m:][Yt0 == c] = -1 / len(Yt0[Yt0 == c])
        M = M + e * e.T
    M = M / tr.norm(M, p='fro')  # can reduce the training loss only for jmmd
    tmp = tr.mm(tr.mm(K.cpu(), M.cpu()), K.T.cpu())
    loss = tr.trace(tmp).cuda()
    return loss


# functions to compute rbf kernel JPMMD
def rbf_jpmmd(Xs, Xt, Ys, Yt0, args):
    C = args.class_num
    args.sigma = 1

    K = _rbf_kernel(Xs, Xt, args.sigma)
    n = K.size(0)
    m = Xs.size(0)  # assume Xs, Xt are same shape
    M = 0
    for c in range(C):
        e = tr.zeros(n, 1)
        e[:m] = 1 / len(Ys)
        e[m:] = 0 if len(Yt0[Yt0 == c]) == 0 else -1 / len(Yt0)
        M = M + e * e.T
    # M = M / tr.norm(M, p='fro')
    tmp = tr.mm(tr.mm(K.cpu(), M.cpu()), K.T.cpu())
    loss = tr.trace(tmp).cuda()

    return loss


def raw_djpmmd(Xs, Xt, Ys, Yt0, args):
    C = args.class_num
    args.sigma = 1
    
    # 单核版本
    # K = _rbf_kernel(Xs, Xt, args.sigma)
    # K = _linear_kernel(Xs, Xt)  # bad performance
    
    # 多核版本，实现方式1
    kernels = [_rbf_kernel(Xs, Xt, 2 ** k) for k in range(-3, 2)]
    K = sum(kernels)

    # 多核版本，实现方式2
    # from mmd import guassian_kernel
    # args.kernel_num = 5
    # args.kernel_mul = 2.0
    # args.fix_sigma = None
    # args.kernel_type = 'rbf'
    # K = guassian_kernel(Xs, Xt, kernel_mul=args.kernel_mul, kernel_num=args.kernel_num,
    #                     fix_sigma=args.fix_sigma)

    m = Xs.size(0)

    # For transferability
    Ns = 1 / m * tr.zeros(m, C).scatter_(1, Ys.unsqueeze(1).cpu(), 1)
    Nt = tr.zeros(m, C)
    if len(tr.unique(Yt0)) == 1:
        Nt = 1 / m * tr.zeros(m, C).scatter_(1, Yt0.unsqueeze(1).cpu(), 1)
    Rmin = tr.cat((tr.cat((tr.mm(Ns, Ns.T), tr.mm(-Ns, Nt.T)), 0),
                   tr.cat((tr.mm(-Nt, Ns.T), tr.mm(Nt, Nt.T)), 0)), 1)
    # Rmin = Rmin / tr.norm(Rmin, p='fro')

    # For discriminability
    Ms, Mt = tr.empty(m, (C - 1) * C), tr.empty(m, (C - 1) * C)
    for i in range(C):
        idx = tr.arange((C - 1) * i, (C - 1) * (i + 1))
        Ms[:, idx] = Ns[:, i].repeat(C - 1, 1).T
        tmp = tr.arange(0, C)
        Mt[:, idx] = Nt[:, tmp[tmp != i]]
    Rmax = tr.cat((tr.cat((tr.mm(Ms, Ms.T), tr.mm(-Ms, Mt.T)), 0),
                   tr.cat((tr.mm(-Mt, Ms.T), tr.mm(Mt, Mt.T)), 0)), 1)
    # Rmax = Rmax / tr.norm(Rmax, p='fro')

    M = Rmin - 0.1 * Rmax
    tmp = tr.mm(tr.mm(K.cpu(), M.cpu()), K.T.cpu())
    loss = tr.trace(tmp.cuda())

    return loss


# functions to compute rbf kernel DJP-MMD
def mk_djpmmd(Xs, Xt, Ys, Yt0, args):
    C = args.class_num
    args.sigma = 1

    # 单核版本
    # K = _rbf_kernel(Xs, Xt, args.sigma)
    # K = _linear_kernel(Xs, Xt)  # bad performance

    # 多核版本，实现方式1
    features = tr.cat([Xs, Xt], dim=0)
    kernels = [GaussianKernel(alpha=2 ** k) for k in range(-3, 2)]
    K = sum([kernel(features) for kernel in kernels])

    # 多核版本，实现方式2
    # from mmd import guassian_kernel
    # args.kernel_num = 5
    # args.kernel_mul = 2.0
    # args.fix_sigma = None
    # args.kernel_type = 'rbf'
    # K = guassian_kernel(Xs, Xt, kernel_mul=args.kernel_mul, kernel_num=args.kernel_num,
    #                     fix_sigma=args.fix_sigma)

    m = Xs.size(0)

    # For transferability
    Ns = 1 / m * tr.zeros(m, C).scatter_(1, Ys.unsqueeze(1).cpu(), 1)
    Nt = tr.zeros(m, C)
    if len(tr.unique(Yt0)) == 1:
        Nt = 1 / m * tr.zeros(m, C).scatter_(1, Yt0.unsqueeze(1).cpu(), 1)
    Rmin = tr.cat((tr.cat((tr.mm(Ns, Ns.T), tr.mm(-Ns, Nt.T)), 0),
                   tr.cat((tr.mm(-Nt, Ns.T), tr.mm(Nt, Nt.T)), 0)), 1)
    # Rmin = Rmin / tr.norm(Rmin, p='fro')

    # For discriminability
    Ms, Mt = tr.empty(m, (C - 1) * C), tr.empty(m, (C - 1) * C)
    for i in range(C):
        idx = tr.arange((C - 1) * i, (C - 1) * (i + 1))
        Ms[:, idx] = Ns[:, i].repeat(C - 1, 1).T
        tmp = tr.arange(0, C)
        Mt[:, idx] = Nt[:, tmp[tmp != i]]
    Rmax = tr.cat((tr.cat((tr.mm(Ms, Ms.T), tr.mm(-Ms, Mt.T)), 0),
                   tr.cat((tr.mm(-Mt, Ms.T), tr.mm(Mt, Mt.T)), 0)), 1)
    # Rmax = Rmax / tr.norm(Rmax, p='fro')

    M = Rmin - 0.1 * Rmax
    tmp = tr.mm(tr.mm(K.cpu(), M.cpu()), K.T.cpu())
    loss = tr.trace(tmp.cuda())

    return loss


# =============================================================DAN Function=============================================
class MultipleKernelMaximumMeanDiscrepancy(nn.Module):
    r"""
    Args:
        kernels (tuple(tr.nn.Module)): kernel functions.
        linear (bool): whether use the linear version of DAN. Default: False

    Inputs:
        - z_s (tensor): activations from the source domain, :math:`z^s`
        - z_t (tensor): activations from the target domain, :math:`z^t`
    """

    def __init__(self, kernels: Sequence[nn.Module], linear: Optional[bool] = False):
        super(MultipleKernelMaximumMeanDiscrepancy, self).__init__()
        self.kernels = kernels
        self.index_matrix = None
        self.linear = linear

    def forward(self, z_s: tr.Tensor, z_t: tr.Tensor) -> tr.Tensor:
        features = tr.cat([z_s, z_t], dim=0)
        batch_size = int(z_s.size(0))
        self.index_matrix = _update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s.device)

        kernel_matrix = sum([kernel(features) for kernel in self.kernels])  # Add up the matrix of each kernel
        # Add 2 / (n-1) to make up for the value on the diagonal
        # to ensure loss is positive in the non-linear version
        loss = (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)

        return loss


def _update_index_matrix(batch_size: int, index_matrix: Optional[tr.Tensor] = None,
                         linear: Optional[bool] = True) -> tr.Tensor:
    r"""
    Update the `index_matrix` which convert `kernel_matrix` to loss.
    If `index_matrix` is a tensor with shape (2 x batch_size, 2 x batch_size), then return `index_matrix`.
    Else return a new tensor with shape (2 x batch_size, 2 x batch_size).
    """
    if index_matrix is None or index_matrix.size(0) != batch_size * 2:
        index_matrix = tr.zeros(2 * batch_size, 2 * batch_size)
        if linear:
            for i in range(batch_size):
                s1, s2 = i, (i + 1) % batch_size
                t1, t2 = s1 + batch_size, s2 + batch_size
                index_matrix[s1, s2] = 1. / float(batch_size)
                index_matrix[t1, t2] = 1. / float(batch_size)
                index_matrix[s1, t2] = -1. / float(batch_size)
                index_matrix[s2, t1] = -1. / float(batch_size)
        else:
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j:
                        index_matrix[i][j] = 1. / float(batch_size * (batch_size - 1))
                        index_matrix[i + batch_size][j + batch_size] = 1. / float(batch_size * (batch_size - 1))
            for i in range(batch_size):
                for j in range(batch_size):
                    index_matrix[i][j + batch_size] = -1. / float(batch_size * batch_size)
                    index_matrix[i + batch_size][j] = -1. / float(batch_size * batch_size)
    return index_matrix


class GaussianKernel(nn.Module):
    r"""Gaussian Kernel Matrix
    Args:
        sigma (float, optional): bandwidth :math:`\sigma`. Default: None
        track_running_stats (bool, optional): If ``True``, this module tracks the running mean of :math:`\sigma^2`.
          Otherwise, it won't track such statistics and always uses fix :math:`\sigma^2`. Default: ``True``
        alpha (float, optional): :math:`\alpha` which decides the magnitude of :math:`\sigma^2` when track_running_stats is set to ``True``

    Inputs:
        - X (tensor): input group :math:`X`

    Shape:
        - Inputs: :math:`(minibatch, F)` where F means the dimension of input features.
        - Outputs: :math:`(minibatch, minibatch)`
    """

    def __init__(self, sigma: Optional[float] = None, track_running_stats: Optional[bool] = True,
                 alpha: Optional[float] = 1.):
        super(GaussianKernel, self).__init__()
        assert track_running_stats or sigma is not None
        self.sigma_square = tr.tensor(sigma * sigma) if sigma is not None else None
        self.track_running_stats = track_running_stats
        self.alpha = alpha

    def forward(self, X: tr.Tensor) -> tr.Tensor:
        l2_distance_square = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).sum(2)

        if self.track_running_stats:
            self.sigma_square = self.alpha * tr.mean(l2_distance_square.detach())

        return tr.exp(-l2_distance_square / (2 * self.sigma_square))

