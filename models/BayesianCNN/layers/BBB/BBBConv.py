import sys
sys.path.append("..")

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from ..misc import ModuleWrapper


def KL_DIV(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl

class BBBConv2d(ModuleWrapper):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1,
                 stride=1, padding=0, dilation=1, bias=True, priors=None):

        super(BBBConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-3, 0.1),
            }
        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.W_mu = []
        self.W_rho = []
        for i in range(self.groups):
            self.W_mu.append(Parameter(torch.empty((out_channels // self.groups, in_channels // self.groups, *self.kernel_size), device=self.device)))
            self.W_rho.append(Parameter(torch.empty((out_channels // self.groups, in_channels // self.groups, *self.kernel_size), device=self.device)))

        if self.use_bias:
            self.bias_mu = []
            self.bias_rho = []
            for i in range(self.groups):
                self.bias_mu.append(Parameter(torch.empty((out_channels // self.groups), device=self.device)))
                self.bias_rho.append(Parameter(torch.empty((out_channels // self.groups), device=self.device)))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.groups):
            self.W_mu[i].data.normal_(*self.posterior_mu_initial)
            self.W_rho[i].data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            for i in range(self.groups):
                self.bias_mu[i].data.normal_(*self.posterior_mu_initial)
                self.bias_rho[i].data.normal_(*self.posterior_rho_initial)

    def forward(self, input, sample=True):
        if self.training or sample:
            self.W_sigma = []
            self.bias_sigma = []
            for i in range(self.groups):
                W_eps = torch.empty(self.W_mu[i].size()).normal_(0, 1).to(self.device)
                self.W_sigma.append(torch.log1p(torch.exp(self.W_rho[i])))
                weight = self.W_mu[i] + W_eps * self.W_sigma[i]

                if self.use_bias:
                    bias_eps = torch.empty(self.bias_mu[i].size()).normal_(0, 1).to(self.device)
                    self.bias_sigma.append(torch.log1p(torch.exp(self.bias_rho[i])))
                    bias = self.bias_mu[i] + bias_eps * self.bias_sigma[i]
                else:
                    bias = None
        else:
            for i in range(self.groups):
                weight = self.W_mu[i]
                bias = self.bias_mu[i] if self.use_bias else None
        print(self.groups)
        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def kl_loss(self):
        kl = KL_DIV(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            for i in range(self.groups):
                kl += KL_DIV(self.prior_mu, self.prior_sigma, self.bias_mu[i], self.bias_sigma[i])  # TODO
        return kl
