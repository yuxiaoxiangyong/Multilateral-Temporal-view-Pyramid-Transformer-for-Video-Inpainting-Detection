import os
import math
import torch
import logging
import numpy as np
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from einops import rearrange


class Filter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=False, norm=False, fine_grain=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable
        if fine_grain:
            self.base = torch.tensor(generate_fine_grained_filter(band_start, band_end, size)).cuda()
        else:
            self.base = torch.tensor(generate_filter(band_start, band_end, size)).cuda()

        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))),
                                       requires_grad=False)

    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y


def DCT_mat(size):
    m = [[np.sqrt(1. / size) if i == 0 else np.sqrt(2. / size) * np.cos((j + 0.5) * np.pi * i / size) for j in
          range(size)] for i in range(size)]
    return m


def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j < start else 1. for j in range(size)] for i in range(size)]


def generate_fine_grained_filter(start, end, size):
    return [[0. if i != start or j != end else 1. for j in range(size)] for i in range(size)]


class FAF(nn.Module):
    def __init__(self, size=224):
        super(FAF, self).__init__()
        self.fn = 3
        # init DCT matrix
        self._DCT_all = torch.tensor(DCT_mat(size)).float().cuda()
        self._DCT_all_T = torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1).cuda()

        # define base filters and learnable
        # 0 - 1/16 || 1/16 - 1/8 || 1/8 - 1
        low_filter = Filter(size, 0, size // 2.82)  # 2.82
        middle_filter = Filter(size, size // 2.82, size // 2)  # 2
        high_filter = Filter(size, size * 1, size * 2)
        self.filters = nn.ModuleList([low_filter, middle_filter, high_filter])

    def forward(self, x):
        x_freq = self._DCT_all @ x @ self._DCT_all_T  # [N, 3, 299, 299]
        y_list = []
        for i in range(self.fn):
            x_pass = self.filters[i](x_freq)  # [N, 3, 299, 299]
            y = self._DCT_all_T @ x_pass @ self._DCT_all  # [N, 3, 299, 299]
            y_list.append(y)
        out = torch.cat(y_list, dim=-3)  # [N, 12, 299, 299]
        return out

