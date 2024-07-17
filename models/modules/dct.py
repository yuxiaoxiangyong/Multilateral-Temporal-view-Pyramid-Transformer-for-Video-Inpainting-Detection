import logging
import math
import os
import numpy as np 

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from einops import rearrange
#from modules.timm.models.vision_transformer import VisionTransformer, PatchEmbed, Block
import math 
#import timm

# Filter Module
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
            # Todo
            # self.learnable = nn.Parameter(torch.rand((size, size)) * 0.2 - 0.1, requires_grad=True)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))), requires_grad=False)
        #print(self.base.shape)

    def forward(self, x):
        #if self.use_learnable:
            #filt = self.base + norm_sigma(self.learnable)
        #else:
        filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y

# 生成 DCT transform 矩阵
def DCT_mat(size):
    m = [[ np.sqrt(1./size) if i == 0 else np.sqrt(2./size) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m

def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j < start else 1. for j in range(size)] for i in range(size)]

def generate_fine_grained_filter(start, end, size):
    return [[0. if i != start or j != end else 1. for j in range(size)] for i in range(size)]

# FAF Module
class FAF(nn.Module):
    def __init__(self, size=224):
        super(FAF, self).__init__()

        # init DCT matrix
        self._DCT_all = torch.tensor(DCT_mat(size)).float().cuda()
        self._DCT_all_T = torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1).cuda()

        # define base filters and learnable
        # 0 - 1/16 || 1/16 - 1/8 || 1/8 - 1
        # 高通 带通 低通 
        low_filter = Filter(size, 0, size // 2.82)   # 2.82
        middle_filter = Filter(size, size // 2.82, size // 2) # 2
        high_filter = Filter(size, size * 1, size * 2)
        #all_filter = Filter(size, 0, size * 2)

        #self.filters = nn.ModuleList([low_filter, middle_filter, high_filter, all_filter])
        self.filters = nn.ModuleList([low_filter, middle_filter, high_filter])

    def forward(self, x):
        # DCT
        x_freq = self._DCT_all @ x @ self._DCT_all_T    # [N, 3, 299, 299]
        
        # 4 kernel
        y_list = []
        for i in range(3):
            x_pass = self.filters[i](x_freq)  # [N, 3, 299, 299]
            y = self._DCT_all_T @ x_pass @ self._DCT_all    # [N, 3, 299, 299]
            y_list.append(y)
        out = torch.cat(y_list, dim=-3)    # [N, 12, 299, 299] 测试改成dim = 0
        return out  # 三个通道都考虑

class DCT_(nn.Module):
    def __init__(self, size):
        super(DCT_, self).__init__()
        self._DCT_all = torch.tensor(DCT_mat(size)).float().cuda()
        self._DCT_all_T = torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1).cuda()
        
    def forward(self, x):
        # DCT
        x_freq = self._DCT_all @ x @ self._DCT_all_T    # [N, 3, 299, 299]
        return x_freq  

class IDCT_(nn.Module):
    def __init__(self, size):
        super(IDCT_, self).__init__()
        self.size=size
        self._DCT_all = torch.tensor(DCT_mat(size)).float().cuda()
        self._DCT_all_T = torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1).cuda()
        low_filter = Filter(size, 0, size // 2.82)   # 2.82
        middle_filter = Filter(size, size // 2.82 , size) # 2
        high_filter = Filter(size, size // 1, size * 2)
        self.filters =  [low_filter, middle_filter, high_filter] #[low_filter, middle_filter, high_filter]
        #self.filters = [[Filter(size, i, j, fine_grain=True) for j in range(size)] for i in range(size)]
    def forward(self, x):
        idct_y = []
        for i in range(len(self.filters)):
                filterd_x = self.filters[i](x)
                res = self._DCT_all_T @ filterd_x @ self._DCT_all
                #if i == 2:
                    #res = 1 - res
                idct_y.append(res.unsqueeze(-4))
        out = torch.cat(idct_y, dim=-4)
        # visualization
        out = rearrange(out, 'b t (h w) n c p1 p2 -> b t (n c) (h p1) (w p2)', h=56)
        # tranfer to model
        #out = rearrange(out, 'b t (h w) n c p1 p2 -> b t (n c p1 p2) h w', h=56)
        #out = rearrange(out, 'b t (n c p1 p2) h w -> b t n c (h p1) (w p2)', p1=4, )
        #print(out.shape)
        #print(self.filters[0][0](x).shape) # [1, 1, 3136, 3, 4, 4]
        #flitered_x = torch.cat()
        # IDCT
        #x_freq = self._DCT_all_T @ x @ self._DCT_all    # [N, 3, 299, 299]
        #return x_freq 
        return out

if __name__ == "__main__":
    
    y = generate_filter(0, 1, 4)
    print(y)
    x = torch.ones((1, 1, 3136, 3, 4, 4)).cuda()
    idct = IDCT_(size=4)
    output = idct(x)


    
