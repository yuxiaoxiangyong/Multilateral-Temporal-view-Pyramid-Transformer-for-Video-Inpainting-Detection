import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models.layers import DropPath


class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return rearrange(x, 'b h w c -> b c h w')


'''
    The DAttention is used for global cross attention. 
'''
class DAttention(nn.Module):
    def __init__(self, 
                 dim1,
                 n_heads,  
                 attn_drop,
                 n_groups,
                 feature_size = 56,
                 cur_stage = 0,
                 stride = 1, 
                 offset_range_factor = 2, 
                 no_off = False,
                 height_scale = [1, 1],
                 dwc_pe = False,
                 use_pe = False,
                 fixed_pe = False
                ):
        super().__init__()
        self.n_head_channels = dim1 // n_heads
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.fs = feature_size

        self.nc = self.n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        
        self.heigth_scale = height_scale
        self.cur_stage = cur_stage
        self.dwc_pe = dwc_pe
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe

        kk = 5

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, kk//2, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )

        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )   

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_drop = nn.Dropout(attn_drop, inplace=False)
        self.attn_drop = nn.Dropout(attn_drop, inplace=False)

        if self.use_pe:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(self.nc, self.nc,
                                           kernel_size=3, stride=1, padding=1, groups=self.nc)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, 196, 196)
                )
                trunc_normal_(self.rpe_table, std=0.01)
            else:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, 2 * 14 * self.heigth_scale[1] - 1, 2 * 14 - 1) # 相对位置的取值范围
                )
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

        # initialize
        nn.init.trunc_normal_(self.proj_q.weight)
        nn.init.zeros_(self.proj_q.bias)
        nn.init.trunc_normal_(self.proj_k.weight)
        nn.init.zeros_(self.proj_k.bias)
        nn.init.trunc_normal_(self.proj_v.weight)
        nn.init.zeros_(self.proj_v.bias)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
    
    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device), 
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device)
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)     # [-1, 1]
        ref[..., 0].div_(H_key).mul_(2).sub_(1)     # [-1, 1]
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g H W 2
        
        return ref

    def forward(self, x1, x2, return_attention=False):
        # x1 x2 ==> B * N * C
        base_size = int(self.fs * math.pow(2, -self.cur_stage))
        height_scale_x1 = self.heigth_scale[0]
        height_scale_x2 = self.heigth_scale[1]
        x1 = rearrange(x1, 'b (h w) c -> b c h w', h = base * height_scale_x1, w = base_size)
        x2 = rearrange(x2, 'b (h w) c -> b c h w', h = base * height_scale_x2, w = base_size)
        B, C, H, W = x1.size()
        dtype, device = x1.dtype, x1.device
        
        q = self.proj_q(x1)
        q_off = rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off) # B * g 2 Hg Wg
        Hk, Wk = offset.size(2), offset.size(3) # H, W
        n_sample = Hk * Wk

        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = rearrange(offset, 'b p h w -> b h w p') # B * g H W 2
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)  # B * g H W 2
        
        if self.no_off:
            offset = offset.fill(0.0)

        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).tanh() # B*g * H * W * 2

        x_sampled = F.grid_sample(
            input=x2.reshape(B * self.n_groups, self.n_group_channels, H * height_scale_x2, W),
            grid=pos[..., (1, 0)], # y, x -> x, y
            mode='bilinear', align_corners=True) # B * g, Cg, Hg, Wg
            
        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W).permute(0, 2, 1)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample).permute(0, 2, 1)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample).permute(0, 2, 1)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # N1 * C x C * N2

        if self.use_pe:
            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_heads, self.n_head_channels, H * W)
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.n_heads, H * W, n_sample)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                
                q_grid = self._get_ref_points(H, W, B, dtype, device)
                
                displacement = (q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)).mul(0.5)
                
                attn_bias = F.grid_sample(
                    input=rpe_bias.reshape(B * self.n_groups, self.n_group_heads, 2 * H * self.heigth_scale[1] - 1, 2 * W - 1),
                    grid=displacement[..., (1, 0)],
                    mode='bilinear', align_corners=True
                ) # B * g, h_g, HW, Ns

                attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)
                attn = attn + attn_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, C, H, W) # N1 * N2 x N2 * C ==> N1 * C
        attn = attn.reshape(B, -1, attn.shape[-2], attn.shape[-1])

        if self.use_pe and self.dwc_pe:
            x = x + residual_lepe

        x = self.proj_out(x)
        x = self.proj_drop(x).reshape(B, H*W, C)

        return x, attn


'''
    The SwinDAttention is used for window-based temporal view cross attention. 
'''
class SwinDAttention(nn.Module):
    def __init__(self, 
                 dim1,
                 n_heads,  
                 attn_drop,
                 n_groups,
                 ws = 7,
                 stride = 1, 
                 offset_range_factor = 2, 
                 no_off = False,
                 height_scale = [1, 1],
                 dwc_pe = False,
                 use_pe = False,
                 fixed_pe = False
                ):
        super().__init__()
        self.n_head_channels = dim1 // n_heads
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.ws = ws

        self.nc = self.n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        
        self.heigth_scale = height_scale
        self.dwc_pe = dwc_pe
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe

        kk = 5

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, kk//2, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )

        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )   

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_drop = nn.Dropout(attn_drop, inplace=False)
        self.attn_drop = nn.Dropout(attn_drop, inplace=False)

        if self.use_pe:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(self.nc, self.nc, 
                                           kernel_size=3, stride=1, padding=1, groups=self.nc)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, 196, 196)
                )
                trunc_normal_(self.rpe_table, std=0.01)
            else:
                # 相对位置编码表的范围没问题
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, 2 * 14 * self.heigth_scale[1] - 1, 2 * 14 - 1) # 相对位置的取值范围
                )
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

        # initialize
        nn.init.trunc_normal_(self.proj_q.weight)
        nn.init.zeros_(self.proj_q.bias)
        nn.init.trunc_normal_(self.proj_k.weight)
        nn.init.zeros_(self.proj_k.bias)
        nn.init.trunc_normal_(self.proj_v.weight)
        nn.init.zeros_(self.proj_v.bias)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device), 
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device)
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)     # [-1, 1]
        ref[..., 0].div_(H_key).mul_(2).sub_(1)     # [-1, 1]
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g H W 2
        
        return ref

    def forward(self, x1, x2, return_attention=False):
        # x1 x2 ==> B * N * C
        x1 = rearrange(x1, 'b (h w) c -> b c h w', h = self.ws, w = self.ws)
        x2 = rearrange(x2, 'b (h w) c -> b c h w', h = self.ws, w = self.ws)
        B, C, H, W = x2.size()
        ratio = int(B // x1.size()[0])
        x1 = x1.repeat(ratio, 1, 1, 1)
        dtype, device = x1.dtype, x1.device
        
        q = self.proj_q(x1)
        q_off = rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off) # B * g 2 Hg Wg
        Hk, Wk = offset.size(2), offset.size(3) # H, W
        n_sample = Hk * Wk
        if self.offset_range_factor > 0: #
            offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)
            
        offset = rearrange(offset, 'b p h w -> b h w p') # B * g H W 2
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)  # B * g H W 2
        
        if self.no_off:
            offset = offset.fill(0.0)
            
        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).tanh() # B*g * H * W * 2

        x_sampled = F.grid_sample(
            input=x2.reshape(B * self.n_groups, self.n_group_channels, H, W), 
            grid=pos[..., (1, 0)], # y, x -> x, y
            mode='bilinear', align_corners=True) # B * g, Cg, Hg, Wg
            
        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W).permute(0, 2, 1)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample).permute(0, 2, 1)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample).permute(0, 2, 1)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # N1 * C x C * N2

        if self.use_pe:
            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_heads, self.n_head_channels, H * W)
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.n_heads, H * W, n_sample)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                
                q_grid = self._get_ref_points(H, W, B, dtype, device)
                
                displacement = (q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)).mul(0.5)
                
                attn_bias = F.grid_sample(
                    input=rpe_bias.reshape(B * self.n_groups, self.n_group_heads, 2 * H * self.heigth_scale[1] - 1, 2 * W - 1),
                    grid=displacement[..., (1, 0)],
                    mode='bilinear', align_corners=True
                ) # B * g, h_g, HW, Ns

                attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)
                attn = attn + attn_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, C, H, W) # N1 * N2 x N2 * C ==> N1 * C
        x = rearrange(x, '(b t) c h w -> b t c h w', t=ratio)
        x = torch.sum(x, dim=1)
        B = x.size(0)
        attn = attn.reshape(B, -1, attn.shape[-2], attn.shape[-1])

        if self.use_pe and self.dwc_pe:
            x = x + residual_lepe

        x = self.proj_out(x)
        x = self.proj_drop(x).reshape(B, H*W, C)

        return x, attn