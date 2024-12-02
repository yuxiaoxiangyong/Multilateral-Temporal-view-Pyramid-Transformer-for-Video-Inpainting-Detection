import torch
import torch.nn as nn
import ml_collections
import functorch
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn.init import zeros_
from models.modules.blocks import Block
from models.modules.swinTransformer import WindowAttention, SwinTransformerBlock, ThreeViewPatchMerging, Mlp, \
    window_partition, window_reverse
from models.modules.dct import FAF
from models.modules.deformableAttention import SwinDAttention
import warnings
warnings.filterwarnings("ignore")


class CrossWindowAttention(nn.Module):
    """ 
        Window based cross view multi-head self attention (W-CVMSA) module with relative position bias.
        Structure: entire attention block
        It only supports non-shifted window.
        dim1: dim of view1
        dim2: dim of view2
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww    (y, x)
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0  #y
        relative_coords[:, :, 1] += self.window_size[1] - 1  # x
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv1 = nn.Linear(dim, dim, bias=qkv_bias)  # view1 : query
        self.qkv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)  # view2 : key&value
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        trunc_normal_(self.qkv1.weight, std=.02)
        trunc_normal_(self.qkv2.weight, std=.02)
        zeros_(self.proj.weight)
        zeros_(self.proj.bias)
        zeros_(self.qkv1.bias)
        zeros_(self.qkv2.bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2, mask=None):
        """
        Args:
            x1: input features with shape of (num_windows*B, N, C) from view1, view1 means smaller model with
        bigger view but less view tublets
            x2: input features with shape of (num_windows*B, N, C) from view2, view2 means bigger model with
        smaller view but more view tublets
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # N means a window contains the number of patches x1.window_size == x2.window_size dim of x1 equals to x2's
        # In some situations, the num of heads of view1 and view2 are different, we need make sure use the same heads
        # and dims to finish cross attention. In my opinion, the above desertion is false, view2 is used for key&value
        # which can be any shape.
        B1, N1, C = x1.shape
        B2, N2, _ = x2.shape
        ratio = B2 // B1

        qkv1 = (self.qkv1(x1).reshape(B1, N1, 1, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4))  # 1 * B1 * heads * N1 * dim
        qkv2 = (self.qkv2(x2).reshape(B2, N2, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4))  # 2 * B2 * heads * N2 * dim
        q, k, v = qkv1[0], qkv2[0], qkv2[1]  # Query : B1 * heads * N1 * dim   Key&Value: B2 * heads * N2 * dim
        q = q * self.scale
        q = q.repeat(ratio, 1, 1, 1)

        attn = (q @ k.transpose(-2, -1))  # B2 * heads * N1 * N2    N1 == N2
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        if B1 == B2:
            x = (attn @ v).transpose(1, 2).reshape(B1, N1, C)
        else:
            x = rearrange((attn @ v), '(b r) h n c -> b r n (h c)', r=ratio)
            x = torch.sum(x, dim=1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class CVAModule(nn.Module):
    def __init__(self, dim1, num_heads, window_size=7, temporal_dims=[],
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., cur_stage=0):
        super().__init__()
        self.crossattn = SwinDAttention(dim1, num_heads, attn_drop, n_groups=3)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x1, x2, mask=None, return_attention=False):
        y, attn = self.crossattn(x1, x2)
        if return_attention:
            return attn
        x1 = x1 + self.drop_path(y)
        return x1, attn


class CrossSwinBlock(nn.Module):
    r"""
        Cross Swin Transformer Block.
        Input_Resolution:
        Todo: Excute the cross attn between view1 and view2
        Tips: The Swin Stage usually contains a window attention block and a shift window attention block, and both appear in pairs, the cva
        is viewed as a module which can be apper with a window attention block or a shift window attention block, but the cva module only supports
        the window atention, because we argue that cva in the first window attention can pass the mesage to the following shift window attention block.
    """

    def __init__(self, dim1, dim2, input_resolution, num_heads,
                 window_size=7, shift_size=0, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False,
                 last_view=False,
                 temporal_dims=1,
                 cur_stage=0):
        super().__init__()
        self.dim = dim1
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.last_view = last_view  # no cross attention
        self.temporal_dims = temporal_dims
        self.cur_stage = cur_stage

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(self.dim)
        self.attn = WindowAttention(self.dim, window_size=to_2tuple(self.window_size),
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim1)
        mlp_hidden_dim = int(dim1 * mlp_ratio)
        self.mlp = Mlp(in_features=dim1, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.pre = nn.Identity() if self.last_view else nn.Linear(dim2, dim1)
        if not self.last_view:
            trunc_normal_(self.pre.weight, std=.02)
            zeros_(self.pre.bias)
        self.cva = nn.Identity() if self.last_view else CVAModule(dim1,
                                                                  temporal_dims=self.temporal_dims,
                                                                  window_size=to_2tuple(self.window_size),
                                                                  num_heads=num_heads,
                                                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                                  attn_drop=attn_drop, drop=drop,
                                                                  drop_path=drop_path, cur_stage=cur_stage)
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            # attn_mask : n * (ws * ws) * (ws * ws)
            H, W = self.input_resolution
            img_mask = torch.zeros((1, self.temporal_dims * H, W, 1))  # 1 t*H W 1  view B * N * C
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # nW, window_size * window_size
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(
                2)  # nW, window_size * window_size, window_size * window_size
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x1, x2):
        # H, W means the shape of x1
        # Cross Swin Block = Attention + Cross Attention
        H, W = self.input_resolution  # normal H & W, doesn't consider multi-view
        B1, L1, C1 = x1.shape
        B2, L2, C2 = x2.shape
        assert L1 % H * W == 0 and L2 % H * W == 0, "input feature has wrong size"
        T1, T2 = L1 / (H * W), L2 / (H * W)

        shortcut = x1
        x1 = self.norm1(x1)
        x1 = x1.view(B1, int(T1 * H), W, C1)
        x2 = x2.view(B2, int(T2 * H), W, C2)

        # cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                # implement shift window
                shifted_x1 = torch.roll(x1, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # partition windows
                x1_windows = window_partition(shifted_x1, self.window_size)  # nW*B, window_size, window_size, C1
            else:
                x1_windows = WindowProcess.apply(x1, B1, H, W, C1, -self.shift_size, self.window_size)  # Todo
        else:
            shifted_x1 = x1
            # partition windows
            x1_windows = window_partition(shifted_x1, self.window_size)  # nW*B, window_size, window_size, C1

        x1_windows = x1_windows.view(-1, self.window_size * self.window_size, C1)  # nW1*B1, window_size*window_size, C1
        # W-MSA/SW-MSA
        attn_windows = self.attn(x1_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C1)

        # reverse cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x1 = window_reverse(attn_windows, self.window_size, T1 * H, W)  # B H' W' C
                x1 = torch.roll(shifted_x1, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x1 = WindowProcessReverse.apply(attn_windows, B1, H, W, C1, self.shift_size, self.window_size)
        else:
            shifted_x1 = window_reverse(attn_windows, self.window_size, int(T1 * H), W)  # B H' W' C
            x1 = shifted_x1
        # if we excute the cross attention after the shift window attention,
        # x1 and x2 to be the input of the cross view attention after the shift reverse
        x1 = x1.view(B1, int(T1 * H * W), C1)
        out = x1
        x1 = shortcut + self.drop_path(x1)

        if not self.last_view:
            # window based cross attention
            x1_windows = window_partition(x1.view(B1, int(T1 * H), W, C1), self.window_size).view(-1, self.window_size *
                                                                                                  self.window_size, C1)
            x2_windows = window_partition(x2, self.window_size).view(-1, self.window_size * self.window_size, C2)
            x2_windows = self.pre(x2_windows)
            y, attn = self.cva(x1_windows, x2_windows)
            y = rearrange(y, '(b n) ws c -> b (n ws) c', ws=self.window_size * self.window_size, b=B1)
            x1 = x1 + self.drop_path(y)

        # FFN
        x1 = x1 + self.drop_path(self.mlp(self.norm2(x1)))

        return x1, out


class CrossThreeViewSwinBlock(nn.Module):
    # Input Resolution : [view, view2, view3]
    def __init__(self,
                 view_configs,
                 input_resolution,
                 cur_stage, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()
        self.block1 = CrossSwinBlock(view_configs[0]["hidden_size"][cur_stage],
                                     view_configs[1]["hidden_size"][cur_stage],
                                     input_resolution[0],
                                     view_configs[0]["num_heads"][cur_stage],
                                     window_size=view_configs[0]["window_size"],
                                     shift_size=0, mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                                     act_layer=nn.GELU, norm_layer=norm_layer,
                                     fused_window_process=fused_window_process,
                                     temporal_dims=view_configs[0]["temporal_ratio"],
                                     cur_stage=cur_stage)

        self.block2 = CrossSwinBlock(view_configs[1]["hidden_size"][cur_stage],
                                     view_configs[2]["hidden_size"][cur_stage],
                                     input_resolution[1],
                                     view_configs[1]["num_heads"][cur_stage],
                                     window_size=view_configs[1]["window_size"],
                                     shift_size=0, mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                                     act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                                     fused_window_process=fused_window_process,
                                     temporal_dims=view_configs[0]["temporal_ratio"],
                                     cur_stage=cur_stage)

        self.block3 = CrossSwinBlock(view_configs[2]["hidden_size"][cur_stage],
                                     view_configs[2]["hidden_size"][cur_stage],
                                     input_resolution[2],
                                     view_configs[2]["num_heads"][cur_stage],
                                     window_size=view_configs[2]["window_size"],
                                     shift_size=0, mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                                     act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                                     fused_window_process=fused_window_process,
                                     last_view=True,
                                     temporal_dims=3,
                                     cur_stage=cur_stage)

    def forward(self, x):
        # x ==> List[view1, view2, view3]
        x[2], out2 = self.block3(x[2], x[2])
        x[1], out1 = self.block2(x[1], out2)
        x[0], _ = self.block1(x[0], out1)
        return x


class CrossMultipleViewSwinBlock(nn.Module):
    # Input Resolution : [view, view2, ..., viewn]
    def __init__(self,
                 view_configs,
                 input_resolution,
                 cur_stage, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            CrossSwinBlock(view_configs[i]["hidden_size"][cur_stage],
                           view_configs[i + 1]["hidden_size"][cur_stage],
                           input_resolution[i],
                           view_configs[i]["num_heads"][cur_stage],
                           window_size=view_configs[i]["window_size"],
                           shift_size=0, mlp_ratio=mlp_ratio,
                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                           drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                           act_layer=nn.GELU, norm_layer=norm_layer,
                           fused_window_process=fused_window_process,
                           temporal_dims=view_configs[i]["temporal_ratio"],
                           last_view=True if i == view_configs["num_views"] - 1 else False,
                           cur_stage=cur_stage)
            for i in range(view_configs["num_views"])
        ])

    def forward(self, x):
        # x ==> List[view1, view2, ..., viewn]
        len = len(self.blocks)
        out = x[len - 1]
        for idx, block in enumerate(reversed(self.blocks)):
            x[len - 1 - idx], out = block(x[len - 1 - idx], out)
        return x


class OriginalThreeViewSwinBlock(nn.Module):
    # cur_stage means i-th Stage, and cur_lyr means i-th layer in current stage
    def __init__(self,
                 view_configs,
                 input_resolution,
                 cur_stage, cur_lyr,
                 mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()
        self.block1 = SwinTransformerBlock(dim=view_configs[0]["hidden_size"][cur_stage],
                                           input_resolution=input_resolution[0],
                                           num_heads=view_configs[0]["num_heads"][cur_stage],
                                           window_size=view_configs[0]["window_size"],
                                           shift_size=0 if (cur_lyr % 2 == 0) else view_configs[0]["window_size"] // 2,
                                           mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           drop=drop, attn_drop=attn_drop,
                                           drop_path=drop_path,
                                           norm_layer=norm_layer,
                                           fused_window_process=fused_window_process,
                                           temporal_dim=view_configs[0]['temporal_dim']
                                           ) if cur_lyr < view_configs[0]["depths"][cur_stage] else nn.Identity()

        self.block2 = SwinTransformerBlock(dim=view_configs[1]["hidden_size"][cur_stage],
                                           input_resolution=input_resolution[1],
                                           num_heads=view_configs[1]["num_heads"][cur_stage],
                                           window_size=view_configs[1]["window_size"],
                                           shift_size=0 if (cur_lyr % 2 == 0) else view_configs[0]["window_size"] // 2,
                                           mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           drop=drop, attn_drop=attn_drop,
                                           drop_path=drop_path,
                                           norm_layer=norm_layer,
                                           fused_window_process=fused_window_process,
                                           temporal_dim=view_configs[1]["temporal_dim"]
                                           ) if cur_lyr < view_configs[1]["depths"][cur_stage] else nn.Identity()

        self.block3 = SwinTransformerBlock(dim=view_configs[2]["hidden_size"][cur_stage],
                                           input_resolution=input_resolution[2],
                                           num_heads=view_configs[2]["num_heads"][cur_stage],
                                           window_size=view_configs[2]["window_size"],
                                           shift_size=0 if (cur_lyr % 2 == 0) else view_configs[0]["window_size"] // 2,
                                           mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           drop=drop, attn_drop=attn_drop,
                                           drop_path=drop_path,
                                           norm_layer=norm_layer,
                                           fused_window_process=fused_window_process,
                                           temporal_dim=view_configs[2]["temporal_dim"]
                                           ) if cur_lyr < view_configs[2]["depths"][cur_stage] else nn.Identity()

    def forward(self, x):
        # x instances of list which contains view1 view2 view3
        x[0] = self.block1(x[0])
        x[1] = self.block2(x[1])
        x[2] = self.block3(x[2])
        return x


class OriginalTMultipleViewSwinBlock(nn.Module):
    # cur_stage means i-th Stage, and cur_lyr means i-th layer in current stage
    def __init__(self,
                 view_configs,
                 input_resolution,
                 cur_stage, cur_lyr,
                 mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=view_configs[i]["hidden_size"][cur_stage],
                                 input_resolution=input_resolution[i],
                                 num_heads=view_configs[i]["num_heads"][cur_stage],
                                 window_size=view_configs[i]["window_size"],
                                 shift_size=0 if (cur_lyr % 2 == 0) else view_configs[i]["window_size"] // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process,
                                 temporal_dim=view_configs[i]['temporal_dim']
                                 ) if cur_lyr < view_configs[i]["depths"][cur_stage] else nn.Identity()
            for i in range(view_configs["num_views"])])

    def forward(self, x):
        # x instances of list which contains view1, view2, ..., view3
        for idx, block in enumerate(self.blocks):
            x[i] = self.block(x[i])
        return x


class MultiViewBasicLayer(nn.Module):
    '''
        This defines a single stage for the Cross-Swin Transformer.
        A Cross-Swin Transformer consists of four stages in total.
        The "depth" refers to the number of blocks that are contained within the current stage.
    '''
    def __init__(self,
                 view_configs,
                 cur_stage, depth,
                 mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 fused_window_process=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            CrossThreeViewSwinBlock(view_configs,
                                    input_resolution=[view_configs[k]["input_resolution"][cur_stage] for k in range(3)],
                                    cur_stage=cur_stage, mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                    norm_layer=norm_layer,
                                    fused_window_process=fused_window_process) if i == 0 else
            OriginalThreeViewSwinBlock(view_configs,
                                       input_resolution=[view_configs[k]["input_resolution"][cur_stage] for k in
                                                         range(3)],
                                       cur_stage=cur_stage, cur_lyr=i,
                                       mlp_ratio=mlp_ratio,
                                       qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       drop=drop, attn_drop=attn_drop,
                                       drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                       norm_layer=norm_layer,
                                       fused_window_process=fused_window_process) for i in range(depth)
        ])

        if downsample is not None:
            self.downsample = downsample(view_configs, cur_stage)
        else:
            self.downsample = None

    def forward(self, x):
        out = []
        for blk in self.blocks:
            x = blk(x)
            out = x.copy()
        if self.downsample is not None:
            x = self.downsample(x)
        return x, out


class CreateStages(nn.Module):
    def __init__(self, view_configs, depths=[2, 2, 18, 2],
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 stages=4,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 ape=False, patch_norm=True,
                 use_checkpoint=False,
                 fused_window_process=False):
        super().__init__()
        self.layers = nn.ModuleList()
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        for i in range(stages):
            layer = MultiViewBasicLayer(view_configs=view_configs,
                                        cur_stage=i, depth=depths[i],
                                        mlp_ratio=mlp_ratio,
                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        drop=drop_rate, attn_drop=attn_drop_rate,
                                        drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                        norm_layer=norm_layer,
                                        downsample=ThreeViewPatchMerging if (i < stages - 1) else None,
                                        fused_window_process=fused_window_process)
            self.layers.append(layer)

    def forward(self, x):
        out = []
        for idx, lyr in enumerate(self.layers):
            x, out_stage = lyr(x)
            out.append(out_stage)
        return x, out


class CrossThreeViewTokenize(nn.Module):
    def __init__(self, view_configs):
        super().__init__()
        self.project1 = nn.Conv3d(in_channels=3,
                                  out_channels=view_configs[0]["hidden_size"][0],
                                  kernel_size=(view_configs[0]["patches"].size[-1], view_configs[0]["patches"].size[0],
                                               view_configs[0]["patches"].size[1]),
                                  stride=(view_configs[0]["patches"].size[-1], view_configs[0]["patches"].size[0],
                                          view_configs[0]["patches"].size[1]),
                                  padding=0)

        self.project2 = nn.Conv3d(in_channels=3,
                                  out_channels=view_configs[1]["hidden_size"][0],
                                  kernel_size=(view_configs[1]["patches"].size[-1], view_configs[1]["patches"].size[0],
                                               view_configs[1]["patches"].size[1]),
                                  stride=(view_configs[1]["patches"].size[-1], view_configs[1]["patches"].size[0],
                                          view_configs[1]["patches"].size[1]),
                                  padding=0)

        self.project3 = nn.Conv3d(in_channels=3,
                                  out_channels=view_configs[2]["hidden_size"][0],
                                  kernel_size=(view_configs[2]["patches"].size[-1], view_configs[2]["patches"].size[0],
                                               view_configs[2]["patches"].size[1]),
                                  stride=(view_configs[2]["patches"].size[-1], view_configs[2]["patches"].size[0],
                                          view_configs[2]["patches"].size[1]),
                                  padding=0)

        self.norm1 = nn.LayerNorm(view_configs[0]["hidden_size"][0])
        self.norm2 = nn.LayerNorm(view_configs[1]["hidden_size"][0])
        self.norm3 = nn.LayerNorm(view_configs[2]["hidden_size"][0])

    def forward(self, x):
        input = [x, x, x]
        project_layers = [self.project1, self.project2, self.project3]
        norm_layers = [self.norm1, self.norm2, self.norm3]

        output = []
        for i in range(3):
            input[i] = rearrange(input[i], 'b t c h w -> b c t h w')
            input[i] = project_layers[i](input[i])
            input[i] = rearrange(input[i], 'b c t h w -> b t (h w) c')
            input[i] = norm_layers[i](input[i])
            output.append(input[i])

        return output


class CrossMultipleViewTokenize(nn.Module):
    def __init__(self, view_configs):
        super().__init__()
        self.project_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for view_config in view_configs:
            hidden_size = view_config["hidden_size"][0]
            patch_size = view_config["patches"].size

            self.project_layers.append(
                nn.Conv3d(
                    in_channels=3,
                    out_channels=hidden_size,
                    kernel_size=(patch_size[-1], patch_size[0], patch_size[1]),
                    stride=(patch_size[-1], patch_size[0], patch_size[1]),
                    padding=0
                )
            )

            self.norm_layers.append(nn.LayerNorm(hidden_size))

    def forward(self, x):
        input = [x] * len(self.project_layers)
        output = []

        for i in range(len(self.project_layers)):
            input[i] = rearrange(input[i], 'b t c h w -> b c t h w')
            input[i] = self.project_layers[i](input[i])
            input[i] = rearrange(input[i], 'b c t h w -> b t (h w) c')
            input[i] = self.norm_layers[i](input[i])
            output.append(input[i])

        return output


class CreateGlobalBlocks(nn.Module):
    def __init__(self, global_encoder_config, dpr, dropout_rate):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(global_encoder_config["hidden_size"], global_encoder_config["num_heads"],
                   global_encoder_config["mlp_dim"], dropout_rate, dpr[i]) for i in
             range(global_encoder_config["num_layers"])]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ThreeViewSwinTransformer(nn.Module):
    def __init__(self, view_configs,
                 input_token_temporal_dims,
                 global_encoder_config,
                 depths=[2, 2, 18, 2],
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, stages=4,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False, patch_norm=True,
                 use_checkpoint=False,
                 fused_window_process=False):
        super().__init__()
        self.faf = FAF()
        self.tokenize = CrossThreeViewTokenize(view_configs)
        self.input_token_temporal_dims = input_token_temporal_dims
        self.layers = CreateStages(view_configs, depths=depths, mlp_ratio=mlp_ratio,
                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   stages=stages,
                                   drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                   drop_path_rate=drop_path_rate,
                                   norm_layer=norm_layer,
                                   ape=ape, patch_norm=patch_norm,
                                   use_checkpoint=use_checkpoint,
                                   fused_window_process=fused_window_process)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.globalembedding = nn.Linear(2560, 768)
        self.global_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, global_encoder_config["num_layers"])]
        self.globalblocks = CreateGlobalBlocks(global_encoder_config, self.global_dpr, drop_rate)

    def align_temporal_dimension_across_views(self, tokens):
        """Reshapes tokens from each view，so they have the same temporal dim."""
        min_temporal_dim = min(self.input_token_temporal_dims)
        outputs = []
        for t in tokens:
            bs, time, n, c = t.shape
            outputs.append(t.reshape(bs, min_temporal_dim, (n * time) // min_temporal_dim, c))
        return outputs

    def merge_views_along_channel_axis(self, tokens):
        """Merges tokens from each view along the channel axis."""
        max_temporal_dim = max(self.input_token_temporal_dims)
        xs = []
        for idx, x in enumerate(tokens):
            bs, time, n, c = x.shape
            x = x.reshape(bs, self.input_token_temporal_dims[idx], (time * n) // self.input_token_temporal_dims[idx], c)
            xs.append(x.repeat((1, max_temporal_dim // x.shape[1], 1, 1)))
        return torch.concatenate(xs, axis=-1)

    def merge_views_along_time_axis(self, tokens):
        """Merges tokens from each view along the time axis."""
        xs = []
        for idx, x in enumerate(tokens):
            bs, time, n, c = x.shape
            x = x.reshape(bs, self.input_token_temporal_dims[idx], (time * n) // self.input_token_temporal_dims[idx], c)
            if c == 768:
                xs.append(x)
            else:
                xs.append(self.mergeembedding(x))
        return torch.concatenate(xs, 1)

    def forward(self, x):
        #x ==> List[] B * T * S * D
        ffinfo = self.faf(x)[:, 1, :, :, :]
        x = self.tokenize(x)
        x = self.align_temporal_dimension_across_views(x)
        x, out_x = functorch.vmap(self.layers, in_dims=1, out_dims=1, randomness="same")(x)

        x = self.merge_views_along_channel_axis(x)  # 1 * 3 * 49 * 2560
        x = self.globalembedding(x)  # b * t * n * c
        x = functorch.vmap(self.globalblocks, in_dims=2, out_dims=2, randomness="same")(x)

        # cat
        #x = torch.cat((x[:,0,:,:], x[:,1,:,:], x[:,2,:,:], x[:,3,:,:], x[:,4,:,:]), dim=-1)
        x = torch.cat((x[:, 0, :, :], x[:, 1, :, :], x[:, 2, :, :]), dim=-1)
        return x, out_x, ffinfo

