import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, trunc_normal_

from mmcv.runner import load_checkpoint
from mmcv.utils import get_logger
import logging

from functools import reduce, lru_cache
from operator import mul
from einops import rearrange


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        B, D, H, W, C:  batch_size, depth, hight, width, channels
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x

# 不一样
def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention3D(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads     # 每个 attention head 对应的通道数
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        B_, N, C = x.shape    # N = window_size_d * window_size_h * window_size_w
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0) # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(2,7,7), shift_size=(0,0,0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint=use_checkpoint

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape

        # 1. 先计算出当前block的window_size, 和shift_size
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        # 2. 经过一个layer_norm
        x = self.norm1(x)

        # pad feature maps to multiples of window size, 避免 feature maps 除不开
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape

        # 3. 判断是否需要对特征图进行shift
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # 4. 将特征图切成一个个的窗口（reshape) -- partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C

        # 5. W-MSA/SW-MSA (通过attn_mask是否为None判断进行W-MSA还是SW-MSA)
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C

        # 6. 把窗口在合并回来，NO.4 的逆操作(reshape) -- merge windows
        attn_windows = attn_windows.view(-1, *(window_size+(C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C

        # reverse cyclic shift, 还原之前的 shift 过
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        # remove pad
        if pad_d1 >0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x):
        # pass FFN
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """
        """
        # !calculate attention mask for SW-MSA
        _, _, D, H, W = x.shape
        window_size, shift_size = get_window_size((D,H,W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        # ends"""
        # print('------2----', x.shape)
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer, 下采样层

    Input x: Input feature, tensor size (B, D, H, W, C).
    Output x: Input feature, tensor size (B, D, H/2, W/2, 2*C).
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        # 用 FC layer reduce channels from 4C->2C
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, D, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, :, 0::2, 0::2, :]    # (B, D, H/2, W/2, C)
        x1 = x[:, :, 1::2, 0::2, :]    # (B, D, H/2, W/2, C)
        x2 = x[:, :, 0::2, 1::2, :]    # (B, D, H/2, W/2, C)
        x3 = x[:, :, 1::2, 1::2, :]    # (B, D, H/2, W/2, C)
        x = torch.cat([x0, x1, x2, x3], -1)    # (B, D, H/2, W/2, 4C)

        x = self.norm(x)               # norm, shape unchanged
        x = self.reduction(x)          # (B, D, H/2, W/2, 2C), channel halves

        return x


# cache each stage results
@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0],None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1],None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2],None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class PatchExpanding(nn.Module):
    """ Patch Expanding Layer, 上采样层

    Input size = (B, C, D, H, W)
    Output size = (B, C/2, D, 2H, 2W)
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm): # input_resolution
        super().__init__()
        # self.input_resolution = input_resolution
        self.dim = dim
        # expand channel, C->2C while keep H & W same
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
            
        """ 
        #print('.........', x.shape, self.dim)    # (B, C, D, H, W)
        B, C, D, H, W = x.shape
        x = rearrange(x, 'b c d h w -> b d h w c')
        x = self.expand(x)                   # (B, D, H, W, 2C)
        _, _, _, _, C1 = x.shape             # C1 = 2*C

        x = rearrange(x, 'b d h w (p1 p2 c)-> b d (h p1) (w p2) c', p1=2, p2=2, c=C1//4)    # (B, D, 2H, 2W, C/2)
        x = self.norm(x)          # (B, D, 2H, 2W, C/2)
        x = rearrange(x, 'b d h w c -> b c d h w')
        #print('..', x.shape, self.dim)     # (B, C/2, D, 2H, 2W)

        return x
        

class FinalPatchExpand_X4(nn.Module):
    """ Patch Expanding Layer, 上采样层

    Input size = (B, D, H, W, C)
    Output size = (B, D, 4H, 4W, C)
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, patch_scale=(2,4,4), norm_layer=nn.LayerNorm): # input_resolution
        super().__init__()
        # self.input_resolution = input_resolution
        self.dim = dim
        self.patch_scale = patch_scale
        self.expand = nn.Linear(dim, 32*dim, bias=False)  # output dim = 2*4*4
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: (B, D, H, W, C)
        """
        B, D, H, W, C = x.shape
        x = self.expand(x)         # (B, D, H, W, 32C)
        _,_, _, _, C1 = x.shape
        # assert L == H * W, "input feature has wrong size"

        # x = x.view(B, H, W, C)
        # x = x.view(B,-1,self.output_dim)
        x = rearrange(x, 'b d h w (p0 p1 p2 c)-> b (d p0) (h p1) (w p2) c', p0=self.patch_scale[0], p1=self.patch_scale[1], p2=self.patch_scale[2], c=C1//32)  
        # (B, 2D, 4H, 4W, C)
        x= self.norm(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage. 
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,          # !input_resolution
                 dim,           # dim = C
                 depth,         # 跟feature的D不同，指的block深度,也就是这个module会有多少个 W-MSA/SW-MSA 摞在一起 (两种SA交替出现)
                                # vedio swin 对照示意图应该 depth 是 (2，2，6，2) for 4 Basic module
                 num_heads, window_size=(1,7,7), mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 downsample=None,     # PatchMerging
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(dim=dim,
                num_heads=num_heads, window_size=window_size,
                shift_size=(0,0,0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint,
            )                            # (B,D,H,W,C) -> (B,D,H,W,C)
            for i in range(depth)])
        
        # patch merging layer
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D,H,W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)

        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(B, D, H, W, -1)

        if self.downsample is not None:
            x = self.downsample(x)      # (B, D, H, W, C) -> (B, D, H/2, W/2, 2*C)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x

# !BasicLayer_up
class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        upsample (nn.Module | None, optional): Upsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(dim=dim,
                    num_heads=num_heads, window_size=window_size,
                    shift_size=(0,0,0) if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop, attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer, use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])
        
        # patch expanding layer
        self.upsample = upsample
        if upsample is not None:
            self.upsample = upsample(dim=dim, dim_scale=2, norm_layer=norm_layer)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        
        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D,H,W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)

        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(B, D, H, W, -1)
        x = rearrange(x, 'b d h w c -> b c d h w')
        
        if self.upsample is not None:
            x = self.upsample(x)      # (B, D, H, W, C) -> (B, D, H, W, C)
        
        return x


class PatchEmbed3D(nn.Module):
    """ Video to Patch Embedding. Linear Embedding at the start of network.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 4.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=(2,4,4), in_chans=4, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function.
        Input size = (B, C, D, H, W)
        Output size = (B, C = embed_dim, Dh, Hh, Wh)
        (Dh, Hh, Wh) = (D, H, W) / patch_size
        """
        # D, H, W: 3 dimensions of the 3D input feature (depth, height, width)
        # D, H, W = 155, 224, 224 in BraTS dataset
        # print('----000----', x.shape, 'patch=', self.patch_size)
        _, _, D, H, W = x.size()

        # D, H, W = patch_size[0], patch_size[1], patch_size[2], 防止除不开先 padding
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)                   # B C D Wh Ww, (D, Wh, Ww = feature size after 3D convolution)
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)      # flatten used for the upcoming normalisation
            # (B, C, D, Wh, Ww) -> (B, C, D*Wh*Ww) -> (B, D*Wh*Ww, C)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)  # back to (B, C, D, Wh, Ww)
        # print('----020----', x.shape)
        return x

# @BACKBONES.register_module()
class SwinTransformer3D(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: (4,4,4).
        in_chans (int): Number of input image channels. Default: 4.
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(self,
                 pretrained=None, pretrained2d=None, patch_size=(2,4,4), 
                 in_chans=4, num_classes=1000, embed_dim=96, depths=[2, 2, 6, 2], 
                 num_heads=[3, 6, 12, 24], window_size=(2,7,7), mlp_ratio=4., 
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.2,norm_layer=nn.LayerNorm, patch_norm=False, 
                 frozen_stages=-1, use_checkpoint=False, final_upsample="expand_first"):
        super().__init__()

        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        # self.ape = ape
        # self.num_features_up = int(embed_dim * 2)

        self.num_features = int(embed_dim * 2 ** (self.num_layers-1))

        # split image into non-overlapping patches (Linear embeding)
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        # (B, C, D, H, W) -> (B, C=embed_dim , Dh, Hh, Wh)
        
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(                           # basic module, (B,C,D,H,W) -> (B,C,D,H,W) or (B, D, H/2, W/2, 2*C)
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,   # 前三个stage后要用 PatchMerging 下采样
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        # 新增: build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()

        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2*int(embed_dim*2**(self.num_layers-1-i_layer)),
            int(embed_dim*2**(self.num_layers-1-i_layer))) if i_layer > 0 else nn.Identity()

            if i_layer == 0:
                # (B,C,D,H,W) -> (B,C/2,D,2H,2W)
                layer_up = PatchExpanding(dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)), dim_scale=2, norm_layer=norm_layer)

            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)),
                                depth=depths[(self.num_layers-1-i_layer)],
                                num_heads=num_heads[(self.num_layers-1-i_layer)],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:(self.num_layers-1-i_layer)]):sum(depths[:(self.num_layers-1-i_layer) + 1])],
                                norm_layer=norm_layer,
                                upsample=PatchExpanding if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint)   # (B,C,D,H,W) -> (B,C,D,H,W)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        # add a norm layer for each output
        self.norm = norm_layer(self.num_features)
        self.norm_up= norm_layer(self.embed_dim)
        
        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(patch_scale=self.patch_size, dim=embed_dim)   # (B, D, H, W, C) -> (B, D, 4H, 4W, C)
            self.output = nn.Conv3d(in_channels=embed_dim,out_channels=self.num_classes,kernel_size=1,bias=False)
            # 不知道是 2D 还是 3D
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    import logging


    def get_root_logger(log_file=None, log_level=logging.INFO):
        """Use ``get_logger`` method in mmcv to get the root logger.

        The logger will be initialized if it has not been initialized. By default a
        StreamHandler will be added. If ``log_file`` is specified, a FileHandler
        will also be added. The name of the root logger is the top-level package
        name, e.g., "mmaction".

        Args:
            log_file (str | None): The log filename. If specified, a FileHandler
                will be added to the root logger.
            log_level (int): The root logger level. Note that only the process of
                rank 0 is affected, while other processes will set the level to
                "Error" and be silent most of the time.

        Returns:
            :obj:`logging.Logger`: The root logger.
        """
        return get_logger(__name__.split('.')[0], log_file, log_level)

    
    def inflate_weights(self, logger):
        """
        Inflate the swin2d parameters to swin3d.

        The differences between swin3d and swin2d mainly lie in an extra
        axis. To utilize the pretrained parameters in 2d model,
        the weight of swin2d models should be inflated to fit in the shapes of
        the 3d counterpart.

        Args:
            logger (logging.Logger): The logger used to print
                debugging infomation.
        """
        
        checkpoint = torch.load(self.pretrained, map_location='cpu')
        state_dict = checkpoint['model']

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'].unsqueeze(2).repeat(1,1,self.patch_size[0],1,1) / self.patch_size[0]

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = self.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            L2 = (2*self.window_size[1]-1) * (2*self.window_size[2]-1)
            wd = self.window_size[0]
            if nH1 != nH2:
                logger.warning(f"Error in loading {k}, passing")
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(2*self.window_size[1]-1, 2*self.window_size[2]-1),
                        mode='bicubic')
                    relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)
            state_dict[k] = relative_position_bias_table_pretrained.repeat(2*wd-1,1)

        msg = self.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        logger.info(f"=> loaded successfully '{self.pretrained}'")
        del checkpoint
        torch.cuda.empty_cache()

    
    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            if self.pretrained2d:
                # Inflate 2D model into 3D model.
                self.inflate_weights(logger)
            else:
                # Directly load 3D model.
                load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')


    # !! 新增: Encoder and Bottleneck
    def forward_features(self, x):
        """
        x: Input feature, tensor size (B, C, D, H, W)
        """
        x = self.patch_embed(x)        # (B, C, D, H, W) -> (B, C, D, Wh, Ww)
        #if self.ape:
        #    x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        x_downsample = []
        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)              # (B, C, D, H, W) -> (B, C, D, H, W)
        """
        C, D, H, W = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2)      # flatten used for the upcoming normalisation
        # (B, C, D, Wh, Ww) -> (B, C, D*Wh*Ww) -> (B, D*Wh*Ww, C)
        x = self.norm(x)  # B L C (L = D*Wh*Ww)
        x = x.transpose(1, 2).view(-1, C, D, H, W)  # back to (B, C, D, Wh, Ww)
        # (B, D*Wh*Ww, C) -> (B, C, D*Wh*Ww) -> (B, C, D, Wh, Ww)
        """
        
        D, H, W = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2)   #  (B, C, D, Wh, Ww) -> (B, L, C)
        x = self.norm(x)                   # B L C              (L = D*Wh*Ww)
        x = x.transpose(1, 2).view(-1, self.num_features, D, H, W)  # (B, L, C) ->  (B, C, D, Wh, Ww)
        return x, x_downsample
    
    #Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        """
        x: Input feature, tensor size (B, C, D, H, W)
        """
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)    # (B,C,D,H,W) -> (B,C,D,H,W)
            else:
                x = torch.cat([x,x_downsample[3-inx]],1)   # cat at dimension C

                x = rearrange(x, 'n c d h w -> n d h w c')    # (B,C,D,H,W) -> (B,D,H,W,C)
                x = self.concat_back_dim[inx](x) # (B,D,H,W,C) -> (B,D,H,W,C)
                x = rearrange(x, 'n d h w c -> n c d h w')    # (B,D,H,W,C) -> (B,C,D,H,W)
                x = layer_up(x)       # (B,C,D,H,W) -> (B,C,D,H,W)

        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm_up(x)  # B L C (L = D*Wh*Ww)
        x = rearrange(x, 'n d h w c -> n c d h w')
  
        return x

    def up_x4(self, x):
        """
        x: Input feature, tensor size (B, C, D, H, W)
        """

        B, C, D, H, W = x.shape
        x = rearrange(x, 'n c d h w -> n d h w c')
        if self.final_upsample=="expand_first":
            x = self.up(x)              # (B, D, H, W, C) -> (B, D, 4H, 4W, C)
            #x = x.view(B,D,4*H,4*W,-1)
            x = x.permute(0,4,1,2,3)    # (B, D, H, W, C) -> (B, C, D, H, W)
            x = self.output(x)

        return x
    
    """
    def forward(self, x):
        #Forward function.
        #x: Input feature, tensor size (B, C, D, H, W)

        x = self.patch_embed(x)       # (B, C, D, H, W) -> (B, C, D, Wh, Ww)

        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x.contiguous())

        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')

        return x
    """

    def forward(self, x):
        """Forward function.
        x: Input feature, tensor size (B, C, D, H, W)
        """
        x, x_downsample = self.forward_features(x)      # (B, C, D, H, W) -> (B, C, D, Wh, Ww)
        x = self.forward_up_features(x,x_downsample)    # (B, C, D, H, W) -> (B, C, D, Wh, Ww)
        x = self.up_x4(x)                               # (B, C, D, H, W) -> (B, C, D, Wh, Ww)
        return x

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer3D, self).train(mode)
        self._freeze_stages()
