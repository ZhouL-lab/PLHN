import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_3tuple, trunc_normal_
import torch.nn.functional as F


def window_partition(x, window_size):
    B, S, H, W, C = x.shape
    x = x.view(B, S // window_size, window_size, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, S, H, W):
    B = int(windows.shape[0] / (S * H * W / window_size / window_size / window_size))
    x = windows.view(B, S // window_size, H // window_size, W // window_size, window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, S, H, W, -1)
    return x

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
class LearnedPositionalEncoding(nn.Module):
    def __init__(self,position_size, embedding_dim):
        super(LearnedPositionalEncoding, self).__init__()

        self.position_embeddings = nn.Parameter(torch.zeros(1, position_size, embedding_dim)) #8x

    def forward(self, x):

        position_embeddings = self.position_embeddings
        return x + position_embeddings

class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_embeddings = nn.Parameter(torch.zeros(1, window_size[1]*window_size[1]*window_size[2], dim))


        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.conv = nn.Conv1d(dim, dim * 3,kernel_size=3,stride=1,padding=1,dilation=1)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        #trunc_normal_(self.relative_position_bias_table, std=.02)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        B_, N, C = x.shape
        x = x+self.position_embeddings

        qkv = self.qkv(x)
        # qkv = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)

        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1).contiguous())

        #attn = attn #+  self.position_embeddings


        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        #
        # if min(self.input_resolution) <= self.window_size:
        #     # if window size is larger than input resolution, we don't partition windows
        #     self.shift_size = 0
        #     self.window_size = min(self.input_resolution)
        #
        # assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)

        self.attn = WindowAttention(
            dim, window_size=to_3tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):

        B, L, C = x.shape
        S, H, W = self.input_resolution

        assert L == S * H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, S, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_g = (self.window_size - S % self.window_size) % self.window_size

        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g))
        _, Sp, Hp, Wp, _ = x.shape

        # cyclic shift
        # if self.shift_size > 0:
        #     shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size, -self.shift_size), dims=(1, 2, 3))
        #     attn_mask = mask_matrix
        # else:
        shifted_x = x
            #attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size * self.window_size,
                                   C)
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Sp, Hp, Wp)

        # reverse cyclic shift
        # if self.shift_size > 0:
        #     x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size, self.shift_size), dims=(1, 2, 3))
        # else:
        x = shifted_x

        if pad_r > 0 or pad_b > 0 or pad_g > 0:
            x = x[:, :S, :H, :W, :].contiguous()

        x = x.view(B, S * H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        # nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        # nn.InstanceNorm3d(out_dim),
        # activation,
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
        nn.InstanceNorm3d(out_dim),
        activation,
    )
def conv_3d_NoDown(in_dim, out_dim, activation):
    return nn.Sequential(
        # nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        # nn.InstanceNorm3d(out_dim),
        # activation,
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm3d(out_dim),
        activation,
    )


def conv_block_2_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm3d(out_dim),
        activation,
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=2, padding=1),
        nn.InstanceNorm3d(out_dim),
        activation,
    )


class TokenSeg(nn.Module):
    def __init__(self, inch=1, outch=1, downlayer=3, base_channeel=32, imgsize=[128, 128, 128], hidden_size=256,
                 window_size=8, TransformerLayerNum=8):
        super().__init__()
        self.imgsize = imgsize
        self.bottlensize = [i // (2 ** downlayer) for i in imgsize]
        activation = nn.LeakyReLU(0.2, inplace=True)
        self.proxylayers = nn.ModuleList()
        self.proxylayers.append(conv_block_3d(inch, 8 * inch, activation))
        for i in range(1, downlayer):
            self.proxylayers.append(conv_block_3d(inch * 8 ** i, inch * 8 ** (i + 1), activation))
        self.proxy = nn.Sequential(
            *self.proxylayers
        )

        self.stemlayers = nn.ModuleList()
        self.stemlayers.append(conv_block_2_3d(inch, base_channeel, activation))
        for i in range(1, downlayer):
            self.stemlayers.append(
                conv_block_2_3d(base_channeel * (2 ** (i - 1)), base_channeel * (2 ** i), activation))
        self.stem = nn.Sequential(
            *self.stemlayers
        )

        self.line = nn.Linear(base_channeel * (2 ** (downlayer - 1)), hidden_size)
        self.project = nn.Linear(hidden_size, 8 ** downlayer * inch)
        self.block = nn.ModuleList()  # [SwinTransformerBlock(192,[32,32,32],8)]*10
        for i in range(TransformerLayerNum):
            self.block.append(SwinTransformerBlock(hidden_size, self.bottlensize, window_size))
        self.window_size = window_size
        self.shift_size = self.window_size // 2

        self.out = nn.Conv3d(inch, outch, 3, 1, 1)

    def forward(self, x):
        B, C, D, H, W = x.size()
        x1 = self.proxy(x)

        x2 = self.stem(x).view(B, -1, self.bottlensize[0] * self.bottlensize[1] * self.bottlensize[2]).permute(0, 2,
                                                                                                               1).contiguous()
        x2 = self.line(x2)
        for i in self.block:
            x2 = i(x2)
        x2 = self.project(x2)
        x2 = x2.view(B, self.bottlensize[0], self.bottlensize[1], self.bottlensize[2], -1).permute(0, 4, 1, 2,
                                                                                                   3).contiguous()
        print(x2.size())
        x = x1 * x2
        x = x.view(B, C, D, H, W)
        x = self.out(x)

        return x

def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        activation,)

class TokenSegV2(nn.Module):
    def __init__(self, inch=2, outch=1, downlayer=3, base_channeel=32, imgsize=[48, 128, 128], hidden_size=256,
                 window_size=8, TransformerLayerNum=8):
        super().__init__()
        self.imgsize = imgsize
        self.bottlensize = [i // (2 ** downlayer) for i in imgsize]
        activation = nn.LeakyReLU(0.2, inplace=True)
        # self.proxylayers = nn.ModuleList()
        # self.proxylayers.append(conv_block_3d(inch, 8 * inch, activation))
        # for i in range(1, downlayer):
        #     self.proxylayers.append(conv_block_3d(inch * 8 ** i, inch * 8 ** (i + 1), activation))
        # self.proxy = nn.Sequential(
        #     *self.proxylayers
        # )
        self.proxylayers = nn.ModuleList()
        self.proxylayers.append(conv_block_2_3d(inch, base_channeel, activation))
        self.proxylayers.append(conv_block_2_3d(base_channeel, base_channeel*2, activation))
        self.proxylayers.append(conv_block_2_3d(base_channeel*2, hidden_size, activation))
        #self.proxylayers.append(conv_block_2_3d(base_channeel * 4, base_channeel * 8, activation))
        #self.proxylayers.append(conv_block_2_3d(base_channeel * 8, base_channeel * 8, activation))

        self.stemlayers = nn.ModuleList()
        self.stemlayers.append(conv_block_2_3d(inch, base_channeel, activation))

        for i in range(1, downlayer):
            self.stemlayers.append(
                conv_block_2_3d(base_channeel * (2 ** (i - 1)), base_channeel * (2 ** i), activation))
        self.stem = nn.Sequential(
            *self.stemlayers
        )

        self.line = nn.Linear(base_channeel * (2 ** (downlayer - 1)), hidden_size)
       # self.project = nn.Linear(hidden_size, 8 ** downlayer * inch)
        self.block = nn.ModuleList()  # [SwinTransformerBlock(192,[32,32,32],8)]*10
        for i in range(TransformerLayerNum): ##
            self.block.append(SwinTransformerBlock(hidden_size, self.bottlensize, window_size))
        self.window_size = window_size
        self.shift_size = self.window_size // 2

        self.out = nn.Conv3d(base_channeel, outch, 3, 1, 1)
        self.trans_1 = conv_trans_block_3d(hidden_size, base_channeel*2, activation)
        self.trans_2 = conv_trans_block_3d(base_channeel*2, base_channeel , activation)
        self.trans_3 = conv_trans_block_3d(base_channeel, base_channeel, activation)

    def forward(self, x):
        B, C, D, H, W = x.size()
        #x1 = self.proxy(x)
        skip1 = self.proxylayers[0](x)
        skip2 = self.proxylayers[1](skip1)
        skip3 = self.proxylayers[2](skip2)

        x2 = self.stem(x).view(B, -1, self.bottlensize[0] * self.bottlensize[1] * self.bottlensize[2]).permute(0, 2,
                                                                                                               1).contiguous()
        x2 = self.line(x2)
        for i in self.block: ##加入了swin-transformer的模块
            x2 = i(x2)
        #x2 = self.project(x2)
        x2 = x2.view(B, self.bottlensize[0], self.bottlensize[1], self.bottlensize[2], -1).permute(0, 4, 1, 2,
                                                                                                   3).contiguous()
        x2 =  self.trans_1(skip3 + x2)
        x2 = self.trans_2(skip2 + x2)+ skip1
        x2 = self.trans_3(x2)
        x2 = self.out(x2)
        # x = x.view(B, C, D, H, W)
        # x = self.out(x)
        return torch.sigmoid(x2)

if __name__ == '__main__':
    '''
    128 x 128 x 128  (建议分辨率)
    TokenSeg x12L flops: 700.30 G, params: 30.23 M
    TokenSeg x8L flops: 638.70 G, params: 15.17 M
    TokenSeg x4L flops: 619.97 G, params: 10.59 M

    TokenSeg x12M flops: 265.07 G, params: 25.04 M
    TokenSeg x8M flops: 203.53 G, params: 9.99 M
    TokenSeg x4M flops: 184.84 G, params: 5.42 M

    TokenSeg x12s flops: 150.73 G, params: 23.73 M
    TokenSeg x8s flops: 89.22 G, params: 8.69 M
    TokenSeg x4s flops: 70.55 G, params: 4.12 M
    runet16  flops: 404.09 G, params: 11.16 M 
    runet32  flops: 1402.49 G, params: 44.61 M

    配置
                    TransformerNum      hidden_size   base_channeel
     TokenSeg x12L         12                 384           64
     TokenSeg x8L           8                 256           64
     TokenSeg x4L           4                 192           64

     TokenSeg x12m         12                 384           32
     TokenSeg x8m           8                 256           32
     TokenSeg x4m           4                 192           32

     TokenSeg x12s         12                 384           16
     TokenSeg x8s           8                 256           16
     TokenSeg x4s           4                 192           16
    '''
    from thop import profile

    #model = TokenSeg(inch=2, TransformerLayerNum=4, hidden_size=192, base_channeel=64).cuda()
    model = TokenSegV2(inch=2, TransformerLayerNum=4, hidden_size=192, base_channeel=16).cuda()
    img = torch.randn((1, 2, 128, 128, 128)).cuda()
    # Token = torch.randn((1, 16*16*16, 16, 16, 16)).cuda()
    print(model(img).size())
    flops, params = profile(model, (img,))
    print('flops: %.2f G, params: %.2f M' % (flops / 1e9, params / 1e6))