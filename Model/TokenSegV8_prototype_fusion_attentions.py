import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_3tuple, trunc_normal_
import torch.nn.functional as F
from sklearn.cluster import KMeans
from Model.modelv5.logger import Logger as Log
from Model.modelv5.contrast import momentum_update, l2_normalize, ProjectionHead
from Model.modelv5.sinkhorn import distributed_sinkhorn
from timm.models.layers import trunc_normal_
from einops import rearrange, repeat
from Model.modelv5.loss_proto import PixelPrototypeCELoss,PPD,PPC
import numpy as np
import torch.distributed as dist
import time

def normalization(planes, norm='bn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    elif norm == 'sync_bn':
        m = nn.BatchNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m

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

# class LearnedPositionalEncoding(nn.Module):
#     def __init__(self,position_size, embedding_dim):
#         super(LearnedPositionalEncoding, self).__init__()
#
#         self.position_embeddings = nn.Parameter(torch.zeros(1, position_size, embedding_dim)) #8x
#
#     def forward(self, x):
#
#         position_embeddings = self.position_embeddings
#         return x + position_embeddings

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

        qkv = self.qkv(x) ##一个特征分为3个部分;
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

class Fusion_Attention(nn.Module):

    def __init__(self, dim, qk_scale=1, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.scale = qk_scale
        #trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q,k,v):
        #B_, N, C = x.shape
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1).contiguous())
        #attn = attn #+  self.position_embeddings
        attn = self.softmax(attn)
        #attn = self.attn_drop(attn)
        x = (attn @ v).contiguous() #reshape(B_, N, C) .transpose(1, 2)
        #x = self.proj(x)
        #x = self.proj_drop(x)
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

def BNReLU(num_features, bn_type=None, **kwargs):
    if bn_type == 'torchbn':
        return nn.Sequential(
            nn.BatchNorm3d(num_features, **kwargs),
            nn.ReLU()
        )
    elif bn_type == 'torchsyncbn':
        return nn.Sequential(
            nn.SyncBatchNorm(num_features, **kwargs),
            nn.ReLU()
        )
    else:
        Log.error('Not support BN type: {}.'.format(bn_type))
        exit(1)


class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, proj='convmlp', bn_type='torchbn'):
        super(ProjectionHead, self).__init__()

        Log.info('proj_dim: {}'.format(proj_dim))

        if proj == 'linear':
            self.proj = nn.Conv3d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv3d(dim_in, dim_in, kernel_size=1),
                BNReLU(dim_in, bn_type=bn_type),
                nn.Conv3d(dim_in, proj_dim, kernel_size=1)
            )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)

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

def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1), ##用了反卷积的结构。
        nn.BatchNorm3d(out_dim),
        activation,)

class TokenSegV8(nn.Module):
    def __init__(self, inch=1, outch=1, downlayer=3, base_channeel=32, imgsize=[128, 128, 128], hidden_size=256,
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

        ####模型P-1
        self.proxylayers = nn.ModuleList()
        self.proxylayers.append(conv_block_2_3d(inch, base_channeel, activation)) ## N
        self.proxylayers.append(conv_block_2_3d(base_channeel, base_channeel*2, activation)) ## 2*N
        self.proxylayers.append(conv_block_2_3d(base_channeel*2, hidden_size, activation)) ### hs
        #self.proxylayers.append(conv_block_2_3d(base_channeel * 4, base_channeel * 8, activation))
        #self.proxylayers.append(conv_block_2_3d(base_channeel * 8, base_channeel * 8, activation))

        ####第二个encoder分支####
        self.stemlayers = nn.ModuleList()
        self.stemlayers.append(conv_block_2_3d(inch, base_channeel, activation))
        for i in range(1, downlayer): ##1,2
            self.stemlayers.append(
                conv_block_2_3d(base_channeel * (2 ** (i - 1)), base_channeel * (2 ** i), activation)) ###
        self.stem = nn.Sequential(
            *self.stemlayers
        )
        self.line = nn.Linear(base_channeel * (2 ** (downlayer - 1)), hidden_size)
       # self.project = nn.Linear(hidden_size, 8 ** downlayer * inch)
        self.block = nn.ModuleList()  # [SwinTransformerBlock(192,[32,32,32],8)]*10

        for i in range(TransformerLayerNum):
            self.block.append(SwinTransformerBlock(hidden_size, self.bottlensize, window_size)) ##叠加了transformer模块。
        self.window_size = window_size
        self.shift_size = self.window_size // 2

        #####基于反卷积网络的decoder####
        self.trans_1 = conv_trans_block_3d(hidden_size, base_channeel*2, activation) ## 2*N
        self.trans_2 = conv_trans_block_3d(base_channeel*2, base_channeel , activation) ## N
        self.trans_3 = conv_trans_block_3d(base_channeel, base_channeel, activation)  ## N
        self.out = nn.Conv3d(base_channeel, outch, 3, 1, 1)

        #-------------------------prototype----------------------
        self.gamma = 0.999
        self.num_prototype = 10
        self.use_prototype = True
        self.update_prototype = True
        self.pretrain_prototype = False
        self.num_classes = 2

        in_channels = 32

        self.cls_head = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            BNReLU(in_channels, bn_type='torchbn'),
            nn.Dropout3d(0.10)
        )  # cls头

        self.prototypes = nn.Parameter(torch.zeros(self.num_classes, self.num_prototype, in_channels),
                                       requires_grad=True) ##是需要进行梯度更新的;

        self.proj_head = ProjectionHead(in_channels, in_channels)  ##后面就不进行分类了。
        self.feat_norm = nn.LayerNorm(in_channels)
        self.mask_norm = nn.LayerNorm(self.num_classes)

        self.out_1 = nn.Conv3d(20, outch, 3, 1, 1)

        #for p in self.parameters(): ## 只训练一部分参数。
        #    p.requires_grad = False

    ##############新加的输出卷积###################
        self.fatten=Fusion_Attention(32)
        self.out_2 = nn.Sequential( ###prototypes相关的特征。
            nn.Conv3d(52, outch, kernel_size=3, stride=1, padding=1)
        )
        trunc_normal_(self.prototypes, std=0.02)

    def prototype_learning(self, _c, prob,out_seg, gt_seg, masks): ###以下为更新prototypes的模块；

        pred_seg = torch.max(out_seg, 1)[1] #(prob>=0.5).type(torch.float32)
        mask = (gt_seg == pred_seg.view(-1))  ##哪些完全预测正确？
        cosine_similarity = torch.mm(_c, self.prototypes.view(-1, self.prototypes.shape[-1]).t())  # mm是点乘矩阵，.view是reshape矩阵，-1代表不确定及根据已知的数reshape。
        proto_logits = cosine_similarity
        proto_target = gt_seg.clone().float()
        # clustering for each class
        protos = self.prototypes.data.clone()
        for k in range(self.num_classes):
            init_q = masks[..., k]  # 每一类的mask
            init_q = init_q[gt_seg == k, ...]  ##获取初始LABEL，MASK相关的信息；
            if init_q.shape[0] == 0:
                continue
            q, indexs = distributed_sinkhorn(init_q)  ##获取提取信息的索引；与距离

            m_k = mask[gt_seg == k]
            print(mask.shape)

            c_k = _c[gt_seg == k, ...]  # _c是变化了数据结构的特征，b c h w -> (b h w) c。
            print(c_k.shape)
            m_k_tile = repeat(m_k, 'n -> n tile',
                              tile=self.num_prototype)  # einops库，用爱因斯坦标识操作维度，把m_k增加到增加维度，[n,10,k,10]
            m_q = q * m_k_tile  # n x self.num_prototype [m_k,10]
            c_k_tile = repeat(m_k, 'n -> n tile',
                              tile=c_k.shape[-1])  ##根据所有的维度，进行repeating操作。c_k最后一维是c及增加通道数。[m_k,c]
            c_q = c_k * c_k_tile  # n x embedding_dim ##提取特征[n,c]*[n,10,k,c]
            f = m_q.transpose(0,1) @ c_q  # self.num_prototype x embedding_dim；##进行矩阵相乘。转换维度0-1 [10,n,k,10]@[n,10,k,c]
            n = torch.sum(m_q, dim=0)  ## 有一定相似度的看做同类？
            if torch.sum(n) > 0 and self.update_prototype is True:  ###判断是否需要更新prototypes信息，如果需要更新；
                f = F.normalize(f, p=2, dim=-1)  ##对特征进行归一化操作。
                new_value = momentum_update(old_value=protos[k, n != 0, :], new_value=f[n != 0, :], ###通过soft assignment来计算。
                                            ##对相关prototypes这些信息进行更新。。
                                            momentum=self.gamma, debug=False)  ##这样更新具有针对性。
                protos[k, n != 0, :] = new_value

            proto_target[gt_seg == k] = indexs.float() + (self.num_prototype * k)

        self.prototypes = nn.Parameter(l2_normalize(protos),
                                       requires_grad=False)  ###将prototypes定义为一组参数。

        if dist.is_available() and dist.is_initialized():
            protos = self.prototypes.data.clone()
            dist.all_reduce(protos.div_(dist.get_world_size()))
            self.prototypes = nn.Parameter(protos, requires_grad=False)  ##proto参数化，可以方便后面进行更新；
        return proto_logits, proto_target

    def forward(self, x, gt_semantic_seg=None, pretrain_prototype=False,TEST=0,epoch=1, index=0,max_len =200):
        B, C, D, H, W = x.size()
        skip1 = self.proxylayers[0](x)
        skip2 = self.proxylayers[1](skip1)
        skip3 = self.proxylayers[2](skip2)

        x2 = self.stem(x).view(B, -1, self.bottlensize[0] * self.bottlensize[1] * self.bottlensize[2]).permute(0, 2,
                                                                                                               1).contiguous()
        x2 = self.line(x2) ###先进行变换，通道到hidden_size;
        for i in self.block:
            x2 = i(x2)
        x2 = x2.view(B, self.bottlensize[0], self.bottlensize[1], self.bottlensize[2], -1).permute(0, 4, 1, 2,
                                                                   3).contiguous()
        x2 =  self.trans_1(skip3 + x2)
        x2 = self.trans_2(skip2 + x2)+ skip1
        x2 = self.trans_3(x2)
        proto=x2
        x2 = self.out(x2)
        ###############################################
        '''
        .Prototye loss calculat
        '''
        if 0==0:
        ######################################
            _, _, d, h, w = proto.size()
            c = self.cls_head(proto)  ##只提取了特征，不加后面的分类HEAD？
            c = self.proj_head(c)
            _c = rearrange(c, 'b c d h w -> (b d h w) c')  ##组织数据结构;
            _c = self.feat_norm(_c)
            _c = l2_normalize(_c)
            self.prototypes.data.copy_(l2_normalize(self.prototypes))

        ################ n: h*w, k: num_class, m: num_prototype
            masks = torch.einsum('nd,kmd->nmk', _c, self.prototypes)  ###矩阵乘法，直接根据计算出的与PROTOS的相似性，来分类；[n,10,k]
            new_pro=rearrange(self.prototypes,'h w c->(h w) c')  ### (20*C)
            mprob=masks.reshape(masks.shape[0],masks.shape[1]*masks.shape[2]) ####得到prototypes与不同类别之间的权重值。
            masks_seg = rearrange(mprob, "(b d h w) k -> b k d h w", b=x.shape[0], d=x.shape[2], h=x.shape[3]) ###相似性矩阵，如何解耦出来？
            pro_seg = self.out_1(masks_seg)  ###得到基于prototype的分割概率;

            ATTEN=1
            if ATTEN==1:
                new_fea = self.fatten(_c, new_pro, new_pro)  ##注意力机制。
                _cn=rearrange(_c, "(b d h w) k -> b k d h w", b=x.shape[0], d=x.shape[2], h=x.shape[3])
                new_c=rearrange(new_fea, "(b d h w) k -> b k d h w", b=x.shape[0], d=x.shape[2], h=x.shape[3])+_cn

            if ATTEN==1:
                new_c = torch.cat([new_c, masks_seg], 1)  ##k=32+20=52
                fusion_seg = self.out_2(new_c)
            else:
                fusion_seg=pro_seg

        ########### 用softmax进行优化的结果  ###############
            out_seg = torch.amax(masks, dim=1)  ##与哪组的prototypes 的相似度最小，就会被分配为哪个类别？
            out_seg = self.mask_norm(out_seg)
            out_seg = rearrange(out_seg, "(b d h w) k -> b k d h w", b=x.shape[0], d=x.shape[2], h=x.shape[3])
            out_d_seg = torch.max(out_seg, 1)[1].unsqueeze(1)

        #########################################
            if pretrain_prototype is False and self.use_prototype is True and gt_semantic_seg is not None:
                gt_seg = F.interpolate(gt_semantic_seg.float(), size=x.size()[2:], mode='nearest').view(-1)
                #gt_seg = gt_semantic_seg.float().view(-1)
                contrast_logits, contrast_target = self.prototype_learning(_c, torch.sigmoid(pro_seg),out_seg, gt_seg, masks)
                return {'seg': out_seg, 'logits': contrast_logits, 'target': contrast_target},torch.sigmoid(pro_seg),torch.sigmoid(x2),torch.sigmoid(fusion_seg)
            return masks_seg,torch.sigmoid(pro_seg),torch.sigmoid(x2),torch.sigmoid(fusion_seg) ###two channels+one channel.

if __name__ == '__main__':
    '''
    128 x 128 x 128  (建议分辨率)
    TokenSeg x12L flops: 700.30 G, params: 30.23 M
    TokenSeg x8L flops: 638.70 G, params: 15.17 M
    TokenSeg x4L flops: 619.97 G, params: 10.59 M

    TokenSeg x12M flops: 265.07 G, params: 25.04 M
    TokenSeg x8M flops: 203.53 G, params: 9.99 M  ##对这种网络来说，不一定参数越大就越好。
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

    PPCELOSS = PixelPrototypeCELoss()
    #model = TokenSeg(inch=2, TransformerLayerNum=4, hidden_size=192, base_channeel=64).cuda()
    model = TokenSegV4(inch=1, TransformerLayerNum=4, hidden_size=192, base_channeel=16).cuda()
    img = torch.rand((1, 1, 128, 128, 128)).round().cuda()
    label = torch.rand((1, 1, 128, 128, 128)).round().cuda()
    # Token = torch.randn((1, 16*16*16, 16, 16, 16)).cuda()
    print('-----------------------------------')
    dict,proto_logist = model(img, label)
    lossce = PPCELOSS(dict, img)
    print(lossce)
    flops, params = profile(model, (img,))
    print('flops: %.2f G, params: %.2f M' % (flops / 1e9, params / 1e6))
