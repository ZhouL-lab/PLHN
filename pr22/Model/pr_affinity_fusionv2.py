from torch import nn
import torch
import torch.nn.functional as F
from Model.attention import multihead_attention_3d
from Model.affrefine3D import affrefine3D,affrefine3DONLY,affrefine3DSEG
from collections import OrderedDict
from Model.affinitynetutil import ConvGRUCellfeaturefusion, Donwconv,DonwconvT

class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        rate_list = (1, rate, rate)
        self.atrous_convolution = nn.Conv3d(inplanes, planes, kernel_size=3,
                                            stride=1, padding=rate_list, dilation=rate_list)
        self.group_norm = nn.GroupNorm(32, planes)
        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.group_norm(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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

class Conv3d_Block(nn.Module):
    def __init__(self,num_in,num_out,kernel_size=1,stride=1,g=1,padding=None,norm=None):
        super(Conv3d_Block, self).__init__()
        if padding == None:
            padding = (kernel_size - 1) // 2
        self.bn = normalization(num_in,norm=norm)
        self.act_fn = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(num_in, num_out, kernel_size=kernel_size, padding=padding,stride=stride, groups=g, bias=False)

    def forward(self, x): # BN + Relu + Conv
        h = self.act_fn(self.bn(x))
        h = self.conv(h)
        return h


class DilatedConv3DBlock(nn.Module):
    def __init__(self, num_in, num_out, kernel_size=(1,1,1), stride=1, g=1, d=(1,1,1), norm=None):
        super(DilatedConv3DBlock, self).__init__()
        assert isinstance(kernel_size,tuple) and isinstance(d,tuple)

        padding = tuple(
            [(ks-1)//2 *dd for ks, dd in zip(kernel_size, d)]
        )

        self.bn = normalization(num_in, norm=norm)
        self.act_fn = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(num_in,num_out,kernel_size=kernel_size,padding=padding,stride=stride,groups=g,dilation=d,bias=False)

    def forward(self, x):
        h = self.act_fn(self.bn(x))
        h = self.conv(h)
        return h


class MFunit(nn.Module):
    def __init__(self, num_in, num_out, g=1, stride=1, d=(1,1),norm=None):
        """  The second 3x3x1 group conv is replaced by 3x3x3.
        :param num_in: number of input channels
        :param num_out: number of output channels
        :param g: groups of group conv.
        :param stride: 1 or 2
        :param d: tuple, d[0] for the first 3x3x3 conv while d[1] for the 3x3x1 conv
        :param norm: Batch Normalization
        """
        super(MFunit, self).__init__()
        num_mid = num_in if num_in <= num_out else num_out
        self.conv1x1x1_in1 = Conv3d_Block(num_in,num_in//4,kernel_size=1,stride=1,norm=norm)
        self.conv1x1x1_in2 = Conv3d_Block(num_in//4,num_mid,kernel_size=1,stride=1,norm=norm)
        self.conv3x3x3_m1 = DilatedConv3DBlock(num_mid,num_out,kernel_size=(3,3,3),stride=stride,g=g,d=(d[0],d[0],d[0]),norm=norm) # dilated
        self.conv3x3x3_m2 = DilatedConv3DBlock(num_out,num_out,kernel_size=(3,3,1),stride=1,g=g,d=(d[1],d[1],1),norm=norm)
        # self.conv3x3x3_m2 = DilatedConv3DBlock(num_out,num_out,kernel_size=(1,3,3),stride=1,g=g,d=(1,d[1],d[1]),norm=norm)

        # skip connection
        if num_in != num_out or stride != 1:
            if stride == 1:
                self.conv1x1x1_shortcut = Conv3d_Block(num_in, num_out, kernel_size=1, stride=1, padding=0,norm=norm)
            if stride == 2:
                # if MF block with stride=2, 2x2x2
                self.conv2x2x2_shortcut = Conv3d_Block(num_in, num_out, kernel_size=2, stride=2,padding=0, norm=norm) # params

    def forward(self, x):
        x1 = self.conv1x1x1_in1(x)
        x2 = self.conv1x1x1_in2(x1)
        x3 = self.conv3x3x3_m1(x2)
        x4 = self.conv3x3x3_m2(x3)

        shortcut = x
        if hasattr(self,'conv1x1x1_shortcut'):
            shortcut = self.conv1x1x1_shortcut(shortcut)
        if hasattr(self,'conv2x2x2_shortcut'):
            shortcut = self.conv2x2x2_shortcut(shortcut)

        return x4 + shortcut ##similar to the architecture of resnet.

class DMFUnit(nn.Module):
    # weighred add
    def __init__(self, num_in, num_out, g=1, stride=1,norm=None,dilation=None):
        super(DMFUnit, self).__init__()
        self.weight1 = nn.Parameter(torch.ones(1))
        self.weight2 = nn.Parameter(torch.ones(1))
        self.weight3 = nn.Parameter(torch.ones(1))

        num_mid = num_in if num_in <= num_out else num_out

        self.conv1x1x1_in1 = Conv3d_Block(num_in, num_in // 4, kernel_size=1, stride=1, norm=norm)
        self.conv1x1x1_in2 = Conv3d_Block(num_in // 4,num_mid,kernel_size=1, stride=1, norm=norm)

        self.conv3x3x3_m1 = nn.ModuleList()
        if dilation == None:
            dilation = [1,2,3]
        for i in range(3):
            self.conv3x3x3_m1.append(
                DilatedConv3DBlock(num_mid,num_out, kernel_size=(3, 3, 3), stride=stride, g=g, d=(dilation[i],dilation[i], dilation[i]),norm=norm)
            )

        # It has not Dilated operation
        self.conv3x3x3_m2 = DilatedConv3DBlock(num_out, num_out, kernel_size=(3, 3, 1), stride=(1,1,1), g=g,d=(1,1,1), norm=norm)

        if num_in != num_out or stride != 1:
            if stride == 1:
                self.conv1x1x1_shortcut = Conv3d_Block(num_in, num_out, kernel_size=1, stride=1, padding=0, norm=norm)
            if stride == 2:
                self.conv2x2x2_shortcut = Conv3d_Block(num_in, num_out, kernel_size=2, stride=2, padding=0, norm=norm)

    def forward(self, x):
        x1 = self.conv1x1x1_in1(x)
        x2 = self.conv1x1x1_in2(x1)
        x3 = self.weight1*self.conv3x3x3_m1[0](x2) + self.weight2*self.conv3x3x3_m1[1](x2) + self.weight3*self.conv3x3x3_m1[2](x2)
        x4 = self.conv3x3x3_m2(x3)
        shortcut = x
        if hasattr(self, 'conv1x1x1_shortcut'):
            shortcut = self.conv1x1x1_shortcut(shortcut)
        if hasattr(self, 'conv2x2x2_shortcut'):
            shortcut = self.conv2x2x2_shortcut(shortcut)
        return x4 + shortcut

def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        activation,)

def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        activation,)

def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


def conv_block_2_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),)


class PR_AFFINITY_NET(nn.Module):
    def __init__(self,value=True,c=4,n=32,channels=128,groups = 1,norm='bn', num_classes=4,attenchan=64,num_filters=16):
        super(PR_AFFINITY_NET, self).__init__()
        self.training = value
        self.channels=channels
        self.achannel=attenchan

        self.in_dim = c
        self.out_dim =num_classes
        self.num_filters = num_filters
        activation = nn.LeakyReLU(0.2, inplace=True)
        act1 = nn.Sigmoid()

 # For network1
        self.encoder_block1 = nn.Conv3d(c, n, kernel_size=3, padding=1, stride=2, bias=False)  # H//2,the channel number
        self.encoder_block2 = nn.Sequential(
            MFunit(n, channels, g=groups, stride=2, norm=norm),  # H//4 down
            MFunit(channels, channels, g=groups, stride=1, norm=norm),
            MFunit(channels, channels, g=groups, stride=1, norm=norm)
        )
        #
        self.encoder_block3 = nn.Sequential(
            MFunit(channels, channels * 2, g=groups, stride=2, norm=norm),  # H//8
            MFunit(channels * 2, channels * 2, g=groups, stride=1, norm=norm),
            MFunit(channels * 2, channels * 2, g=groups, stride=1, norm=norm)
        )

        self.encoder_block4 = nn.Sequential(  # H//8,channels*4
            MFunit(channels * 2, channels * 3, g=groups, stride=2, norm=norm),  # H//16
            MFunit(channels * 3, channels * 3, g=groups, stride=1, norm=norm),
            MFunit(channels * 3, channels * 2, g=groups, stride=1, norm=norm),
        )

        self.upsample1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//8
        self.decoder_block1 = MFunit(channels * 2 + channels * 2, channels * 2, g=groups, stride=1, norm=norm)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//4
        self.decoder_block2 = MFunit(channels * 2 + channels, channels, g=groups, stride=1, norm=norm)

        self.upsample3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//2
        self.decoder_block3 = MFunit(channels + n, n, g=groups, stride=1, norm=norm)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H

        self.seg = nn.Conv3d(n, num_classes, kernel_size=1, padding=0, stride=1, bias=False)


        self.decoder_block1_a = MFunit(channels * 2 + channels * 2, channels * 2, g=groups, stride=1, norm=norm)

        self.decoder_block2_a = MFunit(channels * 2 + channels, channels, g=groups, stride=1, norm=norm) ##一种light-weight的网络结构；

        self.decoder_block4 = MFunit(channels + n, n, g=groups, stride=1, norm=norm)
        self.upsample5 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H
        self.seg1 = nn.Conv3d(n, num_classes, kernel_size=1, padding=0, stride=1, bias=False)


        self.multihead_attention_3dPen = multihead_attention_3d(self.training, channels * 2, 2 * channels, 2 * channels,
                                                               2 * channels,
                                                               2 * channels // 64, False)  ##一个是position attention

        self.multihead_attention_3dP0 = multihead_attention_3d(self.training, channels * 4, 4 * channels, 4 * channels,
                                                               4 * channels,
                                                               4 * channels //128,False) ##一个是position attention


        self.up_block2 = nn.Sequential(
            nn.ConvTranspose3d(3*channels, 3*channels, kernel_size=4,stride=2,
                                          padding=1, groups=1),nn.BatchNorm3d(3*channels), nn.ReLU(inplace=True),
            MFunit(channels*3, channels, g=groups, stride=1, norm=norm)
            )

        self.refine = nn.Sequential(
            nn.Conv3d(channels+n, channels, kernel_size=3,padding=1), nn.BatchNorm3d(channels), nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1), nn.BatchNorm3d(channels), nn.ReLU(inplace=True),
            nn.Conv3d(channels, self.achannel, kernel_size=3, padding=1), nn.BatchNorm3d(self.achannel), nn.ReLU(inplace=True)
        )

        rates = (1, 6, 12)
        self.aspp1 = ASPP_module(self.achannel, 64, rate=rates[0])
        self.aspp2 = ASPP_module(self.achannel, 64, rate=rates[1])
        self.aspp3 = ASPP_module(self.achannel, 64, rate=rates[2])
        #self.aspp4 = ASPP_module(self.achannel, 64, rate=rates[3])

        self.segaspp_conv1 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3,padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=3,padding=1), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.Conv3d(n, num_classes, kernel_size=3, padding=1, stride=1, bias=False)           
        )

        self.segaspp_conv2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3,padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=3,padding=1), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.Conv3d(n, num_classes, kernel_size=3, padding=1, stride=1, bias=False)
        )

        self.segaspp_conv3 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3,padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=3,padding=1), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.Conv3d(n, num_classes, kernel_size=3, padding=1, stride=1, bias=False)
        )

        self.upsample6 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H
        for p in self.parameters(): ## 只训练一部分参数。
         #   print(p)
            p.requires_grad = False

    ########For network2
        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filters, activation)
        self.res_1 = conv_block_3d(self.in_dim, self.num_filters, activation)
        self.pool_1 = max_pooling_3d()

        self.down_2 = conv_block_2_3d(self.num_filters, self.num_filters * 2, activation)
        self.res_2 = conv_block_3d(self.num_filters, self.num_filters * 2, activation)
        self.pool_2 = max_pooling_3d()

        self.down_3 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation)
        self.res_3 = conv_block_3d(self.num_filters * 2, self.num_filters * 4, activation)
        self.pool_3 = max_pooling_3d()

        self.down_4 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation)
        self.res_4 = conv_block_3d(self.num_filters * 4, self.num_filters * 8, activation)
        self.pool_4 = max_pooling_3d()

        # Bridge
        self.bridge = conv_block_2_3d(self.num_filters * 8, self.num_filters * 16, activation)
        self.res_bridge = conv_block_3d(self.num_filters * 8, self.num_filters * 16, activation)

        self.trans_2 = conv_trans_block_3d(self.num_filters * 16, self.num_filters * 16, activation)
        self.up_2 = conv_block_2_3d(self.num_filters * 24, self.num_filters * 8, activation)
        self.res_up2 = conv_block_3d(self.num_filters * 24, self.num_filters * 8, activation)

        self.trans_3 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation)
        self.up_3 = conv_block_2_3d(self.num_filters * 12, self.num_filters * 4, activation)
        self.res_up3 = conv_block_3d(self.num_filters * 12, self.num_filters * 4, activation)

        self.trans_4 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation)
        self.up_4 = conv_block_2_3d(self.num_filters * 6, self.num_filters * 2, activation)
        self.res_up4 = conv_block_3d(self.num_filters * 6, self.num_filters * 2, activation)

        self.trans_5 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation)
        self.up_5 = conv_block_2_3d(self.num_filters * 3, self.num_filters * 1, activation)

        # Output
        self.out = conv_block_3d(self.num_filters, self.out_dim, act1)

        ####Affinity net.
        for p in self.parameters(): ## 只训练一部分参数。
         #   print(p)
            p.requires_grad = False

    ###################
        in_features=34 ##16+2+16
        out_features=16
        ###AF1, AF2####
        self.affsubnetwork1 = nn.Sequential(
            nn.Conv3d(64*3, 64, kernel_size=3,padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.Conv3d(64, 16, kernel_size=3, padding=1, stride=1, bias=False),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        )

        self.affsubnetwork2 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=3, padding=1, stride=1, bias=False)
        )

        ##### AF3 ##
        self.down_block=DonwconvT(in_features,out_features) ##

        self.up_block = nn.ConvTranspose3d(out_features, out_features, kernel_size=2 ,
                                          stride=2 ,
                                          padding=1, groups=1, bias=False)

        ####FUSION####
        self.Gru3dAF1=ConvGRUCellfeaturefusion(26,26,1,3) ##Fusion in H/2
        self.Gru3dAF2=ConvGRUCellfeaturefusion(26,26,1,1) ##Fusion in H/2

        PAMR_KERNEL = [1] ## 3*3 neighbouing
        PAMR_ITER = 10 ## iter number
        
        self._aff = affrefine3D(PAMR_ITER, PAMR_KERNEL)
        self._affonly = affrefine3DONLY(PAMR_ITER, PAMR_KERNEL)
        self._affseg=affrefine3DSEG(PAMR_ITER, PAMR_KERNEL)

    def forward(self, x):
        #####Network-1#####
        img=x.clone()
        x1 = self.encoder_block1(x)  # H//2 down--n 32
        x2 = self.encoder_block2(x1)  # H//4 down--channels 128
        x3 = self.encoder_block3(x2)  # H//8 down--2*channels 256
        x4 = self.encoder_block4(x3)  # H//16-- 2* channels
        x4=self.multihead_attention_3dPen(x4)
        # Decoder
        y1 = self.upsample1(x4)  # H//8
        y11 = torch.cat([x3, y1], dim=1)
        ### To add attention, to modify the decoder architecture ####
        y11 =self.multihead_attention_3dP0(y11)  #
        y1a = self.decoder_block1_a(y11)  #  H//8, 256
        y2a = self.upsample2(y1a)  # H//4
        y21a = torch.cat([x2, y2a], dim=1)  # 3*ch
        y2a = self.decoder_block2_a(y21a) ##H/4, 128

        y3a = self.upsample3(y2a)  
        y31a = torch.cat([x1, y3a], dim=1)  # ch+n
        #y31a = self.multihead_attention_3dP2(y31a) #ch+n
        ay3 = self.decoder_block4(y31a)  # H/2 32
        
        ay4 = self.upsample4(ay3)
        pred = self.seg1(ay4)  #能否用不同的LOSS？
        pred_org=torch.sigmoid(pred)
        down2 =self.up_block2(y21a) #F.upsample(y2a, size=x1.size()[2:], mode='trilinear') ##ch
        #print(down2.size())
        refine=self.refine(torch.cat((ay3,down2),1)) ##
        asp1=self.aspp1(refine)
        asp2=self.aspp2(refine)
        asp3=self.aspp3(refine)

        seg1=self.segaspp_conv1(asp1)
        seg2=self.segaspp_conv2(asp2)
        seg3=self.segaspp_conv3(asp3) ##to expand the vision
        pred_att05=(seg1+seg2+seg3)/3 ##three sigmod is hard for being trained #to add other operation
        ###
        pred_att=self.upsample6(pred_att05) 
        pred_att=torch.sigmoid(pred_att)
        outf=ay4 #self.upsample7(ay4)

        ####Network 2 ###
        down_1 = self.down_1(x)  # -> [1, 4, 128, 128, 128]
        res_1 = self.res_1(x)
        pool_1 = self.pool_1(down_1 + res_1)  # -> [1, 4, 64, 64, 64]
        down_2 = self.down_2(pool_1)  # -> [1, 8, 64, 64, 64]
        res_2 = self.res_2(pool_1)
        pool_2 = self.pool_2(down_2 + res_2)  # -> [1, 8, 32, 32, 32]
        down_3 = self.down_3(pool_2)  # -> [1, 16, 32, 32, 32]
        res_3 = self.res_3(pool_2)
        pool_3 = self.pool_3(down_3 + res_3)  # -> [1, 16, 16, 16, 16]
        down_4 = self.down_4(pool_3)  # -> [1, 32, 16, 16, 16]
        res_4 = self.res_4(pool_3)
        pool_4 = self.pool_4(down_4 + res_4)  # -> [1, 32, 8, 8, 8]
        # Bridge
        bridge = self.bridge(pool_4)  # -> [1, 128, 4, 4, 4]
        res_bridge = self.res_bridge(pool_4)
        trans_2 = self.trans_2(bridge + res_bridge)  # -> [1, 64, 16, 16, 16]
        concat_2 = torch.cat([trans_2, down_4 + res_4], dim=1)  # -> [1, 96, 16, 16, 16]
        up_2 = self.up_2(concat_2)  # -> [1, 32, 16, 16, 16]
        res_up_2 = self.res_up2(concat_2)

        trans_3 = self.trans_3(up_2 + res_up_2)  # -> [1, 32, 32, 32, 32]
        concat_3 = torch.cat([trans_3, down_3 + res_3], dim=1)  # -> [1, 48, 32, 32, 32]
        up_3 = self.up_3(concat_3)  # -> [1, 16, 32, 32, 32]
        res_up_3 = self.res_up3(concat_3)

        trans_4 = self.trans_4(up_3 + res_up_3)  # -> [1, 16, 64, 64, 64]
        concat_4 = torch.cat([trans_4, down_2 + res_2], dim=1)  # -> [1, 24, 64, 64, 64]
        up_4 = self.up_4(concat_4)  # -> [1, 8, 64, 64, 64]
        res_up_4 = self.res_up4(concat_4)
        trans_5 = self.trans_5(up_4 + res_up_4)  # -> [1, 8, 128, 128, 128]
        concat_5 = torch.cat([trans_5, down_1 + res_1], dim=1)  # -> [1, 12, 128, 128, 128]
        up_5 = self.up_5(concat_5)  # -> [1, 4, 128, 128, 128]
        # Output
        out = self.out(up_5)  # -> [1, 3, 128, 128, 128]
    ####Network for learning features for ensemble learning ##
        Faff1 = self.affsubnetwork1(torch.cat([asp1, asp2, asp3], 1))  ##To produce the affinity matrixes via multi-scale features.
        Faff2 = self.affsubnetwork2(up_5)  ##To produce the affinity via the output features.
        ##################################
        mask1_ref, aff1 = self._aff(Faff1, pred_att, [], sign=0)
        mask2_ref, aff2 = self._aff(Faff2, out, [], sign=0)
        finalseg = mask1_ref * 0.5 + mask2_ref * 0.5  ## 两个网络融合的分割结果，后面测试下动态加权和，效果如何。

        finalseg=torch.where(finalseg>1,torch.ones_like(finalseg),finalseg)

        return finalseg, pred_att, out  ##最后的分割结果，和中间融合的分割结果