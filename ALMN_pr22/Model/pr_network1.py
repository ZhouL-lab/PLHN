from torch import nn
import torch
import torch.nn.functional as F
from Model.attention import multihead_attention_3d
from collections import OrderedDict

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))

        self.add_module('pool_norm', nn.BatchNorm3d(num_output_features))
        self.add_module('pool_relu', nn.ReLU(inplace=True))
        self.add_module('pool', nn.Conv3d(num_output_features, num_output_features, kernel_size=2, stride=2))


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
        # self.conv3x3x3_m2 = DilatedConv3DBlock(num_out, num_out, kernel_size=(1, 3, 3), stride=(1,1,1), g=g,d=(1,1,1), norm=norm)

        # skip connection
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

class PR_net1(nn.Module):
    def __init__(self,value=True,c=4,n=32,channels=128,groups = 1,norm='bn', num_classes=4,attenchan=64,growth_rate=16, block_config=(6, 12, 24, 16),
                 num_init_features=32, bn_size=4, drop_rate=0):
        super(PR_net1, self).__init__()
        self.training = value
        self.channels=channels
        self.achannel=attenchan

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

        self.upsample6 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # He

    def forward(self, x):
        #####Network1 brach##
        img=x.clone()
        ####
        ####The flowchart for network2###
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
        #####
        seg1=self.segaspp_conv1(asp1)
        seg2=self.segaspp_conv2(asp2)
        seg3=self.segaspp_conv3(asp3) ##to expand the vision
        pred_att05=(seg1+seg2+seg3)/3 ##three sigmod is hard for being trained #to add other operation
        ###
        pred_att=self.upsample6(pred_att05) 
        pred_att=torch.sigmoid(pred_att)
        outf=ay4 #self.upsample7(ay4)

        return pred_org, pred_att,outf
