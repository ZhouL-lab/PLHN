from torch import nn
import torch
import torch.nn.functional as F
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

class PR_net2(nn.Module):
    def __init__(self,value=True,c=4,n=32,channels=128,groups = 1,norm='bn', num_classes=4,attenchan=64,growth_rate=16, block_config=(6, 12, 24, 16),
                 num_init_features=32, bn_size=4, drop_rate=0):
        super(PR_net2, self).__init__()
        self.training = value
        self.channels=channels
        self.achannel=attenchan

 # For network1
        ###For network2
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(2, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv3d(num_init_features, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm1', nn.BatchNorm3d(num_init_features)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(num_init_features, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
        ]))
        self.features_bn = nn.Sequential(OrderedDict([
            ('norm2', nn.BatchNorm3d(num_init_features)),
            ('relu2', nn.ReLU(inplace=True)),
        ]))
        self.conv_pool_first = nn.Conv3d(num_init_features, num_init_features, kernel_size=2, stride=2, padding=0,
                                         bias=False)

        # Each denseblock
        num_features = num_init_features
        self.dense_blocks = nn.ModuleList([])
        self.transit_blocks = nn.ModuleList([])
        self.upsampling_blocks = nn.ModuleList([])

        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)

            self.dense_blocks.append(block)

            num_features = num_features + num_layers * growth_rate

            up_block = nn.ConvTranspose3d(num_features, num_classes, kernel_size=2 ** (i + 1) + 2,
                                          stride=2 ** (i + 1),
                                          padding=1, groups=num_classes, bias=False)

            self.upsampling_blocks.append(up_block)

            if i != len(block_config) - 1:


                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.transit_blocks.append(trans)
                #self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        

        self.bn_class = nn.BatchNorm3d(num_classes * 4  +num_init_features)
            
        self.conv_class = nn.Conv3d(num_classes * 4+num_init_features , num_classes, kernel_size=1, padding=0)


    def forward(self, x):
        #####Network1 brach##
        img=x.clone()
        ####
        ####Network 2 ###
        first_three_features = self.features(x)
        first_three_features_bn = self.features_bn(first_three_features)
        out1 = self.conv_pool_first(first_three_features_bn)

        #print(out1.size()) # H//2 32
        out2 = self.dense_blocks[0](out1) #128,32
        up_block1 = self.upsampling_blocks[0](out2)
        out3 = self.transit_blocks[0](out2)    
        #print(out3.size())  #H/4 64
        out4 = self.dense_blocks[1](out3)  #256,128
        
        up_block2 = self.upsampling_blocks[1](out4)
        out5 = self.transit_blocks[1](out4)
        #print(out5.size())  ##H/8 128
        out6 = self.dense_blocks[2](out5) #512, 256
        ######################################
        up_block3 = self.upsampling_blocks[2](out6)
        out7 = self.transit_blocks[2](out6) ##H/16
        out7 = self.dense_blocks[3](out7)
        up_block4 = self.upsampling_blocks[3](out7)
        out8 =  torch.cat([up_block1, up_block2, up_block3, up_block4, first_three_features], 1)
        # ----------------------- classifier -----------------------
        out = self.conv_class(F.relu(self.bn_class(out8)))
        denseout=torch.sigmoid(out)

        return denseout