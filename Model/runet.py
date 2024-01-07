# x: 128x128 resolution for 32 frames.
import torch
import torch.nn as nn


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

class RUNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):
        super(RUNet, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.LeakyReLU(0.2, inplace=True)
        act1=nn.Sigmoid()

        
        # Down sampling
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

        # self.down_5 = conv_block_2_3d(self.num_filters * 8, self.num_filters * 16, activation)
        # self.res_5 = conv_block_3d(self.num_filters * 8, self.num_filters * 16, activation)
        # self.pool_5 = max_pooling_3d()
        
        # Bridge
        self.bridge = conv_block_2_3d(self.num_filters * 8, self.num_filters * 16, activation)
        self.res_bridge = conv_block_3d(self.num_filters * 8, self.num_filters * 16, activation)
        # Up sampling
        # self.trans_1 = conv_trans_block_3d(self.num_filters * 32, self.num_filters * 32, activation)
        # self.up_1 = conv_block_2_3d(self.num_filters * 48, self.num_filters * 16, activation)
        # self.res_up1 = conv_block_3d(self.num_filters * 48, self.num_filters * 16, activation)

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
        self.out = conv_block_3d(self.num_filters, out_dim, act1)
    
    def forward(self, x):
        # Down sampling
        down_1 = self.down_1(x) # -> [1, 4, 128, 128, 128]
        res_1 = self.res_1(x)
        pool_1 = self.pool_1(down_1+res_1) # -> [1, 4, 64, 64, 64]
        
        down_2 = self.down_2(pool_1) # -> [1, 8, 64, 64, 64]
        res_2 = self.res_2(pool_1)
        pool_2 = self.pool_2(down_2+res_2) # -> [1, 8, 32, 32, 32]
        
        down_3 = self.down_3(pool_2) # -> [1, 16, 32, 32, 32]
        res_3 = self.res_3(pool_2)
        pool_3 = self.pool_3(down_3+res_3) # -> [1, 16, 16, 16, 16]
        
        down_4 = self.down_4(pool_3) # -> [1, 32, 16, 16, 16]
        res_4 = self.res_4(pool_3)
        pool_4 = self.pool_4(down_4+res_4) # -> [1, 32, 8, 8, 8]
        
        # down_5 = self.down_5(pool_4) # -> [1, 64, 8, 8, 8]
        # res_5 = self.res_5(pool_4)
        # pool_5 = self.pool_5(down_5+res_5) # -> [1, 64, 4, 4, 4]
        
        # Bridge
        bridge = self.bridge(pool_4) # -> [1, 128, 4, 4, 4]
        res_bridge = self.res_bridge(pool_4)
        
        # Up sampling
        # trans_1 = self.trans_1(bridge+res_bridge) # -> [1, 128, 8, 8, 8]
        # concat_1 = torch.cat([trans_1, down_5+res_5], dim=1) # -> [1, 192, 8, 8, 8]
        # up_1 = self.up_1(concat_1) # -> [1, 64, 8, 8, 8]
        # res_up_1 = self.res_up1(concat_1)
        
        trans_2 = self.trans_2(bridge+res_bridge) # -> [1, 64, 16, 16, 16]
        concat_2 = torch.cat([trans_2, down_4+res_4], dim=1) # -> [1, 96, 16, 16, 16]
        up_2 = self.up_2(concat_2) # -> [1, 32, 16, 16, 16]
        res_up_2 = self.res_up2(concat_2)
        
        trans_3 = self.trans_3(up_2+res_up_2) # -> [1, 32, 32, 32, 32]
        concat_3 = torch.cat([trans_3, down_3+res_3], dim=1) # -> [1, 48, 32, 32, 32]
        up_3 = self.up_3(concat_3) # -> [1, 16, 32, 32, 32]
        res_up_3 = self.res_up3(concat_3)
        
        trans_4 = self.trans_4(up_3+res_up_3) # -> [1, 16, 64, 64, 64]
        concat_4 = torch.cat([trans_4, down_2+res_2], dim=1) # -> [1, 24, 64, 64, 64]
        up_4 = self.up_4(concat_4) # -> [1, 8, 64, 64, 64]
        res_up_4 = self.res_up4(concat_4)
        
        trans_5 = self.trans_5(up_4+res_up_4) # -> [1, 8, 128, 128, 128]
        concat_5 = torch.cat([trans_5, down_1+res_1], dim=1) # -> [1, 12, 128, 128, 128]
        up_5 = self.up_5(concat_5) # -> [1, 4, 128, 128, 128]
        
        # Output
        out = self.out(up_5) # -> [1, 3, 128, 128, 128]
        return out

class RUnet_encoder(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):
        super(RUnet_encoder, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.LeakyReLU(0.2, inplace=True)
        act1 = nn.Sigmoid()

        # Down sampling
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

        # self.down_5 = conv_block_2_3d(self.num_filters * 8, self.num_filters * 16, activation)
        # self.res_5 = conv_block_3d(self.num_filters * 8, self.num_filters * 16, activation)
        # self.pool_5 = max_pooling_3d()

        # Bridge
        self.bridge = conv_block_2_3d(self.num_filters * 8, self.num_filters * 16, activation)
        self.res_bridge = conv_block_3d(self.num_filters * 8, self.num_filters * 16, activation)


    def forward(self, x):
        # Down sampling
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

        # down_5 = self.down_5(pool_4)  # -> [1, 64, 8, 8, 8]
        # res_5 = self.res_5(pool_4)
        # pool_5 = self.pool_5(down_5 + res_5)  # -> [1, 64, 4, 4, 4]

        # Bridge
        bridge = self.bridge(pool_4)  # -> [1, 128, 4, 4, 4]
        res_bridge = self.res_bridge(pool_4)+bridge


        return res_bridge.view(x.shape[0], -1)

def D(p, z, version='simplified'):  # negative cosine similarity
    if version == 'original':
        z = z.detach()  # stop gradient
        p = F.normalize(p, dim=1)  # l2-normalize
        z = F.normalize(z, dim=1)  # l2-normalize
        return -(p * z).sum(dim=1).mean()

    elif version == 'simplified':  # same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception

class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=256*16*4, out_dim=256*8):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.num_layers = 3

    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x

class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):  # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class SimSiam(nn.Module):
    def __init__(self, in_dim,out_dim,num_filters):
        super().__init__()

        self.backbone = RUnet_encoder(in_dim,out_dim,num_filters)
        self.projector = projection_MLP(128*16*16)

        self.encoder = nn.Sequential(  # f encoder
            self.backbone,
            self.projector
        )
        self.predictor = prediction_MLP()

    def forward(self, real_pos, fake_pos, fake_neg):
        f, h = self.encoder, self.predictor
        z1, z2, z3 = f(real_pos), f(fake_pos), f(fake_neg)
        p1, p2, p3 = h(z1), h(z2), h(z3)

        L_pos = D(p1, z2) / 2 + D(p2, z1) / 2
        L_neg = -1*(D(p1, z3) / 2 + D(p3, z1) / 2)
        return L_pos+L_neg
