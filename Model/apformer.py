from .unets_parts import *
from .transformer_partsR import TransformerDown, TransformerDown_HP, TransformerDown_SPrune, TransformerDown_SPrune_Test

class APFormer(nn.Module):
    def __init__(self, down_block, n_channels,out_channels, n_classes, imgsize, bilinear=True):
        super(APFormer, self).__init__()
        self.n_channels = n_channels
        self.out_channels=out_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.scale = 4 # 1 2 4

        self.inc = DoubleConv(n_channels, 64//self.scale)
        self.down1 = Down(64//self.scale, 128//self.scale)
        self.down2 = Down(128 // self.scale, 256 // self.scale)
        self.down3 = Down(256 // self.scale, 512 // self.scale)
        factor = 2 if bilinear else 1

        #self.trans4 = down_block(512//self.scale, 512//self.scale, imgsize//8, 2, heads=6, dim_head=64, patch_size=1)  # 256, 1024
        #self.conv4 = nn.Conv3d(512//self.scale, 512//self.scale//factor, kernel_size=1, padding=0, bias=False)
        #self.conv4 = nn.Conv3d(512 // self.scale, 512 // self.scale , kernel_size=1, padding=0, bias=False)
        self.conv4 = nn.Conv3d(512 // self.scale, 512 // self.scale, kernel_size=1, padding=0, bias=False)

        self.up2 = Up(512 // self.scale, 256  // self.scale, bilinear)
        self.up3 = Up(256 // self.scale, 128 // self.scale, bilinear)
        self.up4 = Up(128 // self.scale, 64 // self.scale, bilinear)
        self.outc = OutConv(64 // self.scale, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x4 = self.down3(x3)

        #x4, qkvs1, attns1 = self.trans4(x4)
        x4 = self.conv4(x4)
        x = self.up2(x4, x3)

        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        #return logits, qkvs1, attns1
        act1 =nn.Sigmoid()
        logits=act1(logits)
        return logits

def APFormer_Model(**kwargs):
    model = APFormer(TransformerDown_SPrune, **kwargs)
    return model

def APFormer_Model_Test(**kwargs):
    model = APFormer(TransformerDown_SPrune_Test, **kwargs)
    return model