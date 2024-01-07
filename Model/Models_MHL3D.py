"""
This code was write by Dr. Jun Zhang. If you use this code please follow the licence of Attribution-NonCommercial-ShareAlike 4.0 International. 

"""

import torch 
import torch.nn as nn

def center_crop(layer, n_size):
    cropidx = (layer.size(2) - n_size) // 2
    return layer[:, :, cropidx:(cropidx + n_size), cropidx:(cropidx + n_size),cropidx:(cropidx + n_size)]


class ModelTumor(nn.Module):
    def __init__(self, in_channel, n_classes):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = 32
        super(ModelTumor, self).__init__()
        self.eninput = self.encoder(self.in_channel, self.start_channel, bias=False)
        self.ec1 = self.encoder(self.start_channel, self.start_channel, bias=False)
        self.ec2 = self.encoder(self.start_channel, self.start_channel*2, bias=False)
        self.ec3 = self.encoder(self.start_channel*2, self.start_channel*2, bias=False)
        self.ec4 = self.encoder(self.start_channel*2, self.start_channel*4, bias=False)
        self.ec5 = self.encoder(self.start_channel*4, self.start_channel*2, bias=False)
        self.pool = nn.MaxPool3d(2)
        self.dc1 = self.encoder(self.start_channel*2+self.start_channel*2, self.start_channel*4, kernel_size=3, stride=1, bias=False)
        self.dc2 = self.encoder(self.start_channel*4, self.start_channel*2, kernel_size=3, stride=1, bias=False)
        self.dc3 = self.encoder(self.start_channel*2+self.start_channel*1, self.start_channel*2, kernel_size=3, stride=1, bias=False)
        self.dc4 = self.encoder(self.start_channel*2, self.start_channel*2, kernel_size=3, stride=1, bias=False)
        self.dc5 = self.outputs(self.start_channel*2, self.n_classes, kernel_size=1, stride=1,padding=0, bias=False)

        self.up1 = self.decoder(self.start_channel*2, self.start_channel*2)
        self.up2 = self.decoder(self.start_channel*2, self.start_channel*2)

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=True):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU())
        return layer


    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer


    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=True):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Sigmoid())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Sigmoid())
        return layer

    def forward(self, x):
        e0 = self.eninput(x)
        e0 = self.ec1(e0)
        
        e1 = self.pool(e0)
        e1 = self.ec2(e1)
        e1 = self.ec3(e1)
        e2 = self.pool(e1)
        e2 = self.ec4(e2)
        e2 = self.ec5(e2)
        ###############3
        d0 = torch.cat((self.up1(e2), center_crop(e1,e2.size(2)*2)), 1)
        d0 = self.dc1(d0)
        d0 = self.dc2(d0)
        d1 = torch.cat((self.up2(d0), center_crop(e0,d0.size(2)*2)), 1)
        d1 = self.dc3(d1)
        d1 = self.dc4(d1)
        d1 = self.dc5(d1)
        
        return d1    
    
    
    
# Note we trained the model with the same size (96*96*96) of input and output. 
# We used zero padding to guarantee the same size of output after filtering
    
class ModelTumor_train(nn.Module):
    def __init__(self, in_channel, n_classes):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = 32
        
        super(ModelTumor_train, self).__init__()
        self.eninput = self.encoder(self.in_channel, self.start_channel, bias=False)
        self.ec1 = self.encoder(self.start_channel, self.start_channel, bias=False)
        self.ec2 = self.encoder(self.start_channel, self.start_channel*2, bias=False)
        self.ec3 = self.encoder(self.start_channel*2, self.start_channel*2, bias=False)
        self.ec4 = self.encoder(self.start_channel*2, self.start_channel*4, bias=False)
        self.ec5 = self.encoder(self.start_channel*4, self.start_channel*2, bias=False)


        self.pool = nn.MaxPool3d(2)
        

        self.dc1 = self.encoder(self.start_channel*2+self.start_channel*2, self.start_channel*4, kernel_size=3, stride=1, bias=False)
        self.dc2 = self.encoder(self.start_channel*4, self.start_channel*2, kernel_size=3, stride=1, bias=False)
        self.dc3 = self.encoder(self.start_channel*2+self.start_channel*1, self.start_channel*2, kernel_size=3, stride=1, bias=False)
        self.dc4 = self.encoder(self.start_channel*2, self.start_channel*2, kernel_size=3, stride=1, bias=False)
        self.dc5 = self.outputs(self.start_channel*2, self.n_classes, kernel_size=1, stride=1,padding=0, bias=False)

        self.up1 = self.decoder(self.start_channel*2, self.start_channel*2)
        self.up2 = self.decoder(self.start_channel*2, self.start_channel*2)

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, ##padding=0?
                bias=False, batchnorm=True):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU())
        return layer


    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer


    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=True):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Sigmoid())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Sigmoid())
        return layer

    def forward(self, x):
        e0 = self.eninput(x)
        e0 = self.ec1(e0)
        
        e1 = self.pool(e0)
        e1 = self.ec2(e1)
        e1 = self.ec3(e1)

        e2 = self.pool(e1)
        e2 = self.ec4(e2)
        e2 = self.ec5(e2)

        d0 = torch.cat((self.up1(e2), e1), 1)
        d0 = self.dc1(d0)
        d0 = self.dc2(d0)

        d1 = torch.cat((self.up2(d0), e0), 1)

        d1 = self.dc3(d1)
        d1 = self.dc4(d1)
        d1 = self.dc5(d1)
        
        return d1    
    

    
    
def Dice_loss(input, target):
    smooth = 0.00000001

    y_true_f = input.view(-1)
    y_pred_f = target.view(-1)
    intersection = torch.sum(torch.mul(y_true_f,y_pred_f))
    
    return 1 - ((2. * intersection ) /
              (torch.mul(y_true_f,y_true_f).sum() + torch.mul(y_pred_f,y_pred_f).sum() + smooth))

def DICESEN_loss(input, target):
    smooth = 0.00000001
    y_true_f = input.view(-1)
    y_pred_f = target.view(-1)
    intersection = torch.sum(torch.mul(y_true_f,y_pred_f))
    dice= (2. * intersection ) / (torch.mul(y_true_f,y_true_f).sum() + torch.mul(y_pred_f,y_pred_f).sum() + smooth)
    sen = (1. * intersection ) / (torch.mul(y_true_f,y_true_f).sum() + smooth)
    return 2-dice-sen   