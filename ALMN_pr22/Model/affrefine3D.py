import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import nn
from functools import partial

#
# written by lei zhou, used to
#
class LocalAffinity(nn.Module):

    def __init__(self, dilations=[1]):
        super(LocalAffinity, self).__init__()
        self.dilations = dilations
        weight = self._init_aff()
        self.register_buffer('kernel', weight) ## 在内存中定一个常量，同时，模型保存和加载的时候可以写入和读出

    def _init_aff(self):
        # initialising the shift kernel
        weight = torch.zeros(26, 1, 3, 3, 3) ##一个 3*3 的 kernel

        for i in range(weight.size(0)):
            weight[i, 0,1, 1, 1] = 1

        weight[0, 0, 0,0, 0] = -1 ##把相应的值取出。
        weight[1, 0, 0,0, 1] = -1
        weight[2, 0, 0,0, 2] = -1

        weight[3, 0, 0,1, 0] = -1
        weight[4, 0, 0,1, 1] = -1
        weight[5, 0, 0,1, 2] = -1

        weight[6, 0, 0,2, 0] = -1
        weight[7, 0, 0,2, 1] = -1
        weight[8, 0, 0,2, 2] = -1  ##取一个出来

        weight[9,  0, 1, 0, 0] = -1 ##把相应的值取出。
        weight[10, 0, 1, 0, 1]= -1
        weight[11, 0, 1, 0, 2]= -1

        weight[12, 0, 1, 1, 0] = -1
        weight[13, 0, 1, 1, 2] = -1

        weight[14, 0, 1, 2, 0] = -1
        weight[15, 0, 1, 2, 1] = -1
        weight[16, 0, 1, 2, 2] = -1  ##取一个出来

        weight[17, 0, 2, 0, 0] = -1  ##把相应的值取出。
        weight[18, 0, 2, 0, 1] = -1
        weight[19, 0, 2, 0, 2] = -1

        weight[20, 0, 2, 1, 0] = -1
        weight[21, 0, 2, 1, 1] = -1
        weight[22, 0, 2, 1, 2] = -1

        weight[23, 0, 2, 2, 0] = -1
        weight[24, 0, 2, 2, 1] = -1
        weight[25, 0, 2, 2, 2] = -1  ##取一个出来

        self.weight_check = weight.clone()

        return weight

    def forward(self, x):
        
        self.weight_check = self.weight_check.type_as(x)
        assert torch.all(self.weight_check.eq(self.kernel))

        B,K,C1,H,W = x.size()
        x = x.view(B*K,1,C1,H,W) ##输出特征
        #print('ok')

        x_affs = []
        for d in self.dilations:
            x_pad = F.pad(x, [d]*6, mode='replicate')
            x_aff = F.conv3d(x_pad, self.kernel, dilation=d)
            x_affs.append(x_aff)

        x_aff = torch.cat(x_affs, 1)
        out=x_aff.view(B,K,-1,C1,H,W)
        
        return out

class LocalAffinityCopy(LocalAffinity):

    def _init_aff(self):
        # initialising the shift kernel
        weight = torch.zeros(26, 1, 3, 3, 3)

        weight[0, 0, 0, 0, 0] = 1  ##把相应的值取出。
        weight[1, 0, 0, 0, 1] = 1
        weight[2, 0, 0, 0, 2] = 1

        weight[3, 0, 0, 1, 0] = 1
        weight[4, 0, 0, 1, 1] = 1
        weight[5, 0, 0, 1, 2] = 1

        weight[6, 0, 0, 2, 0] = 1
        weight[7, 0, 0, 2, 1] = 1
        weight[8, 0, 0, 2, 2] = 1  ##取一个出来

        weight[9, 0, 1, 0, 0] = 1  ##把相应的值取出。
        weight[10, 0, 1, 0, 1] = 1
        weight[11, 0, 1, 0, 2] = 1

        weight[12, 0, 1, 1, 0] = 1
        weight[13, 0, 1, 1, 2] = 1

        weight[14, 0, 1, 2, 0] = 1
        weight[15, 0, 1, 2, 1] = 1
        weight[16, 0, 1, 2, 2] = 1  ##取一个出来

        weight[17, 0, 2, 0, 0] = 1  ##把相应的值取出。
        weight[18, 0, 2, 0, 1] = 1
        weight[19, 0, 2, 0, 2] = 1

        weight[20, 0, 2, 1, 0] = 1
        weight[21, 0, 2, 1, 1] = 1
        weight[22, 0, 2, 1, 2] = 1

        weight[23, 0, 2, 2, 0] = 1
        weight[24, 0, 2, 2, 1] = 1
        weight[25, 0, 2, 2, 2] = 1  ##取一个出来

        self.weight_check = weight.clone()
        return weight

class LocalStDev(LocalAffinity):

    def _init_aff(self):
        weight = torch.zeros(27, 1, 3, 3, 3)
        weight.zero_()

        weight[0, 0, 0, 0, 0] = 1  ##把相应的值取出。
        weight[1, 0, 0, 0, 1] = 1
        weight[2, 0, 0, 0, 2] = 1

        weight[3, 0, 0, 1, 0] = 1
        weight[4, 0, 0, 1, 1] = 1
        weight[5, 0, 0, 1, 2] = 1

        weight[6, 0, 0, 2, 0] = 1
        weight[7, 0, 0, 2, 1] = 1
        weight[8, 0, 0, 2, 2] = 1  ##取一个出来

        weight[9, 0, 1, 0, 0] = 1  ##把相应的值取出。
        weight[10, 0, 1, 0, 1] = 1
        weight[11, 0, 1, 0, 2] = 1

        weight[12, 0, 1, 1, 0] = 1
        weight[13, 0, 1, 1, 1] = 1
        weight[14, 0, 1, 1, 2] = 1

        weight[15, 0, 1, 2, 0] = 1
        weight[16, 0, 1, 2, 1] = 1
        weight[17, 0, 1, 2, 2] = 1  ##取一个出来

        weight[18, 0, 2, 0, 0] = 1  ##把相应的值取出。
        weight[19, 0, 2, 0, 1] = 1
        weight[20, 0, 2, 0, 2] = 1

        weight[21, 0, 2, 1, 0] = 1
        weight[22, 0, 2, 1, 1] = 1
        weight[23, 0, 2, 1, 2] = 1

        weight[24, 0, 2, 2, 0] = 1
        weight[25, 0, 2, 2, 1] = 1
        weight[26, 0, 2, 2, 2] = 1  ##取一个出来

        self.weight_check = weight.clone()
        return weight

    def forward(self, x):
        # returns (B,K,P,H,W), where P is the number
        # of locations
        x = super(LocalStDev, self).forward(x)

        return x.std(2, keepdim=True)

class LocalAffinityAbs(LocalAffinity):

    def forward(self, x):
        x = super(LocalAffinityAbs, self).forward(x)
        return torch.abs(x)

#
# PAMR module
#
class affrefine3D(nn.Module):

    def __init__(self, num_iter=1, dilations=[1]):
        super(affrefine3D, self).__init__()

        self.num_iter = num_iter
        self.aff_x = LocalAffinityAbs(dilations)
        self.aff_m = LocalAffinityCopy(dilations)
        self.aff_std = LocalStDev(dilations)

    def forward(self, x, mask,aff_in,sign):
        #mask = F.interpolate(mask, size=x.size()[-2:], mode="bilinear", align_corners=True) ## 分割的mask

        # x: [BxKxHxW]
        # mask: [BxCxHxW]
        B,K,C1,H,W = x.size()
        _,C,_,_,_ = mask.size()

        x_std = self.aff_std(x) #[BxKx1xHxW]
        #print(x_std.size())

        x = -self.aff_x(x) #[BxKx48xHxW]
        s1=(1e-8 + 0.1 * x_std)
        x=x/s1
        x = x.mean(1, keepdim=True) ##在所有特征通道上求取了平均值。

        x = F.softmax(x, 2) #[BxKx48xHxW] 这里的乘法进行的是一种approxining操作

        if sign==1:
            x=x*0.5+aff_in*0.5
        

        for _ in range(self.num_iter):
            m = self.aff_m(mask)  # [BxCxPxHxW] ##只是进行了复制操作
            #print(m.size())
            #mask=(m * x).sum(2)
            mask = torch.max(mask,(m * x).sum(2))  ## mask 被迭代了多次； ##the max operation can be tested.

        # xvals: [BxCxC1*HxW]
        return mask,x

class affrefine3DONLY(nn.Module):

    def __init__(self, num_iter=1, dilations=[1]):
        super(affrefine3DONLY, self).__init__()

        self.num_iter = num_iter
        self.aff_x = LocalAffinityAbs(dilations)
        self.aff_m = LocalAffinityCopy(dilations)
        self.aff_std = LocalStDev(dilations)

    def forward(self, x):
        #mask = F.interpolate(mask, size=x.size()[-2:], mode="bilinear", align_corners=True) ## 分割的mask

        # x: [BxKxHxW]
        # mask: [BxCxHxW]
        B,K,C1,H,W = x.size()
        #_,C,_,_,_ = mask.size()

        x_std = self.aff_std(x) #[BxKx1xHxW]
        #print(x_std.size())

        x = -self.aff_x(x) #[BxKx48xHxW]
        s1=(1e-8 + 0.1 * x_std)
        x=x/s1
        x = x.mean(1, keepdim=True) ##在所有特征通道上求取了平均值。

        x = F.softmax(x, 2) #[BxKx48xHxW] 这里的乘法进行的是一种approxining操作
        
        # xvals: [BxCxC1*HxW]
        return x


class affrefine3DSEG(nn.Module):

    def __init__(self, num_iter=1, dilations=[1]):
        super(affrefine3DSEG, self).__init__()

        self.num_iter = num_iter
        self.aff_x = LocalAffinityAbs(dilations)
        self.aff_m = LocalAffinityCopy(dilations)
        self.aff_std = LocalStDev(dilations)

    def forward(self, x, mask):
        #mask = F.interpolate(mask, size=x.size()[-2:], mode="bilinear", align_corners=True) ## 分割的mask

        for _ in range(self.num_iter):
            m = self.aff_m(mask)  # [BxCxPxHxW] ##只是进行了复制操作
            #print(m.size())
            #print(x.size())
            #mask=(m * x).sum(2)
            mask = torch.max(mask,(m * x).sum(2))  ## mask 被迭代了多次； ##the max operation can be tested.

        # xvals: [BxCxC1*HxW]
        return mask