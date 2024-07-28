from torch import nn
import torch
import torch.nn.functional as F
from torch.nn import init
from Model.affrefine3D import affrefine3D
#####To design a module for affinity leanring and refinement.

class ConvGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, all_dim, kernel_size):
        super(ConvGRUCell,self).__init__()
        self.input_size  = input_size
        self.cuda_flag   = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.padding = int((kernel_size - 1) / 2)
        self.reset_gate = nn.Conv3d(input_size + hidden_size, hidden_size, kernel_size, padding=self.padding)
        self.update_gate = nn.Conv3d(input_size + hidden_size, hidden_size, kernel_size, padding=self.padding)
        self.out_gate = nn.Conv3d(input_size + hidden_size, hidden_size, kernel_size, padding=self.padding)


    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if torch.cuda.is_available():
                prev_state = Variable(torch.zeros(state_size)).cuda()
            else:
                prev_state = Variable(torch.zeros(state_size))

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))

        new_state = prev_state * (1 - update) + torch.sigmoid(out_inputs) * update

        #print(new_state)
        #print(new_state.size())
        return new_state

class ConvGRUCellfeaturefusion(nn.Module):
    
    def __init__(self, input_size, hidden_size, all_dim, kernel_size):
        super(ConvGRUCellfeaturefusion,self).__init__()
        self.input_size  = input_size
        self.cuda_flag   = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.padding = int((kernel_size - 1) / 2)
        #self.reset_gate = nn.Conv3d(input_size , hidden_size, kernel_size, padding=self.padding)
        self.update_gate = nn.Conv3d(input_size+hidden_size, hidden_size, kernel_size, padding=self.padding)
        #self.out_gate = nn.Conv3d(input_size, hidden_size, kernel_size, padding=self.padding)


    def forward(self, input_, prev_state):
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        
        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if torch.cuda.is_available():
                prev_state = Variable(torch.zeros(state_size)).cuda()
            else:
                prev_state = Variable(torch.zeros(state_size))

        # data size is [batch, channel, height, width]
        stacked_inputs =torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        #reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        #out_inputs = torch.tanh(self.out_gate(input_+prev_state * reset)) #torch.tanh
        new_state = prev_state * (1 - update) + input_ * update ##To fuse the commo infomration.
        #print(new_state)
        #print(new_state.size())
        return new_state

class IRUCell(nn.Module):
    
    def __init__(self, input_size, hidden_size, all_dim, kernel_size):
        super(IRUCell,self).__init__()
        self.input_size  = input_size
        self.cuda_flag   = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.padding = int((kernel_size - 1) / 2)
        self.reset_gate = nn.Conv3d(input_size, hidden_size, kernel_size, padding=self.padding)
        self.update_gate = nn.Conv3d(input_size, hidden_size, kernel_size, padding=self.padding)
        self.out_gate = nn.Conv3d(input_size+hidden_size, hidden_size, kernel_size, padding=self.padding)


    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if torch.cuda.is_available():
                prev_state = Variable(torch.zeros(state_size)).cuda()
            else:
                prev_state = Variable(torch.zeros(state_size))

        # data size is [batch, channel, height, width]
        stacked_inputs =input_+prev_state+input_*prev_state #torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1))) ##how many information can be obtained

        new_state = prev_state * (1 - update) + out_inputs * update

        #print(new_state)
        #print(new_state.size())
        return new_state


class Donwconv(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(Donwconv, self).__init__()
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=2, stride=2,padding=1, bias=False))

        self.add_module('pool_norm', nn.BatchNorm3d(num_output_features))
        self.add_module('pool_relu', nn.ReLU(inplace=True))
        self.add_module('pool', nn.Conv3d(num_output_features, num_output_features//2, kernel_size=1, stride=1))


class DonwconvT(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(DonwconvT, self).__init__()
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=2, stride=2,padding=1, bias=False))

        self.add_module('pool_norm', nn.BatchNorm3d(num_output_features))
        self.add_module('pool_relu', nn.ReLU(inplace=True))
        self.add_module('pool', nn.Conv3d(num_output_features, num_output_features, kernel_size=1, stride=1))



class refinementnetwork_ffSPNv3(nn.Module):
    def __init__(self,in_features,out_features):
        super(refinementnetwork_ffSPNv3, self).__init__()

        self.Gru3dff=ConvGRUCellfeaturefusion(36,32,1,3) ##feature level fusion.

        #self.seg1 = nn.Conv3d(66, 1, kernel_size=1, padding=0, stride=1, bias=False)

        self.down_block=Transition(in_features,out_features) ##

        self.up_block = nn.ConvTranspose3d(out_features, out_features, kernel_size=2 ,
                                          stride=2 ,
                                          padding=1, groups=1, bias=False)


        PAMR_KERNEL = [1] ## 3*3 neighbou10ing
        PAMR_ITER = 10    ## iter number
        
        self._aff = affrefine3D(PAMR_ITER, PAMR_KERNEL)


    def forward(self, fea1,fea2,img,mask):
        #print(fea2.size())
        #print(fea1.size())
        x=self.Gru3dff(fea1,fea2) ## Learning based Fusion
        x1=self.down_block(torch.cat([fea1,img,x],1))
        x2=self.up_block(x1) ##TO learn the features for affinity.
        #x2=x
        res=self._aff(x2,mask)
        res=torch.where(res>1,torch.ones_like(res),res)
        #res=torch.where(res>1,tensor.ones(1),res) ##to remove the value bigger than 1.

        return res