import torch
import numpy as np
import torch.optim as optim
import SimpleITK as sitk
from options.Options import Options_x
from dataset.dataset_lits_train import Lits_DataSet
from dataset.dataset_lits_test import Test_Datasets
from Model.runet import RUNet
from torch.utils.data import DataLoader
from utils.common import adjust_learning_rate
from utils import logger,util
import torch.nn as nn
from utils.metrics import LossAverage, DiceLoss, seg_metric
import os
from test import test_all
from collections import OrderedDict


def train (train_dataloader,epoch):
    print("=======Epoch:{}======Learning_rate:{}=========".format(epoch,optimizer.param_groups[0]['lr']))

    Loss = LossAverage()
    DICE_Loss = LossAverage()
    BCE_Loss = LossAverage()
    # DSC_metric = LossAverage()
    # PPV_metric = LossAverage()
    # SEN_metric = LossAverage()
    # ASD_metric = LossAverage()
    #HFD_metric = LossAverage()
    #RAVD_metric = LossAverage()
    model.train()

    for i, (pre,pos,sub,gt) in enumerate(train_dataloader):  # inner loop within one epoch
        ##main model update param
        b,c,l,w,e = pre.shape[0],pre.shape[1],pre.shape[2],pre.shape[3],pre.shape[4],
        pre = pre.view(-1,1,l,w,e).to(device)
        pos = pos.view(-1,1,l,w,e).to(device)
        sub = sub.view(-1,1,l,w,e).to(device)
        gt = gt.view(-1,1,l,w,e).to(device)
        #print(pre.shape)
        pred = model(torch.cat([pos,sub],1))  #pos sub
        #print(pred.dtype,gt.dtype)

        Dice_loss = dice_loss(pred, gt)
        Bce_loss = bce_loss(pred, gt)
        loss= Bce_loss + 5 * Dice_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        adjust_learning_rate(optimizer,epoch,opt)

        Loss.update(loss.item(),pre.size(0))
        DICE_Loss.update(Dice_loss.item(),pre.size(0))
        BCE_Loss.update(Bce_loss.item(), pre.size(0))
        # DSC_metric.update(DSC,pre.size(0))
        # PPV_metric.update(PPV,pre.size(0))
        # SEN_metric.update(SEN,pre.size(0))
        # ASD_metric.update(ASD, pre.size(0))
        #HFD_metric.update(HFD, pre.size(0))
        #RAVD_metric.update(RAVD, pre.size(0))
        
    return OrderedDict({'Loss': Loss.avg,'DICE_Loss':DICE_Loss.avg,'BCE_Loss':BCE_Loss.avg})
                        #'DSC': DSC_metric.avg,'PPV': PPV_metric.avg,'SEN': SEN_metric.avg,'ASD': ASD_metric.avg})

def test (img_dataset):

    model.eval() 

    with torch.no_grad():
        pre,pos,sub,gt= img_dataset[0].type(torch.float32),img_dataset[1].type(torch.float32),img_dataset[2].type(torch.float32),img_dataset[3].type(torch.float32)
        pre,pos,sub,gt = pre.to(device),pos.cuda(),sub.to(device),gt.to(device)
        #print(pre.shape,pos.shape,sub.shape,gt.shape)
        pred = model(torch.cat([pos,sub],1))

        Pred = pred.cpu().detach().numpy().squeeze()
        GT = gt.cpu().detach().numpy().squeeze()
        Pos_image = pos.cpu().detach().numpy().squeeze()

        DSC, PPV, SEN, ASD = seg_metric(Pred,GT)
        Pred = (Pred>0.5).astype(np.uint16)   
    return Pos_image,Pred,GT,OrderedDict({'DSC': DSC,'PPV': PPV,'SEN': SEN,
                        'ASD': ASD})


    
if __name__ == '__main__':
    opt = Options_x().parse()   # get training options
    device = torch.device('cuda:'+opt.gpu_ids if torch.cuda.is_available() else "cpu")

    model = RUNet(2,1,8).to(device)
    #model = nn.DataParallel(model)
    #model = model.cuda()
    #ckpt = torch.load(opt.checkpoints_dir + '/' + opt.task_name + '/model/' + 'latest_model.pth', map_location=device)
    #model.load_state_dict(ckpt['model'])
    save_path = opt.checkpoints_dir
    dice_loss = DiceLoss()
    bce_loss = torch.nn.BCELoss()
        
    save_result_path = os.path.join(save_path,opt.task_name)
    util.mkdir(save_result_path)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr,weight_decay=1e-5) #optim.Adam(model.parameters(), lr=opt.lr)

    model_save_path = os.path.join(save_result_path,'model')
    util.mkdir(model_save_path)
    logger_save_path = os.path.join(save_result_path,'logger')
    util.mkdir(logger_save_path)
    log_train = logger.Train_Logger(logger_save_path,"train_log")

    train_dataset = Lits_DataSet(opt.datapath, 'partial', opt.patch_size, opt.data_folder)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, \
                                  num_workers=opt.num_threads, shuffle=True)

    for epoch in range(opt.epoch):
        epoch = epoch +1
        train_log= train (train_dataloader,epoch)
        
        log_train.update(epoch,train_log)
        
        state = {'model': model.state_dict(), 'epoch': epoch}
        torch.save(state, os.path.join(model_save_path, 'latest_model.pth'))
        
        
        if epoch%opt.model_save_fre ==0:
            torch.save(state, os.path.join(model_save_path, 'model_'+np.str(epoch)+'.pth'))

        torch.cuda.empty_cache() 
        
        if epoch%opt.test_fre ==0:
            
            test_dataset = Test_Datasets(opt.datapath,opt.patch_size, opt.data_folder)
            for test_loader, file in test_dataset:

                image,pred,gt,test_log = test(test_loader)
                log_test.update(file,test_log)
                pred = sitk.GetImageFromArray(np.array(pred))
                gt = sitk.GetImageFromArray(np.array(gt))
                image = sitk.GetImageFromArray(np.array(image))
                util.mkdir(os.path.join(result_save_path,file))
                sitk.WriteImage(pred,os.path.join(result_save_path,file,'pred.nii.gz'))
                sitk.WriteImage(gt,os.path.join(result_save_path,file,'gt.nii.gz'))
                sitk.WriteImage(image,os.path.join(result_save_path,file,'pos_image.nii.gz'))

                #print(file +' has finished!')
    test_all('latest_model.pth')

 
        

            
            

            
            
            
            
            
            
