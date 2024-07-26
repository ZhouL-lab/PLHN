import torch
import numpy as np
import torch.optim as optim
import SimpleITK as sitk
from options.Options import Options_x
from dataset.dataset_lits_train import Lits_DataSet
from dataset.dataset_lits_test import Test_Datasets
from Model.TokenSegV8_prototype_fusion_attentions import TokenSegV8
from modelv5.loss_proto import PixelPrototypeCELoss,PPD,PPC
from torch.utils.data import DataLoader
from utils.common import adjust_learning_rate
from utils import logger,util
import torch.nn as nn
from utils.metrics import LossAverage, DiceLoss, seg_metric
import os
from test import test_all
from collections import OrderedDict
from PrototypeAndContrast import Prototype_Consistency_Constrast_loss
import time

def train (train_dataloader,epoch):
    print("=======Epoch:{}======Learning_rate:{}=========".format(epoch,optimizer.param_groups[0]['lr']))
    PPCELOSS = PixelPrototypeCELoss()

    Loss = LossAverage()
    DICE_Loss = LossAverage()
    DICE_Loss_1 = LossAverage()
    DICE_Loss_2 = LossAverage()
    BCE_Loss = LossAverage()
    BCE_Loss_1 = LossAverage()
    BCE_Loss_2 = LossAverage()
    CON_Loss = LossAverage()

    model.train()

    for i, (pre,pos,sub,gt) in enumerate(train_dataloader):  # inner loop within one epoch
        ##main model update param
        b,c,l,w,e = pre.shape[0],pre.shape[1],pre.shape[2],pre.shape[3],pre.shape[4],
        pre = pre.view(-1,1,l,w,e).to(device)
        pos = pos.view(-1,1,l,w,e).to(device)
        sub = sub.view(-1,1,l,w,e).to(device)
        gt = gt.view(-1,1,l,w,e).to(device)

        #t1=time.time()
        pred_pro,pro_seg,pred,fusion_pred = model(torch.cat([pos, sub], 1),gt)  # pos sub
        #print(time.time() - t1)

        loss_con=PPCELOSS(pred_pro, gt) ##相关对比学习的LOSS

        output_fus =fusion_pred #0.5*pro_seg+0.5*pred #0.6*pred+0.4*pro_seg1 #(pro_seg * pre) / k

        Dice_loss = dice_loss(pred, gt) ##
        Dice_loss_1 = dice_loss(pro_seg, gt)
        Dice_loss_2 = dice_loss(output_fus, gt)

        Bce_loss = bce_loss(pred, gt)
        Bce_loss_1 = bce_loss(pro_seg, gt)
        Bce_loss_2 = bce_loss(output_fus, gt)

        loss = Bce_loss+ 8* (Dice_loss)+Bce_loss_1+1.5* (Dice_loss_1)+Bce_loss_2+1* (Dice_loss_2)+0.3*loss_con  ####4->2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        adjust_learning_rate(optimizer,epoch,opt)

        Loss.update(loss.item(),pre.size(0))
        DICE_Loss.update(Dice_loss.item(),pre.size(0))
        DICE_Loss_1.update(Dice_loss_1.item(), pre.size(0))
        DICE_Loss_2.update(Dice_loss_2.item(), pre.size(0))
        BCE_Loss.update(Bce_loss.item(), pre.size(0))
        BCE_Loss_1.update(Bce_loss_1.item(), pre.size(0))
        BCE_Loss_2.update(Bce_loss_2.item(), pre.size(0))
        CON_Loss.update(loss_con.item(), pre.size(0))
        
    return OrderedDict({'Loss': Loss.avg,'DICE_Loss':DICE_Loss.avg,'DICE_Loss_1':DICE_Loss_1.avg,'DICE_Loss_2':DICE_Loss_2.avg,'BCE_Loss':BCE_Loss.avg,'BCE_Loss_1':BCE_Loss_1.avg,'CON_Loss':CON_Loss.avg})
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

    model = TokenSegV8(inch=2, TransformerLayerNum=8, hidden_size=256, base_channeel=32,imgsize=[48, 128, 128]).to(device)
    path1='/data/NET6_transformer/checkpoints_n2/transformer/TokenSegx8m/TokenSegx8m_78.9.pth'

    ckpt = torch.load(path1,map_location=device)
    model.load_state_dict(ckpt['model'], strict=False)

    save_path = opt.checkpoints_dir
    dice_loss = DiceLoss()
    bce_loss = torch.nn.BCELoss()
        
    save_result_path = os.path.join(save_path,opt.task_name)
    util.mkdir(save_result_path)

    for name, paramer in model.named_parameters():
        if paramer.requires_grad:
            print(name)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr,weight_decay=1e-5) #optim.Adam(model.parameters(), lr=opt.lr)

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

        t1=time.time()
        train_log= train (train_dataloader,epoch)
        print(time.time() - t1)
        
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


            
            
            
            
