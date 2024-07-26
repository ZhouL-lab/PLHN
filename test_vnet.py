import torch
import gc
import numpy as np
import torch.optim as optim
import SimpleITK as sitk
from options.Options import Options_x
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import logger,util
from utils.metrics import seg_metric
import torch.nn as nn
import os
from dataset.dataset_lits_test import Test_all_Datasets,Recompone_tool
from collections import OrderedDict
from vnet_sdf import VNet

def load(file):
    itkimage = sitk.ReadImage(file)
    image = sitk.GetArrayFromImage(itkimage)
    return image

def test_all(model_name='model_200.pth'):
    opt = Options_x().parse()  # get training options
    device = torch.device('cuda:' + opt.gpu_ids if torch.cuda.is_available() else "cpu")

    #model = RUNet(2, 1, 16).to(device)
    model = VNet(n_channels=2, n_classes=1, normalization='batchnorm').to(device)

    ckpt = torch.load(opt.checkpoints_dir + '/' + opt.task_name + '/model/' + model_name,map_location=device)
    model.load_state_dict(ckpt['model'])
    #weightspaf=opt.checkpoints_dir + '/' + opt.task_name + '/model/' + model_name
    #model.load_state_dict(torch.load(weightspaf,map_location=device) ,strict=False) #'checkpointsfixed/pretrain77.9.pth'

    save_result_path = os.path.join(opt.checkpoints_dir, opt.task_name, 'localization')
    util.mkdir(save_result_path)
    model.eval()
    log_test = logger.Test_Logger(save_result_path,"results")
    cut_param = {'patch_s': opt.patch_size[0], 'patch_h': opt.patch_size[1], 'patch_w': opt.patch_size[2],
                 'stride_s': opt.patch_stride[0], 'stride_h': opt.patch_stride[1], 'stride_w': opt.patch_stride[2]}
    datasets = Test_all_Datasets(opt.datapath, cut_param, opt.data_folder)

    for img_dataset, original_shape, new_shape,mask,file_idx in datasets:
        save_tool = Recompone_tool(original_shape, new_shape, cut_param)
        dataloader = DataLoader(img_dataset, batch_size=opt.test_batch, num_workers=opt.num_threads, shuffle=False)
        with torch.no_grad():
            for pre, pos, sub, gt in tqdm(dataloader):
                pre, pos, sub, gt = pre.to(device), pos.to(device), sub.to(device), gt.to(device)
		
                #pre = pre.unsqueeze(1).type(torch.float32)
                pos = pos.unsqueeze(1).type(torch.float32)
                #print(pos.shape)
                sub = sub.unsqueeze(1).type(torch.float32)
                #gt = gt.unsqueeze(1).type(torch.float32)
                #print(torch.cat([pos, sub], 1).shape)
                output = model(torch.cat([pos, sub], 1))
                output = (output>=0.5).type(torch.float32)
                #print(output.sum())
                save_tool.add_result(output.detach().cpu())

        pred = save_tool.recompone_overlap()
        recon = (pred.numpy()>0.5).astype(np.uint16)*mask
        #gt = load(os.path.join(opt.datapath, 'label_fine/', file_idx + '.nii.gz'))
        gt = load(os.path.join(opt.datapath, file_idx, 'GT.nii.gz'))
        DSC, PPV, SEN, ASD = seg_metric(recon,gt)
        index_results = OrderedDict({'DSC': DSC,'PPV': PPV,'SEN': SEN,'ASD': ASD})
        log_test.update(file_idx,index_results)
        Pred = sitk.GetImageFromArray(np.array(recon))
        #gt_img = sitk.GetImageFromArray(np.array(GT))
        result_save_path = os.path.join(save_result_path, file_idx)
        util.mkdir(result_save_path)
        sitk.WriteImage(Pred,os.path.join(result_save_path,'pred.nii.gz'))
        del pred,recon,Pred,save_tool,gt
        gc.collect()
        #torch.cuda.empty_cache()

if __name__ == '__main__':
    test_all('model_250.pth')
                
