import torch
import gc
import numpy as np
import torch.optim as optim
import SimpleITK as sitk
from options.Options import Options_x
from tqdm import tqdm
from network_architecture.MTLN import MTLN3D
from torch.utils.data import DataLoader
from utils import logger,util
from utils.metrics import seg_metric
import torch.nn as nn
import os
from dataset.dataset_lits_test import Test_all_Datasets,Recompone_tool
from collections import OrderedDict

def load(file):
    itkimage = sitk.ReadImage(file)
    image = sitk.GetArrayFromImage(itkimage)
    return image

def test_all(model_name='model_200.pth'):
    opt = Options_x().parse()  # get training options
    device = torch.device('cuda:' + opt.gpu_ids if torch.cuda.is_available() else "cpu")
    
    model =MTLN3D(1).to(device) #RUNet(2,1,16).to(device) #RUNet(2,1,16).to(device)
    ckpt = torch.load(opt.checkpoints_dir + '/' + opt.task_name + '/model/' + model_name,map_location=device)
    model.load_state_dict(ckpt['model'])
    #weightspaf=opt.checkpoints_dir + '/' + opt.task_name + '/model/' + model_name
    #model.load_state_dict(torch.load(weightspaf,map_location=device) ,strict=False) #'checkpointsfixed/pretrain77.9.pth'

    save_result_path = os.path.join(opt.checkpoints_dir, opt.task_name, 'Predyn')
    util.mkdir(save_result_path)
    model.eval()
    log_test = logger.Test_Logger(save_result_path,"results")
    log_test_1 = logger.Test_Logger(save_result_path, "results_1")
    log_test_2 = logger.Test_Logger(save_result_path, "results_2")

    cut_param = {'patch_s': opt.patch_size[0], 'patch_h': opt.patch_size[1], 'patch_w': opt.patch_size[2],
                 'stride_s': opt.patch_stride[0], 'stride_h': opt.patch_stride[1], 'stride_w': opt.patch_stride[2]}
    datasets = Test_all_Datasets(opt.datapath, cut_param, opt.data_folder)

    for img_dataset, original_shape, new_shape,mask,file_idx in datasets:
        save_tool = Recompone_tool(original_shape, new_shape, cut_param)
        save_tool_1 = Recompone_tool(original_shape, new_shape, cut_param)
        save_tool_2 = Recompone_tool(original_shape, new_shape, cut_param)

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
                pred1,pred2,output = model(torch.cat([pos, sub], 1))

                output = (output>=0.5).type(torch.float32)
                save_tool.add_result(output.detach().cpu())

                p1 = (pred1 >= 0.5).type(torch.float32)
                save_tool_1.add_result(p1.detach().cpu())

                p2 = (pred2 >= 0.5).type(torch.float32)
                save_tool_2.add_result(p2.detach().cpu())

        pred = save_tool.recompone_overlap()
        pred_1 = save_tool_1.recompone_overlap()
        pred_2 = save_tool_2.recompone_overlap()

        recon = (pred.numpy() > 0.5).astype(np.uint16) * mask
        recon_1 = (pred_1.numpy() > 0.5).astype(np.uint16) * mask
        recon_2 = (pred_2.numpy() > 0.5).astype(np.uint16) * mask

        gt = load(os.path.join(opt.datapath, 'label', '{}_GT.nii.gz'.format(file_idx)))

        DSC, PPV, SEN, ASD = seg_metric(recon,gt)
        index_results = OrderedDict({'DSC': DSC,'PPV': PPV,'SEN': SEN,'ASD': ASD})

        DSC, PPV, SEN, ASD = seg_metric(recon_1, gt)
        index_results_1 = OrderedDict({'DSC': DSC, 'PPV': PPV, 'SEN': SEN, 'ASD': ASD})

        DSC, PPV, SEN, ASD = seg_metric(recon_2, gt)
        index_results_2 = OrderedDict({'DSC': DSC, 'PPV': PPV, 'SEN': SEN, 'ASD': ASD})

        log_test.update(file_idx,index_results)
        log_test_1.update(file_idx, index_results_1)
        log_test_2.update(file_idx, index_results_2)

        Pred = sitk.GetImageFromArray(np.array(recon))
        #gt_img = sitk.GetImageFromArray(np.array(GT))
        result_save_path = os.path.join(save_result_path, file_idx)
        util.mkdir(result_save_path)
        sitk.WriteImage(Pred,os.path.join(result_save_path,'pred.nii.gz'))
        del pred,recon,Pred,save_tool,gt
        gc.collect()
        #torch.cuda.empty_cache()

if __name__ == '__main__':
    test_all('Final_77.56-MTLN.pth')
                
