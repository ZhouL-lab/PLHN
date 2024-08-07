import torch
import gc
import numpy as np
import torch.optim as optim
import SimpleITK as sitk
from options.Options import Options_x
from tqdm import tqdm
from Model.pr_network1 import PR_net1
from Model.runet import RUNet

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

    model_1 = PR_net1(c=2, num_classes=1).to(device)
    ckpt = torch.load('/data/shaofengzou/PanSTEEL/MIDATA/PR_network/OTHER_CODE/NET1/checkpoints_n1/runet_4/model/model_320.pth',
                      map_location=device)
    model_1.load_state_dict(ckpt['model'])

    model_2 = RUNet(2, 1, 16).to(device)
    ckpt = torch.load('/data/shaofengzou/PanSTEEL/MIDATA/PR_network/NET3/checkpoints_runet/runet_4/model/model_300_78.4.pth', map_location=device)
    model_2.load_state_dict(ckpt['model'])
    #weightspaf=opt.checkpoints_dir + '/' + opt.task_name + '/model/' + model_name
    #model.load_state_dict(torch.load(weightspaf,map_location=device) ,strict=False) #'checkpointsfixed/pretrain77.9.pth'

    save_result_path = os.path.join(opt.checkpoints_dir, opt.task_name, 'localization')
    util.mkdir(save_result_path)
    model_1.eval()
    model_2.eval()

    log_test = logger.Test_Logger(save_result_path, "results_fusion")
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
		
                pre = pre.unsqueeze(1).type(torch.float32)
                pos = pos.unsqueeze(1).type(torch.float32)
                sub = sub.unsqueeze(1).type(torch.float32)

                _, output_1, _ = model_1(torch.cat([pos, sub], 1)) ## subnetwork1
                output_2 = model_2(torch.cat([pos, sub], 1))## res_unet
                #output=(output_1+output_2)/2
                k=output_1*output_2+(1-output_1)*(1-output_2)
                output=(output_1*output_2)/k

                output = (output >= 0.5).type(torch.float32)
                output_1 = (output_1 >= 0.5).type(torch.float32)
                output_2 = (output_2 >= 0.5).type(torch.float32)
                # print(output.sum())
                save_tool.add_result(output.detach().cpu())
                save_tool_1.add_result(output_1.detach().cpu())
                save_tool_2.add_result(output_2.detach().cpu())

        pred = save_tool.recompone_overlap()
        fusion_prob=pred

        pred_1 = save_tool_1.recompone_overlap()
        pred_2 = save_tool_2.recompone_overlap()

        recon = (pred.numpy() > 0.5).astype(np.uint16)*mask
        recon_1 = (pred_1.numpy() > 0.5).astype(np.uint16)*mask
        recon_2 = (pred_2.numpy() > 0.5).astype(np.uint16)*mask

        gt = load(os.path.join(opt.datapath, file_idx, 'GT.nii.gz'))

        DSC, PPV, SEN, ASD = seg_metric(recon, gt)
        index_results = OrderedDict({'DSC': DSC, 'PPV': PPV, 'SEN': SEN, 'ASD': ASD})

        DSC, PPV, SEN, ASD = seg_metric(recon_1, gt)
        index_results_1 = OrderedDict({'DSC': DSC, 'PPV': PPV, 'SEN': SEN, 'ASD': ASD})

        DSC, PPV, SEN, ASD = seg_metric(recon_2, gt)
        index_results_2 = OrderedDict({'DSC': DSC, 'PPV': PPV, 'SEN': SEN, 'ASD': ASD})

        log_test.update(file_idx, index_results)
        log_test_1.update(file_idx, index_results_1)
        log_test_2.update(file_idx, index_results_2)

        Pred = sitk.GetImageFromArray(np.array(recon))
        Pred_1 = sitk.GetImageFromArray(np.array(recon_1))
        Pred_2 = sitk.GetImageFromArray(np.array(recon_2))
        # gt_img = sitk.GetImageFromArray(np.array(GT))
        result_save_path = os.path.join(save_result_path, file_idx)
        util.mkdir(result_save_path)

        #saveprob=sitk.GetImageFromArray(np.array(fusion_prob).astype(np.float32))
        #sitk.WriteImage(saveprob, os.path.join(result_save_path, 'savedprob.nii.gz'))
        sitk.WriteImage(Pred, os.path.join(result_save_path, 'pred.nii.gz'))
        sitk.WriteImage(Pred_1, os.path.join(result_save_path, 'pred_1.nii.gz'))
        sitk.WriteImage(Pred_2, os.path.join(result_save_path, 'pred_2.nii.gz'))

        del pred, recon, Pred,Pred_1,Pred_2, save_tool, gt
        gc.collect()
        # torch.cuda.empty_cache()

if __name__ == '__main__':
    test_all('model_300.pth')
                
