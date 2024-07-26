import torch
import gc
import numpy as np
import torch.optim as optim
import SimpleITK as sitk
from options.Options import Options_x
from tqdm import tqdm
from Model.TokenSegV8_prototype_fusion_attention_sampling import TokenSegV8
from modelv5.loss_proto import PixelPrototypeCELoss,PPD,PPC

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

    #model = TokenSegV2(inch=2, TransformerLayerNum=4, hidden_size=192, base_channeel=16,imgsize=[48, 128, 128]).to(device)
    model = TokenSegV8(inch=2, TransformerLayerNum=8, hidden_size=256, base_channeel=32, imgsize=[48, 128, 128]).to(device)
    ckpt1 = torch.load(opt.checkpoints_dir + '/' + opt.task_name + '/model/' + model_name,map_location=device)
    model.load_state_dict(ckpt1['model'])

    # print(dense_dict.keys())
    #print(dict_cur.keys())
    # AA = list(dict_80.keys())
    # #print(AA)
    # for ii in range(len(dict_80.keys())):
    #     # print(AA[ii])
    #     dict_cur[AA[ii]] = dict_cur[AA[ii]]
    #
    # model.load_state_dict(dict_cur)

    save_result_path = os.path.join(opt.checkpoints_dir, opt.task_name, 'localization')
    util.mkdir(save_result_path)
    model.eval()
    log_test = logger.Test_Logger(save_result_path,"results")
    log_test_1 = logger.Test_Logger(save_result_path, "results_1")
    log_test_fus = logger.Test_Logger(save_result_path, "results_fus")

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
                out_seg,output_p,output,output_fus= model(torch.cat([pos, sub], 1),TEST=1)

                #k = output_p + (1 - output_p) * (1 - output_p)  # pro_seg * pre + (1 - pro_seg) * (1 - pred)
                #output_p = (output_p) / k  # (pro_seg * pre) / k

                #k = output_p * output + (1 - output_p) * (1 - output)
                #output_fus = 0.5*output+0.5*output_p  #(output_p * output) / k #
                output = (output>=0.5).type(torch.float32)
                output_p = (output_p >= 0.5).type(torch.float32)
                output_fus = (output_fus >= 0.5).type(torch.float32) ##out_seg，是用softmax得到的分割结果。

                save_tool.add_result(output.detach().cpu())
                save_tool_1.add_result(output_p.detach().cpu())
                save_tool_2.add_result(output_fus.detach().cpu())

        pred = save_tool.recompone_overlap()
        pred_1 = save_tool_1.recompone_overlap()
        pred_fus = save_tool_2.recompone_overlap()
        #pred_1 = save_tool_1.recompone_overlap()

        recon = (pred.numpy()>0.5).astype(np.uint16)*mask
        recon_1 = (pred_1.numpy() > 0.5).astype(np.uint16) * mask
        recon_fus = (pred_fus.numpy() > 0.5).astype(np.uint16) * mask

        #gt = load(os.path.join(opt.datapath, 'label_fine/', file_idx + '.nii.gz'))
        gt = load(os.path.join(opt.datapath, file_idx, 'GT.nii.gz'))
        DSC, PPV, SEN, ASD = seg_metric(recon,gt)
        index_results = OrderedDict({'DSC': DSC,'PPV': PPV,'SEN': SEN,'ASD': ASD})

        DSC, PPV, SEN, ASD = seg_metric(recon_1, gt)
        index_results_1 = OrderedDict({'DSC': DSC, 'PPV': PPV, 'SEN': SEN, 'ASD': ASD})

        DSC, PPV, SEN, ASD = seg_metric(recon_fus, gt)
        index_results_fus = OrderedDict({'DSC': DSC, 'PPV': PPV, 'SEN': SEN, 'ASD': ASD})

        log_test.update(file_idx,index_results)
        log_test_1.update(file_idx, index_results_1)
        log_test_fus.update(file_idx, index_results_fus)

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

    from thop import profile

    model = TokenSegV8(inch=2, TransformerLayerNum=8, hidden_size=256, base_channeel=32, imgsize=[48, 128, 128]).to('cuda:0')
    img = torch.randn((1, 2, 48, 128, 128)).to('cuda:0')

    flops, params = profile(model, (img,))
    print('flops: %.2f G, params: %.2f M' % (flops / 1e9, params / 1e6))
                
