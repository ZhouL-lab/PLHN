import torch
import gc
import numpy as np
import torch.optim as optim
import SimpleITK as sitk
from options.Options import Options_x
from tqdm import tqdm
from Model.pr_attention_fusionv3_MAML import PR_AFFINITY_NET
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

    model = PR_AFFINITY_NET(c=2, num_classes=1).to(device)
    # ckpt = torch.load(opt.checkpoints_dir + '/' + opt.task_name + '/model/' + 'latest_model.pth', map_location=device)
    # model.load_state_dict(ckpt['model'])
    ckpt = torch.load('/data/shaofengzou/PanSTEEL/MIDATA/PR_network/OTHER_CODE/NET4_fusion/checkpoints_fusionv2/runet_4/model/latest_model.pth',
        map_location=device)
    model.load_state_dict(ckpt['model'])

    org = 1
    if org == 1:  ##To save all the model.
        weightsp2 = '/data/shaofengzou/PanSTEEL/MIDATA/PR_network/NET1/checkpoints_n1/runet_4/model/model_320.pth'
        weightsp = '/data/shaofengzou/PanSTEEL/MIDATA/PR_network/NET3/checkpoints_runet/runet_4/model/model_300_78.4.pth'
        backb = torch.load(weightsp2)  ##the backbone network
        backb_dict = backb['model']
        dense = torch.load(weightsp)
        dense_dict = dense['model']
        # print(dense_dict.keys())
        AA = list(dense_dict.keys())
        for ii in range(len(dense_dict.keys())):
            backb_dict[AA[ii]] = dense_dict[AA[ii]]
        model.load_state_dict(backb_dict, strict=False)  ##加载相关模型。 strict=False

    save_result_path = os.path.join(opt.checkpoints_dir, opt.task_name, 'localization')
    util.mkdir(save_result_path)
    model.eval()

    log_test = logger.Test_Logger(save_result_path, "results_final")
    log_test_1 = logger.Test_Logger(save_result_path, "results_avef")
    log_test_2 = logger.Test_Logger(save_result_path, "results_net1")

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
                finalseg, pred_att, out = model(torch.cat([pos, sub], 1))
                output_net1 = (pred_att >= 0.5).type(torch.float32)
                output_predf = (finalseg >= 0.5).type(torch.float32)
                output_fusion = ((pred_att+out)/2 >= 0.5).type(torch.float32)
                # print(output.sum())
                save_tool.add_result(output_predf.detach().cpu()) ##result_fsion
                save_tool_1.add_result(output_fusion.detach().cpu()) ##result_1
                save_tool_2.add_result(output_net1.detach().cpu()) ##result_2

        pred = save_tool.recompone_overlap()
        pred_1 = save_tool_1.recompone_overlap()
        pred_2 = save_tool_2.recompone_overlap()

        recon = (pred.numpy() > 0.5).astype(np.uint16)
        recon_1 = (pred_1.numpy() > 0.5).astype(np.uint16)
        recon_2 = (pred_2.numpy() > 0.5).astype(np.uint16)

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
        # gt_img = sitk.GetImageFromArray(np.array(GT))
        result_save_path = os.path.join(save_result_path, file_idx)
        util.mkdir(result_save_path)
        sitk.WriteImage(Pred, os.path.join(result_save_path, 'pred.nii.gz'))
        del pred, recon, Pred, save_tool, gt
        gc.collect()
        # torch.cuda.empty_cache()

if __name__ == '__main__':
    #test_all('latest_model.pth')
    from thop import profile

    model = PR_AFFINITY_NET(c=2, num_classes=1).to('cuda:0')
    img = torch.randn((1, 2, 48, 128, 128)).to('cuda:0')

    #print(model(img).size())
    flops, params = profile(model, (img,))
    print('flops: %.2f G, params: %.2f M' % (flops / 1e9, params / 1e6))
                
