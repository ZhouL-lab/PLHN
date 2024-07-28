import SimpleITK as sitk
import numpy as np
#import pytorch_ssim
#from skimage.metrics import structural_similarity
import math
from torch.autograd import Variable
from scipy import ndimage
import torch, random



def normalization(img):
    out=(img - np.min(img))/(np.max(img) - np.min(img) + 0.000001 )
    return out


def normalization_test (img):
    out=(img - np.min(img))/(np.max(img) - np.min(img) + 0.000001 )
    return out, np.max(img), np.min(img)

def center_crop_3d(img, label, slice_num=16):
    if img.shape[0] < slice_num:
        return None
    left_x = img.shape[0]//2 - slice_num//2
    right_x = img.shape[0]//2 + slice_num//2

    crop_img = img[left_x:right_x]
    crop_label = label[left_x:right_x]
    return crop_img, crop_label

def load_file_name_list(file_path):
    file_name_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()  # 整行读取数据
            if not lines:
                break
                pass
            file_name_list.append(lines)
            pass
    return file_name_list


def MaskContour(image, position='xy', line=1):
    itkimage = sitk.GetImageFromArray(image)
    if position == 'xy':
        erode_m = [line, line, 0]
    elif position == 'yz':
        erode_m = [0, line, line]
    elif position == 'zx':
        erode_m = [line, 0, line]
    else:
        erode_m = [line, line, 0]

    mask = sitk.GetArrayFromImage(sitk.BinaryErode(itkimage, erode_m))
    boundary = image - mask
    out = sitk.GetImageFromArray(boundary)
    return out


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def adjust_learning_rate(optimizer, epoch, opt):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.5 ** (epoch // opt.step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_V2(optimizer, lr):
    """Sets the learning rate to a fixed number"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        
        
def get_mse(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )

    return mse


def get_psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 10 * math.log10(PIXEL_MAX / math.sqrt(mse))

'''
def get_ssim(img1,img2):
    n=img1.shape[0]
    out = 0
    for i in range(n):
        out+=structural_similarity(img1[i].squeeze(),img2[i].squeeze())
        
    return out/n
'''    
    

def save_result(low_dose, high_dose, output, i, epoch):
    def save_img(img, name):
        # img = SimpleITK.GetImageFromArray(img[0,0].cpu().detach().numpy())
        img = sitk.GetImageFromArray(img)
        sitk.WriteImage(img, 'result/image/'+name+'.nii.gz')
        
    save_img(low_dose, 'low_dose_epoch_'+str(epoch) + "_" + str(i))
    save_img(high_dose, 'high_dose_epoch_'+str(epoch) + "_" + str(i))
    save_img(output, 'output_epoch_'+str(epoch) + "_" + str(i))


def de_normalization(img,max_x,min_x):
    return img*(max_x - min_x) + min_x