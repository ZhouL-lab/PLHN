import random
import numpy as np
import SimpleITK as sitk
import os
from torch.utils.data import Dataset
  
class Lits_DataSet(Dataset):
    def __init__(self,root, sample_index = 'partial', size = (32,128,128), train_folder=0):
        self.root = root
        self.size = size
        self.sample_index = sample_index
        f = open(os.path.join(self.root,'data_folder', 'train'+str(train_folder)+'.txt'))
        self.filename = f.read().splitlines()
        self.p1 = None
        self.p2=None
        self.p3=None
        self.p4=None

    def __getitem__(self, index):

        file = self.filename[index]
        pre = self.normalization(self.load(os.path.join(self.root,'plain/',file+'.nii.gz'))).astype(np.float32)
        pos = self.normalization(self.load(os.path.join(self.root,'dyn1/',file+'.nii.gz'))).astype(np.float32)
        if pre.shape != pos.shape:
            pos=pre
            #return self.p1,self.p2,self.p3,self.p4

        sub = pos - pre
        gt = self.load(os.path.join(self.root,'label_fine/',file+'.nii.gz')).astype(np.float32)
        #print('hjsiadha',index,np.count_nonzero(gt))
        pre_patch,pos_patch,sub_patch,gt_patch = [],[],[],[]
        for i in range(3):
            if i==1:
                pre_patch1, pos_patch1, sub_patch1, gt_patch1 = self.random_crop_3d_contain(pre, pos, sub, gt, self.size)
            else:
                pre_patch1, pos_patch1, sub_patch1, gt_patch1 = self.random_crop_3d_partial(pre, pos, sub, gt, self.size)
            #print(index,i,np.count_nonzero(gt_patch1))
            pre_patch.append(pre_patch1), pos_patch.append(pos_patch1), sub_patch.append(sub_patch1), gt_patch.append(gt_patch1)

        self.p1=np.array(pre_patch)
        self.p2=np.array(pos_patch)
        self.p3=np.array(sub_patch)
        self.p4=np.array(gt_patch)

        return np.array(pre_patch),np.array(pos_patch),np.array(sub_patch),np.array(gt_patch)

    def __len__(self):
        return len(self.filename)

    def random_crop_3d_contain(self,pre,pos,sub,gt,crop_size):

        cor_box = self.maskcor_extract_3d(gt)
        random_x_min, random_x_max = max(cor_box[0,1] - crop_size[0], 0), min(cor_box[0,0], pre.shape[0]-crop_size[0])
        random_y_min, random_y_max = max(cor_box[1,1] - crop_size[1], 0), min(cor_box[1,0], pre.shape[1]-crop_size[1])
        random_z_min, random_z_max = max(cor_box[2,1] - crop_size[2], 0), min(cor_box[2,0], pre.shape[2]-crop_size[2])
        if random_x_min >random_x_max:
            random_x_min, random_x_max = cor_box[0,0], cor_box[0,1] - crop_size[0]
        if random_y_min >random_y_max:
            random_y_min, random_y_max = cor_box[1,0], cor_box[1,1] - crop_size[1]
        if random_z_min > random_z_max:
            random_z_min, random_z_max = cor_box[2, 0], cor_box[2, 1] - crop_size[2]
        #print(cor_box[2,1], cor_box[2,0],pre.shape[2]-crop_size[2])
        x_random = random.randint(random_x_min, random_x_max)
        y_random = random.randint(random_y_min, random_y_max)
        z_random = random.randint(random_z_min, random_z_max)
    
        pre_patch = pre[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1], z_random:z_random + crop_size[2]]
        pos_patch = pos[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1], z_random:z_random + crop_size[2]]
        sub_patch = sub[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1], z_random:z_random + crop_size[2]]
        gt_patch = gt[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1], z_random:z_random + crop_size[2]]

        return pre_patch,pos_patch,sub_patch,gt_patch

    def random_crop_3d_partial(self, pre, pos, sub, gt, crop_size):

        cor_box = self.maskcor_extract_3d(gt)
        random_x_min, random_x_max = max(cor_box[0, 0] - crop_size[0], 0), min(cor_box[0, 1],
                                                                               pre.shape[0] - crop_size[0])
        random_y_min, random_y_max = max(cor_box[1, 0] - crop_size[1], 0), min(cor_box[1, 1],
                                                                               pre.shape[1] - crop_size[1])
        random_z_min, random_z_max = max(cor_box[2, 0] - crop_size[2], 0), min(cor_box[2, 1],
                                                                               pre.shape[2] - crop_size[2])
        x_random = random.randint(random_x_min, random_x_max)
        y_random = random.randint(random_y_min, random_y_max)
        z_random = random.randint(random_z_min, random_z_max)

        pre_patch = pre[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                    z_random:z_random + crop_size[2]]
        pos_patch = pos[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                    z_random:z_random + crop_size[2]]
        sub_patch = sub[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                    z_random:z_random + crop_size[2]]
        gt_patch = gt[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                   z_random:z_random + crop_size[2]]

        return pre_patch, pos_patch, sub_patch, gt_patch

    def random_crop_3d(self, pre, pos, sub, gt, crop_size):
        count_num=0
        while (count_num<100):
            count_num +=1
            x_random = random.randint(0, pre.shape[0]-crop_size[0])
            y_random = random.randint(0, pre.shape[1]-crop_size[1])
            z_random = random.randint(0, pre.shape[2]-crop_size[2])

            gt_patch = gt[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                    z_random:z_random + crop_size[2]]

            pre_patch = pre[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                        z_random:z_random + crop_size[2]]
            pos_patch = pos[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                        z_random:z_random + crop_size[2]]
            sub_patch = sub[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                        z_random:z_random + crop_size[2]]
            if np.count_nonzero(gt_patch) > 10:
                break

        if count_num>=100:
            print('cannot locate mask')

        return pre_patch, pos_patch, sub_patch, gt_patch


    def min_max_normalization(self,img):
        out=(img - np.min(img))/(np.max(img) - np.min(img) + 0.000001 )
        return out

    def normalization(self, img, lmin=1, rmax=None, dividend=None, quantile=None):
        newimg = img.copy()
        newimg = newimg.astype(np.float32)
        if quantile is not None:
            maxval = round(np.percentile(newimg, 100 - quantile))
            minval = round(np.percentile(newimg, quantile))
            newimg[newimg >= maxval] = maxval
            newimg[newimg <= minval] = minval

        if lmin is not None:
            newimg[newimg < lmin] = lmin
        if rmax is not None:
            newimg[newimg > rmax] = rmax

        minval = np.min(newimg)
        if dividend is None:
            maxval = np.max(newimg)
            newimg = (np.asarray(newimg).astype(np.float32) - minval) / (maxval - minval)
        else:
            newimg = (np.asarray(newimg).astype(np.float32) - minval) / dividend
        return newimg

    def load(self,file):
        itkimage = sitk.ReadImage(file)
        image = sitk.GetArrayFromImage(itkimage)
        return image

    def maskcor_extract_3d(self,mask, padding=(0, 0, 0)):
        # mask_s = mask.shape
        p = np.where(mask > 0)
        a = np.zeros([3, 2], dtype=np.int)
        for i in range(3):
            s = p[i].min()
            e = p[i].max() + 1

            ss = s - padding[i]
            ee = e + padding[i]
            if ss < 0:
                ss = 0
            if ee > mask.shape[i]:
                ee = mask.shape[i]

            a[i, 0] = ss
            a[i, 1] = ee
        return a






