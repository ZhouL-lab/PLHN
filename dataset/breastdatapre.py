import SimpleITK as sitk
import numpy as np
import os
#import dicom2nifti
import pandas as pd
import zipfile
import shutil
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def dicom_nii(path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(dicom_names)
    image_itk = reader.Execute()
    return image_itk
import SimpleITK as sitk


def resampleVolume(outspacing, vol):
    """
    将体数据重采样的指定的spacing大小\n
    paras：
    outpacing：指定的spacing，例如[1,1,1]
    vol：sitk读取的image信息，这里是体数据\n
    return：重采样后的数据
    """
    outsize = [0, 0, 0]
    inputspacing = 0
    inputsize = 0
    inputorigin = [0, 0, 0]
    inputdir = [0, 0, 0]
    #outspacing = [0.81,0.97,0.81]


    inputsize = vol.GetSize()
    inputspacing = vol.GetSpacing()
#    print(inputsize)
#    print(inputspacing)
    transform = sitk.Transform()
    transform.SetIdentity()

    outsize[0] = int(inputsize[0] * inputspacing[0] / outspacing[0] + 0.5)
    outsize[1] = int(inputsize[1] * inputspacing[1] / outspacing[1] + 0.5)
    outsize[2] = int(inputsize[2] * inputspacing[2] / outspacing[2] + 0.5)


    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetOutputOrigin(vol.GetOrigin())
    resampler.SetOutputSpacing(outspacing)
    resampler.SetOutputDirection(vol.GetDirection())
    resampler.SetSize(outsize)
    newvol = resampler.Execute(vol)
    return newvol

def resampleVolume_b(outspacing, vol):
    """
    将体数据重采样的指定的spacing大小\n
    paras：
    outpacing：指定的spacing，例如[1,1,1]
    vol：sitk读取的image信息，这里是体数据\n
    return：重采样后的数据
    """
    outsize = [0, 0, 0]
    inputspacing = 0
    inputsize = 0
    inputorigin = [0, 0, 0]
    inputdir = [0, 0, 0]
    #outspacing = [0.81,0.97,0.81]


    inputsize = vol.GetSize()

    inputspacing = vol.GetSpacing()

#    print(inputsize)
#    print(inputspacing)
    transform = sitk.Transform()
    transform.SetIdentity()
    outsize[0] = int(inputsize[0] * inputspacing[0] / outspacing[0] + 0.5)
    outsize[1] = int(inputsize[1] * inputspacing[1] / outspacing[1] + 0.5)
    outsize[2] = int(inputsize[2] * inputspacing[2] / outspacing[2] + 0.5)


    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputOrigin(vol.GetOrigin())
    resampler.SetOutputSpacing(outspacing)
    resampler.SetOutputDirection(vol.GetDirection())
    resampler.SetSize(outsize)
    newvol = resampler.Execute(vol)
    return newvol


def load(path):
    a = sitk.ReadImage(path)
    s = sitk.GetArrayFromImage(a)
    return s


path='//'

for file in os.listdir(path):
    a = sitk.ReadImage(os.path.join(path,'GT','{}_GT.nii.gz'.format(file)))
    s = sitk.ReadImage(os.path.join(path,file,'{}_P0.nii.gz'.format(file)))
    print(os.path.join(path,file,'{}_P0.nii.gz'.format(file)))
    a.SetOrigin(s.GetOrigin())
    a.SetSpacing(s.GetSpacing())
    a.SetDirection(s.GetDirection())
    new = sitk.ReadImage(os.path.join(path,file,'{}_P0.nii.gz'.format(file)))
    #newvol = resampleVolume_b([1, 1, 1], a)
    wriiter = sitk.ImageFileWriter()
    wriiter.SetFileName(os.path.join('/data/','{}_GT.nii.gz'.format(file)))
    wriiter.Execute(a)
    if new.GetSize() != a.GetSize():
        print('Wrong!')
    print(file)
