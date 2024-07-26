# Prototype Learning Guided Hybrid Network for Breast Tumor Segmentation in DCE-MRI accepted by Transaction on Medical Imaging
## Paper
Automated breast tumor segmentation on the basis of dynamic contrast-enhancement magnetic resonance imaging (DCE-MRI) has shown great promise in clinical practice, particularly for identifying the presence of breast disease. This paper presents a prototype learning guided hybrid network (PLHN) approach that combines the CNN and transformer layers with two parallel encoder subnetworks to effectively segment breast tumors.
### Architecture overview of PLHN
![image](https://github.com/ZhouL-lab/PLHN/blob/main/img/tmi1.png)
### Results
Segmentation performance achieved by different methods in terms of DSC(%), PPV(%), SEN(%) and ASD(mm) with 95% confidence intervals on the internal and external test dataset for breast tumor segmentation. ↑ means the higher value the better and ↓ means the lower value the better.The external data set is the public data set.
![image](https://github.com/ZhouL-lab/PLHN/blob/main/img/TMI2.png)
### Visualization
The visual comparison of segmentation results between different methods, such as DMFnet, MTLN, ResUnet, UXNET, MHL, Tumrosen, ALMN and PLHN, is displayed. Each row corresponds to one subject, and post-contrast images in axial plane overlaid with ground truth (red line) and automatic segmentation results (green line) of different methods are provided.
![image](https://github.com/ZhouL-lab/PLHN/blob/main/img/TMI3.png)
## Requirements
Some important required packages include:
* Python==3.8
* torch==1.9.0
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy ......
## Dataset
A total of three datasets were used in our paper, among which two private datasets were breast cancer and thymus tumor, and the other dataset was an open source breast tumor dataset.If you want to download this dataset you can refer to this [paper](https://www.cell.com/patterns/fulltext/S2666-3899(23)00195-2?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS2666389923001952%3Fshowall%3Dtrue)/[ZENODO](https://zenodo.org/records/8068383).\
We also provide a publicly available breast cancer [dataset](https://drive.google.com/file/d/1ciSV337l9uyoou2GfbSRHPxHH9r6uxC9/view?usp=sharing) after processing.
```
YNpre
│
├── data_folder
│   └── test5.txt
├── breastmask
│   ├── Yunnan_1.nii.gz
│   ├── ...
│   └── Yunnan_100.nii.gz 
├── image_p0
│   ├── Yunnan_1_P0.nii.gz
│   ├── ...
│   └── Yunnan_100_P0.nii.gz
├── image_p1
│   ├── Yunnan_1_P1.nii.gz
│   ├── ...
│   └── Yunnan_100_P1.nii.gz
└── label
    ├── Yunnan_1_GT.nii.gz
    ├── ...
    └── Yunnan_100_GT.nii.gz
```
If you need MRI breast tumor segmentation weights, Download the [model checkpoint](https://drive.google.com/drive/folders/1XjBD-ylWbvKE4ND7yGjbaiE2_dM9Mw8l?usp=drive_link).
## Training
Train the model and infers
```
python training/main.py
```
## Evaluation
1.Clone the repo\
2.Download the [dataset](https://drive.google.com/file/d/1ciSV337l9uyoou2GfbSRHPxHH9r6uxC9/view?usp=sharing) to the following path:
```
PLHN/data
```
3.Download [model](https://drive.google.com/drive/folders/1XjBD-ylWbvKE4ND7yGjbaiE2_dM9Mw8l?usp=drive_link) weights and paste latest_model.pth in the following path:
```
PLHN/checkpoints_RUENT_8/runet_4/model
```
4.If you just need to test the model
```
python evaluation/test.py
```
## Citation
If you use our work, please consider citing:
