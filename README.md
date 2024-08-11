# Prototype Learning Guided Hybrid Network for Breast Tumor Segmentation in DCE-MRI([Transaction on Medical Imaging 2024](https://ieeexplore.ieee.org/document/10614219))
## Paper 
Prototype Learning Guided Hybrid Network for Breast Tumor Segmentation in DCE-MRI
Lei Zhou, Yuzhong Zhang, Jiadong Zhang, Xuejun Qian, Chen Gong, Kun Sun, Zhongxiang Ding, Xing Wang, Zhenhui Li, Zaiyi Liu, and Dinggang Shen
[IEEE TMI 2024](https://ieeexplore.ieee.org/document/10614219)/[arXiv]()
## Abstract
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
A total of three datasets were used in our paper, among which two private datasets were breast cancer and thymus tumor, and the other dataset was a publicly available breast tumor dataset. If you wish to download this publicly available dataset, please refer to the relevant [paper](https://www.cell.com/patterns/fulltext/S2666-3899(23)00195-2?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS2666389923001952%3Fshowall%3Dtrue)/[ZENODO](https://zenodo.org/records/8068383).

Additionally, we provide a publicly available breast cancer [dataset](https://drive.google.com/file/d/1ciSV337l9uyoou2GfbSRHPxHH9r6uxC9/view?usp=sharing) after processing.
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
If you require MRI breast tumor segmentation weights, download the [model checkpoint](https://drive.google.com/drive/folders/1XjBD-ylWbvKE4ND7yGjbaiE2_dM9Mw8l?usp=drive_link).
## Training
Train the model and infers.
```
python training/main.py
```
## Evaluation
1.Clone the repo.

2.Download the [dataset](https://drive.google.com/file/d/1ciSV337l9uyoou2GfbSRHPxHH9r6uxC9/view?usp=sharing) to the following path:
```
PLHN/data
```
3.Download the [model](https://drive.google.com/drive/folders/1XjBD-ylWbvKE4ND7yGjbaiE2_dM9Mw8l?usp=drive_link) weights and paste *latest_model.pth* in the following path:
```
PLHN/checkpoints/task1_breast_tumor/model
```
4.If you only need to test the model.
```
python evaluation/test.py
```
## Classification
The HER2 classification was performed by using the radiomics features.
```
python radiomics/mlclass.py
```
## Citation
If you use PLHN for your research, please cite our papers:
```
@ARTICLE{10614219,
  author={Zhou, Lei and Zhang, Yuzhong and Zhang, Jiadong and Qian, Xuejun and Gong, Chen and Sun, Kun and Ding, Zhongxiang and Wang, Xing and Li, Zhenhui and Liu, Zaiyi and Shen, Dinggang},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Prototype Learning Guided Hybrid Network for Breast Tumor Segmentation in DCE-MRI}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Magnetic resonance imaging;Breast tumor segmentation;Hybrid network;Transformer;Prototype learning},
  doi={10.1109/TMI.2024.3435450}}

```
ALMN is our work published in *Pattern Recognition* in 2022. If you use ALMN for your research, please cite our papers:
```
@article{zhou2022three,
  title={Three-dimensional affinity learning based multi-branch ensemble network for breast tumor segmentation in MRI},
  author={Zhou, Lei and Wang, Shuai and Sun, Kun and Zhou, Tao and Yan, Fuhua and Shen, Dinggang},
  journal={Pattern Recognition},
  volume={129},
  pages={108723},
  year={2022},
  publisher={Elsevier}
}
```
