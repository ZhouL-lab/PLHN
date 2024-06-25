# Prototype Learning Guided Hybrid Network for Breast Tumor Segmentation in DCE-MRI, under reviewed by Transaction on Medical Imaging
## Paper
## Requirements
Some important required packages include:
* Python==3.8
* torch==1.9.0
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy ......
## Dataset
A total of three datasets were used in our paper, among which two private datasets were breast cancer and thymus tumor, and the other dataset was an open source breast tumor dataset.If you want to download this dataset you can refer to this [paper](https://www.cell.com/patterns/fulltext/S2666-3899(23)00195-2?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS2666389923001952%3Fshowall%3Dtrue)/[ZENODO](https://zenodo.org/records/8068383).\
If you need MRI breast tumor segmentation weights, Download the [model checkpoint](https://drive.google.com/file/d/1Y7l5W7KZMoUWKrwhca3mmyzla6dtUjm4/view?usp=sharing)
## Usage
1.Clone the repo\
2.Put the data in data/\
3.Train the model and infers
```
python main.py
```
4.If you just need to test the model
```
python test.py
```
