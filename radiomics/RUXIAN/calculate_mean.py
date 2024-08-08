import numpy as np
import pandas as pd

names = ['test5','ELCnet','PLHN','Tumorsen','MHL','ALMN','ALMN2','all']
rows = open('HER2test500.txt').readlines()
svm = []
rf = []
lr = []
for row in rows:
    row = row.strip()
    if 'SVM'==row.split('\t')[0]:
        svm_ = row.split('\t')[1:]
        svm_ = [eval(i) for i in svm_]
        svm.append(svm_)
    if '随机森林'==row.split('\t')[0]:
        rf_ = row.split('\t')[1:]
        rf_ = [eval(i) for i in rf_]
        rf.append(rf_)
    if '逻辑回归'==row.split('\t')[0]:
        lr_ = row.split('\t')[1:]
        lr_ = [eval(i) for i in lr_]
        lr.append(lr_)
rf = np.array(rf)
svm = np.array(svm)
lr = np.array(lr)
df = pd.DataFrame()
for i in range(len(names)):
    df_rf = np.mean(rf[500*i:500*i+500],axis=0)
    df_svm = np.mean(svm[500*i:500*i+500],axis=0)
    df_lr = np.mean(lr[500*i:500*i+500],axis=0)
    print(names[i])
    df_temp = pd.DataFrame([df_svm,df_rf,df_lr],columns=['AUC','accuracy', 'recall', 'pre', 'f1'],index=['svm','rf','lr'])
    print(df_temp)