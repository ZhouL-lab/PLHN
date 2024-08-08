"""
作者：ztf08
名称：机器学习.py
说明：
日期：2021/11/22 13:11
"""
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr,pearsonr
from sklearn.feature_selection import RFE, RFECV
# from imblearn.over_sampling import SMOTE
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.width', 500)
# from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier as RF  # 随机森林
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split  # k折交叉验证
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.neural_network import MLPClassifier  # bp 神经网络
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC  # 支持向量机
from sklearn.neighbors import KNeighborsClassifier as KNN  # KNN
from sklearn.linear_model import LogisticRegression as LR  # 逻辑回归
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB  # 朴素贝叶斯
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV, Lasso
from collections import defaultdict
from xgboost import XGBClassifier
from scipy import stats
from ITMO_FS.filters.univariate import f_ratio_measure, pearson_corr, spearman_corr, kendall_corr
from scipy.spatial.distance import pdist, squareform
from onekey_algo.custom.components.stats import clinic_stats
from onekey_algo.custom.components.comp1 import select_feature
import onekey_algo.custom.components as okcomp
from onekey_algo.custom.components.Radiology import ConventionalRadiomics
import csv
from sklearn.model_selection import GridSearchCV




def ttest(a,b):
    p_value= stats.ttest_ind(a, b).pvalue
    return p_value

def distcorr(X, Y):
    X = np.atleast_1d(X) # 将输入转换为至少一维的数组
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X): # np.prod 计算所有元素的乘积
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor

def suanscore(y_true, y_pred):
    # 自定义的f1 socre
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return acc, recall, pre, f1


def GetDatasets(name):
    filename1 = '/radiomics/feature/%s_1.csv'%name
    filename0 = '/radiomics/feature/%s_0.csv'%name
    file_test = '/radiomics/feature/test5.xlsx'
    df1 = pd.read_csv(filename1,index_col=0)
    df0 = pd.read_csv(filename0,index_col=0)
    df_test = pd.read_excel(file_test,index_col=0)
    # select_label = 'tumor_grade'
    select_label = 'HER2'
    # select_label = 'BI-RADS'
    if select_label == 'HER2':
        df1.loc[df1[select_label] == 0, select_label] = '0'
        df1.loc[df1[select_label] == 1, select_label] = '1'
        df0.loc[df0[select_label] == 0, select_label] = '0'
        df0.loc[df0[select_label] == 1, select_label] = '1'
    elif select_label == 'tumor_grade':
        df1.loc[df1[select_label] == 2, select_label] = '0'
        df1.loc[df1[select_label] == 3, select_label] = '1'
        df0.loc[df0[select_label] == 2, select_label] = '0'
        df0.loc[df0[select_label] == 3, select_label] = '1'
    df1 = df1[df1[select_label].isin(['0', '1'])]
    df0 = df0[df0[select_label].isin(['0', '1'])]

    # datasets = df0
    datasets = pd.merge(df0,df1.iloc[:,3:],left_index=True,right_index=True,how='inner')
    # datasets = pd.merge(df_test, datasets, left_index=True, right_index=True, how='inner')
    labels = datasets[select_label].values.astype(np.int)

    datasets = datasets.drop(['HER2','tumor_grade','BI-RADS'], axis=1)
    columns = datasets.columns
    norm_datasets = MinMaxScaler().fit_transform(datasets)

    return norm_datasets, labels,columns


def train_model(x_train, y_train, x_test, y_test):
    '''
    用三种不同分类方法对训练集进行五折交叉验证，获得模型评估分数
    :param datasets: 训练集
    :param labels: 标签集，0/1
    :return: 模型评估分数
    '''

    svc = SVC(C=20, random_state=666, probability=True)
    rf = RF(random_state=666, class_weight='balanced_subsample')
    knn = KNN()
    lr = LR(random_state=666, max_iter=500)
    gnb = GaussianNB()
    mnb = MultinomialNB()
    bnb = BernoulliNB()
    bp = MLPClassifier(hidden_layer_sizes=(64,64,64), batch_size=32,
                       solver='sgd', max_iter=2000, random_state=1)
    model_names = ['支持向量机', 'KNN', '随机森林', '逻辑回归', 'BP', '高斯朴素贝叶斯']
    model_select = [svc, knn, rf, lr, bp, gnb]

    model_pred = []
    model_prob = []
    for model in model_select:
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        model_pred.append(pred)
        if model==lr:
            weight = model.coef_
        prob = model.predict_proba(x_test)[:, 1]
        model_prob.append(prob)
    return model_names, model_pred, weight[0], model_prob



def main(num=1,name='all'):
    datas,labels, columns= GetDatasets(name)
    print('正例：',sum(labels),'反例：',len(labels)-sum(labels))

    error_ = {}
    test_assess = []  # 存放CNN分类结果
    ml_assess = []  # 存放CNN+组学分类结果
    zuxue_assess0 = []  # 存放组学分类结果

    print('降维前：',datas.shape)

    X = pd.DataFrame(datas,columns=columns)
    Y = pd.DataFrame(labels,columns=['label'])
    X = pd.concat([Y,X],axis=1)

    # ttest
    pvalues = []
    for i in columns:
        pvalue = ttest(X[X['label']==0][i], X[X['label']==1][i])
        pvalues.append(pvalue)
    pvalues = pd.Series(pvalues,index=columns)
    sel_feature = list(pvalues[pvalues<0.05].index)
    X = X[sel_feature]
    print('ttest后：', X.shape)

    # # ttest检验
    # stats = clinic_stats(X,stats_columns=list(X.columns[1:]),label_column='label',continuous_columns=list(X.columns[1:]))
    # pvalue=0.05
    # a=stats['pvalue']
    # sel_feature = list(stats[stats['pvalue']<pvalue]['feature_name'])
    # X = X[sel_feature]
    # print('ttest后：', X.shape)

    coef = X.corr()  # method{'pearson', 'kendall', 'spearman'} ,default=pearson
    sel_feature = select_feature(coef,threshold=0.9,topn=128,verbose=False)
    X = X[sel_feature]
    columns_name = X.columns
    datas = X.values
    # # del_index = []
    # # b = np.argwhere(abs(coef.values) > 0.9) # 相关性>0.9的特征
    # # for i in b:
    # #     if i[0] < i[1]:
    # #         del_index.append(i[0])
    # # del_index = list(set(del_index))
    # # datas = np.delete(X.values,del_index,1)
    print('泊松相关性分析后：', datas.shape)

    # pca = PCA(0.995)
    # datas = pca.fit_transform(datas)
    # print('pca降维后：', datas.shape)

    best_acc = 0
    best_experiment = 0
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    model_name = ['SVM', '随机森林', '逻辑回归', 'bp', 'xgboost']
    svc = SVC(probability=True,C=3,degree=2,gamma='scale',kernel='poly',random_state=66)
    rf = RF(min_samples_split=2,max_depth=5,max_features='log2',n_estimators=200,random_state=66)
    Lr = LR(max_iter=100,C=1,penalty='l2',solver='newton-cg',random_state=66)
    bp = MLPClassifier(hidden_layer_sizes=(10), batch_size=30,
                       solver='sgd', max_iter=100,random_state=66)
    xgb = XGBClassifier(use_label_encoder=False,random_state=66)
    for tra_index, test_index in skf.split(datas, labels):
        train_data,train_label = datas[tra_index],labels[tra_index]
        test_data,test_label = datas[test_index],labels[test_index]
        # train_data, train_label,val_data,val_label = train_test_split(train_data,train_label,test_size=0.2,random_state=1,stratify=train_label)

        train_data0,test_data0=train_data.copy(),test_data.copy()

        # train_data, val_data,train_label,val_label = train_test_split(train_data, train_label,test_size=0.1, random_state=666, stratify=train_label)

        alpha = okcomp.comp1.lasso_cv_coefs(train_data0, train_label)
        # Lasso特征选择
        print('*'*25, 'Lasso特征选择', '*'*25)
        se_features = []
        reg = Lasso(alpha=alpha).fit(train_data0, train_label)
        feat_coef = [(feat_name,coef) for feat_name,coef in zip(columns_name,reg.coef_) if abs(coef)>1e-6]
        se_features.append([feat for feat,_ in feat_coef])
        # reg = LassoCV(cv=StratifiedKFold(5), random_state=666).fit(train_data0, train_label)
        se = np.where(abs(reg.coef_) >1e-6)[0]
        print(se)
        if len(se)==0:
            se = [i for i in range(len(columns_name))]
        # print('特征：', se.shape[0])
        train_data = train_data0[:, se]
        test_data = test_data0[:, se]
        print('Lasso特征选择：', train_data.shape[1])

        ml_model = [svc, rf, Lr, bp, xgb]
        temp_list0 = []
        for clf in ml_model:
            clf.fit(train_data,train_label)
            # y_pred = clf.predict(test_data)
            y_prob = clf.predict_proba(test_data)
            y_pred = np.argmax(y_prob,axis=1)
            acc, recall, pre, f1 = suanscore(test_label, y_pred)
            auc = roc_auc_score(test_label,y_prob[:,1])
            temp_list0.append([auc, acc, recall, pre, f1])

        print(25 * '-' + '组学' + 25 * '-')
        df = pd.DataFrame(temp_list0, index=model_name, columns=['AUC','accuracy', 'recall', 'precision', 'f1'])
        print(df)
        zuxue_assess0.append(temp_list0)
        # 打印单组学结果
    print(25 * '-' + "组学分类结果" + 25 * '-')
    model_mean = []
    for j, name in enumerate(model_name):
        mean_assess = []
        for i in range(len(zuxue_assess0)):
            mean_assess.append(zuxue_assess0[i][j])
        mean_assess = np.mean(np.array(mean_assess), axis=0)
        model_mean.append(mean_assess)
    df4 = pd.DataFrame(model_mean, index=model_name, columns=['AUC','accuracy', 'recall', 'pre', 'f1'])
    print(df4)
    open('HER2实验500.txt','a').write('experiments: %d\n'%num)
    df4.to_csv('HER2实验500.txt',encoding='utf8',sep='\t',mode='a',float_format='%.3f')
    if max(df4['accuracy'])>best_acc:
        best_acc = max(df4['accuracy'])
        best_experiment = num
    return best_acc,best_experiment

if __name__ == '__main__':
    main()

