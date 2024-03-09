# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 16:59:51 2023

@author: yohoho_ll
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator

attributes = {'Class':pd.Series(['no-recurrence-events','recurrence-events']),\
'age':pd.Series(['10-19','20-29','30-39','40-49','50-59','60-69','70-79',
                  '80-89', '90-99']),\
'menopause': pd.Series(['lt40','ge40','premeno']),\
'tumor_size':pd.Series(['0-4','5-9','10-14','15-19','20-24','25-29','30-34',
                        '35-39','40-44','45-49','50-54','55-59']),\
'inv_nodes':pd.Series(['0-2','3-5','6-8','9-11','12-14','15-17','18-20',
                       '21-23','24-26','27-29','30-32','33-35','36-39']),\
'node_caps':pd.Series(['yes','no']),\
'deg_malig':pd.Series(['1','2','3']),\
'breast':pd.Series(['left','right']),\
'breast_quad':pd.Series(['left_up','left_low','right_up','right_low','central']),\
'irradiat':pd.Series(['yes','no'])}
labels = ['Class','age','menopause','tumor_size','inv_nodes','node_caps','deg_malig','breast','breast_quad','irradiat']

datas = pd.read_csv('D:\\yohoho\\机器学习\\breast+cancer\\breast-cancer.data',header=None, sep=',')

##数据清洗
datas1 = datas.copy()
for i in range(len(labels)):
    for j in range(len(datas)):
        if datas1[i][j] == '?':
            datas1[i][j] = np.nan
datas1.fillna(value = {5:datas[5].value_counts().idxmax()}, inplace = True)
datas1.fillna(value = {8:datas[8].value_counts().idxmax()}, inplace = True)

for label in labels:
    a = dict(attributes[label])
    attributes[label] = dict(zip(a.values(),a.keys()))

datas1[0] = datas1[0].map(attributes['Class'])
datas1[2] = datas[2].map(attributes['menopause'])
datas1[5] = datas1[5].map(attributes['node_caps'])
datas1[7] = datas1[7].map(attributes['breast'])
datas1[8] = datas1[8].map(attributes['breast_quad'])
datas1[9] = datas1[9].map(attributes['irradiat'])

###可视化分析
##频数条形图
plt.rcParams['font.family'] = 'SimHei'
plt.subplots(3, 3, figsize = (15, 15))
letter = list('abcdefghi')

for i in range(1,10):
    ax = plt.subplot(3,3,i)
    ax.set_xlabel(...,fontsize = 10)
    ax.set_ylabel(...,fontsize = 10)
    t = datas[i].value_counts()
    p = t.plot(kind = 'bar')
    plt.ylabel('频数')
    plt.xlabel('(%s)%s'%(letter[i-1],labels[i]))
    if i == 3:
        plt.setp(ax.get_xticklabels(),rotation=10)
    else:
        plt.setp(ax.get_xticklabels(),rotation=0)
     
plt.tight_layout()
    
##并列条形图
x = datas[1:10]
y1 = label  
    

