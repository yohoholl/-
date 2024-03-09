# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 19:28:37 2023

@author: yohoho_ll

knn  with mushroom
"""
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

#读取并处理乳腺癌数据
file1 = open('D:\\哟嚯嚯LL\\机器学习\\breast+cancer\\breast-cancer.data', 'r')
arrayOLines = file1.readlines()
numberOfLines = len(arrayOLines)
file1.close()
index = 0

def changedata(name:list):
    z = []
    for i in range(len(name)):
        z.append(i)
    return dict(zip(name, z))

Class = 'no-recurrence-events,recurrence-events'.split(',')
age = '10-19,20-29,30-39,40-49,50-59,60-69,70-79,80-89,90-99'.split(',')
menopause = 'lt40,ge40,premeno'.split(',')
tumorSize = '0-4,5-9,10-14,15-19,20-24,25-29,30-34,35-39,40-44,45-49,50-54,55-59'.split(',')
invNodes = '0-2,3-5,6-8,9-11,12-14,15-17,18-20,21-23,24-26,27-29,30-32,33-35,36-39'.split(',')
nodeCaps =  'yes,no,?'.split(',')
degMalig = '1,2,3'.split(',')
breast = 'left,right'.split(',')
breastQuad = 'left_up,left_low,right_up,right_low,central,?'.split(',')
irradiat = 'yes,no'.split(',')
attributes = [Class, age, menopause, tumorSize, invNodes, nodeCaps, degMalig, 
          breast, breastQuad, irradiat]
Attributes = []
for i in attributes:
    i = changedata(i)
    Attributes.append(i)

filematrix = zeros((numberOfLines, len(attributes)))
for line in arrayOLines:
    line = line.strip()
    listFromline = line.split(',')
    for i in range(len(listFromline)):
        filematrix[index, i] = Attributes[i][listFromline[i]]
    index += 1
###随机打乱数据（只改变行）
random.shuffle(filematrix)


###创建数据集
def creatDataSet(filematrix:ndarray):
    group = filematrix[:,1:]
    labels = filematrix[:,0]
    return group, labels

### k-近邻算法
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    ##计算距离
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    ##argsort()函数是对数组中的元素从小到大排序，并返回相应序列元素的数组下标
    sortedDistIndicies = distances.argsort()
    classCount = {}
    ## 选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    ##排序
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

group, labels = creatDataSet(filematrix)
a = int(numberOfLines*0.8)
grouplearn = group[0:a,:]
grouptest = group[a:,:]
labelslearn = labels[0:a]
labelstest = labels[a:]
print(grouptest,labelstest)

##测试数据  ##计算误差率
result = []
estimationError = 0
for i in range(len(labelstest)):
    a = classify0(grouptest[i,:], grouplearn, labelslearn, 50)
    result.append(a)
    estimationError += (a-labelstest[i])**2
estimationError = estimationError/(len(labelstest))    
print(result, estimationError)


