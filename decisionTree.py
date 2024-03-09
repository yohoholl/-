# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 17:07:12 2023

@author: yohoho_ll"""

from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from math import log
from sklearn import preprocessing

#读取并处理乳腺癌数据
file1 = open('D:\\yohoho\\机器学习\\breast+cancer\\breast-cancer.data', 'r')
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
    

###数据标签化
le = preprocessing.LabelEncoder()

features = ['Class','age', 'menopause', 'tumorSize', 'invNodes', 'nodeCaps', 'degMalig', 
          'breast', 'breastQuad', 'irradiat']
for feature in features:
    #非数字型和数字型标签值标准化
    le.fit(cancer[feature])
    cancer[feature] = le.transform(cancer[feature])
cancer.head()

###计算给定数据集的香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    ## 为所有可能的分类创建字典
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob* log(prob, 2)
    return shannonEnt

def createDataSet():
    