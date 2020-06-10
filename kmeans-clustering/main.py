#_*_coding:utf-8_*_
'''
@project: cv-classification
@author:
@time: 2020/6/2 8:10 下午
'''

import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms
from numpy import *
from load_data import FlameSet
import matplotlib.pyplot as plt

# 计算两个矩阵的距离
def euclDistance(vector1, vector2):
    return sqrt(sum(power(vector2 - vector1, 2)))


# 在样本集中随机选取k个样本点作为初始质心
def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape  # 矩阵的行数、列数
    centroids = zeros((k, dim))  # 感觉要不要你都可以
    for i in range(k):
        index = int(random.uniform(0, numSamples))  # 随机产生一个浮点数，然后将其转化为int型
        centroids[i, :] = dataSet[index, :]
    return centroids


# k-means cluster
# dataSet为一个矩阵
# k为将dataSet矩阵中的样本分成k个类
def kmeans(dataSet, k):
    numSamples = dataSet.shape[0]  # 读取矩阵dataSet的第一维度的长度,即获得有多少个样本数据
    # first column stores which cluster this sample belongs to,
    # second column stores the error between this sample and its centroid
    clusterAssment = mat(zeros((numSamples, 2)))  # 得到一个N*2的零矩阵
    clusterChanged = True

    ## step 1: init centroids
    centroids = initCentroids(dataSet, k)  # 在样本集中随机选取k个样本点作为初始质心

    while clusterChanged:
        clusterChanged = False
        ## for each sample
        for i in range(numSamples):  # range
            minDist = 100000.0
            minIndex = 0
            ## for each centroid
            ## step 2: find the centroid who is closest
            # 计算每个样本点与质点之间的距离，将其归内到距离最小的那一簇
            for j in range(k):
                distance = euclDistance(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j

                    ## step 3: update its cluster
            # k个簇里面与第i个样本距离最小的的标号和距离保存在clusterAssment中
            # 若所有的样本不在变化，则退出while循环
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist ** 2  # 两个**表示的是minDist的平方

        ## step 4: update centroids
        for j in range(k):
            # clusterAssment[:,0].A==j是找出矩阵clusterAssment中第一列元素中等于j的行的下标，返回的是一个以array的列表，第一个array为等于j的下标
            pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]  # 将dataSet矩阵中相对应的样本提取出来
            centroids[j, :] = mean(pointsInCluster, axis=0)  # 计算标注为j的所有样本的平均值

    print('Congratulations, cluster complete!')
    print(clusterAssment)
    return centroids, clusterAssment
#centroids为k个类别，其中保存着每个类别的质心
#clusterAssment为样本的标记，第一列为此样本的类别号，第二列为到此类别质心的距离


# 肘部法求最佳K值
# 使用各个簇内的样本点到所在簇质心的距离平方和（SSE）作为性能度量，越小则说明各个类簇越收敛。
# 将clusterAssment的第二列求和就行
def chooseK(dataSet, i):
    reslist = []
    for j in range(1, i):
        centroids, clusterAssment = kmeans(dataSet, j)
        sum0 = sum(clusterAssment[:, 1])
        reslist.append(sum0)
    res = [i/78 for i in reslist]
    return res



if __name__ == '__main__':
    # 数据预处理
    temp = FlameSet('./data')
    dataSet = np.zeros((78, 3246303))  # 数据集的大小为1000*12288
    # print(dataSet.shape)
    for i in range(78):  # test中共1000张图片
        arr = temp[i].numpy()  # 将将Tensor张量转化为numpy矩阵
        arr = arr.reshape(3246303)  # 将矩阵拉成向量
        dataSet[i][:] = dataSet[i][:] + arr  # 添加到数据集中，每一行表示一张图片信息
    # print(dataSet)
    centroids, clusterAssment = kmeans(dataSet, 10)

    res = chooseK(dataSet, 10)
    fig = plt.figure()
    plt.plot([i for i in range(1, 10)], res)
    plt.title('k-means cv result')
    plt.savefig('res.png')
    plt.show()

