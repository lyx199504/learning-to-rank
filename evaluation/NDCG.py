#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/3/4 18:27
# @Author : LYX-夜光

import numpy as np
import pointwise
import pandas as pd

def calNDCG(dataset):
    # 输入的dataset是每个query对应的文档数据
    dataset = pd.DataFrame(dataset)  # 防止抛异常“SettingWithCopyWarning”

    # 计算DCG
    D = np.array([1/np.log2(r+1) for r in range(1, dataset.shape[0]+1)])  # 位置折扣因子（position discount factor）
    G = np.array([np.power(2, dataset.iloc[i]['y']) - 1 for i in range(dataset.shape[0])])  # 相关性函数 G=2^y-1
    dataset['DCG'] = (D*G).cumsum()

    # 计算DCG_max
    y_sort = np.array(dataset['y'].sort_values(ascending=False))  # 逆序排列真实相关度y
    G_max = np.array([np.power(2, y_sort[i]) - 1 for i in range(dataset.shape[0])])
    dataset['DCG_max'] = (D*G_max).cumsum()

    # 计算NDCG
    dataset['NDCG'] = dataset['DCG']/dataset['DCG_max']
    return dataset

if __name__ == "__main__":
    trainset = pointwise.getLETORDatasetByPandas("../datasets/LETOR4/MQ2007/Fold1/train.txt")
    print(trainset)
    d = trainset.groupby('q').apply(lambda x: x.sort_values('y', ascending=False))
    data = calNDCG(d.loc[10])
    print(data)
