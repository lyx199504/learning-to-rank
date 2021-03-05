#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/2/26 16:56
# @Author : LYX-夜光

import pandas as pd

def getLETORDatasetByList(fileName):
    with open(fileName, encoding='UTF-8') as readFile:
        readLines = readFile.readlines()
    dataList = []
    for readLine in readLines:
        data = readLine.split('#')[0].strip().split(' ')
        dataCol = [int(data[1].split(':')[1])]
        dataCol.extend([float(data[i].split(':')[1]) for i in range(2, len(data))])
        dataCol.append(int(data[0]))
        dataList.append(dataCol)
    return dataList

def getLETORDatasetByPandas(fileName):
    dataList = getLETORDatasetByList(fileName)
    dataset = pd.DataFrame(dataList, columns=['q']+list(range(46))+['y'])
    return dataset

if __name__ == "__main__":
    dataset = getLETORDatasetByPandas("../datasets/LETOR4/MQ2007/Fold1/train.txt")
    print(dataset)
    x = dataset.loc[:, range(46)]
    print(x.values)