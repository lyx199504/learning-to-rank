#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/2/26 16:56
# @Author : LYX-夜光

def getDataList(fileName):
    with open(fileName, encoding='UTF-8') as readFile:
        readLines = readFile.readlines()
    x, y = [], []
    for readLine in readLines:
        data = readLine.split('#')[0].strip().split(' ')
        y.append(int(data[0]))
        x.append([float(data[i].split(':')[1]) for i in range(2, len(data))])
    return x, y