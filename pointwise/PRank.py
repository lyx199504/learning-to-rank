#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/2/26 16:56
# @Author : LYX-夜光

import pointwise
import numpy as np

class PRank:
    def __init__(self, x, y):
        self.xMat, self.yMat = np.mat(x), np.mat(y).T
        self.T = self.xMat.shape[0]
        self.w = np.mat([0.] * self.xMat.shape[1]).T
        self.l = len(set(y))
        self.b = np.mat([0.] * (self.l-1) + [float('inf')]).T
        self.iterNum = 10

    def train(self):
        for i in range(self.iterNum):
            count = 0
            for t in range(self.T):
                wx = float(self.xMat[t]*self.w)
                for r in range(self.l):
                    if wx - self.b[r] < 0.:
                        if self.yMat[t] != r:
                            z = [-1 if self.yMat[t] <= r else 1 for r in range(self.l - 1)]
                            e = np.mat([z[r] if z[r]*(wx - self.b[r]) <= 0 else 0 for r in range(self.l-1)]).T
                            self.w += e.sum()*self.xMat[t].T
                            self.b[:self.l-1] -= e
                        else:
                            count += 1
                        break
            print("第%d轮训练，精度为：%f" % (i+1, count/self.T))

    def test(self, x, y):
        count = 0
        for t in range(len(x)):
            wx = float(np.mat(x[t]) * self.w)
            for r in range(self.l):
                if wx - self.b[r] < 0.:
                    if r == y[t]:
                        count += 1
                    break
        print("测试集正确率：%f" % (count/len(x)))


if __name__ == "__main__":
    x, y = pointwise.getDataList("../datasets/MQ2007/Fold1/train.txt")
    pRank = PRank(x, y)
    pRank.train()
    x, y = pointwise.getDataList("../datasets/MQ2007/Fold1/test.txt")
    pRank.test(x, y)