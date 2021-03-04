#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/2/26 16:56
# @Author : LYX-夜光

import pointwise
import numpy as np

class PRank:
    def __init__(self, trainset):
        self.x, self.y = np.mat(trainset.loc[:, range(46)]), np.mat(trainset['y']).T
        self.T = self.x.shape[0]
        self.w = np.mat([0.] * self.x.shape[1]).T
        self.l = len(set(trainset['y']))
        self.b = np.mat([0.] * (self.l-1) + [float('inf')]).T
        self.iterNum = 10

    def train(self):
        for i in range(self.iterNum):
            count = 0
            for t in range(self.T):
                wx = float(self.x[t]*self.w)
                for r in range(self.l):
                    if wx - self.b[r] < 0.:
                        if self.y[t] != r:
                            z = [-1 if self.y[t] <= r else 1 for r in range(self.l - 1)]
                            e = np.mat([z[r] if z[r]*(wx - self.b[r]) <= 0 else 0 for r in range(self.l-1)]).T
                            self.w += e.sum()*self.x[t].T
                            self.b[:self.l-1] -= e
                        else:
                            count += 1
                        break
            print("第%d轮训练，精度为：%f" % (i+1, count/self.T))

    def test(self, testset):
        # 预测测试集的相关度
        x = testset.loc[:, range(46)]
        y_pred = []
        for t in x.index:
            wx = float(np.mat(x.loc[t].values) * self.w)
            for r in range(self.l):
                if wx - self.b[r] < 0.:
                    y_pred.append(r)
                    break
        testset['y_pred'] = y_pred  # 将预测相关度加入测试集
        print("测试集正确率：%f" % (sum(testset['y'] == testset['y_pred'])/len(x)))


if __name__ == "__main__":
    trainset = pointwise.getLETORDatasetByPandas("../datasets/LETOR4/MQ2007/Fold1/train.txt")
    pRank = PRank(trainset)
    pRank.train()
    testset = pointwise.getLETORDatasetByPandas("../datasets/LETOR4/MQ2007/Fold1/test.txt")
    pRank.test(testset)