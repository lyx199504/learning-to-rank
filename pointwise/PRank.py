#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/2/26 16:56
# @Author : LYX-夜光

from initialDataset import LETOR4
import numpy as np
from evaluation import NDCG
import pandas as pd

class PRank:
    def __init__(self, x, y):
        self.x, self.y = np.mat(x), np.mat(y).T
        self.T = self.x.shape[0]
        self.w = np.mat([0.] * self.x.shape[1]).T
        self.l = len(set(y))
        self.b = np.mat([0.] * (self.l-1) + [float('inf')]).T

    def train(self, iterNum=20):
        for i in range(iterNum):
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

    def test(self, testset, x):
        # 预测测试集的相关度
        y_pred = []
        for t in x.index:
            wx = float(np.mat(x.loc[t].values) * self.w)
            for r in range(self.l):
                if wx - self.b[r] < 0.:
                    y_pred.append(r)
                    break
        testset['y_pred'] = y_pred  # 将预测相关度加入测试集
        print("测试集分类准确率：%f" % (sum(testset['y'] == testset['y_pred'])/len(x)))

        queryList = list(set(testset['q']))
        testsetByQuery = testset.groupby('q').apply(lambda x: x.sort_values('y_pred', ascending=False))
        testsetNDCG = pd.DataFrame({'NDCG': [None]*len(queryList)}, index=queryList)
        k = 1
        for query in queryList:
            queryTestset = NDCG.calNDCG(testsetByQuery.loc[query])
            testsetNDCG.loc[query, 'NDCG'] = queryTestset.iloc[k-1]['NDCG']
        # testsetNDCG = testsetNDCG['NDCG'].fillna(value=1.0)
        print("测试集平均NDCG(%d)：%f" % (k, testsetNDCG.mean()))


if __name__ == "__main__":
    trainset = LETOR4.getDatasetByPandas("../datasets/LETOR4/MQ2007/Fold5/train.txt")
    x, y = trainset[range(46)], trainset['y']
    pRank = PRank(x, y)
    pRank.train(iterNum=5)
    testset = LETOR4.getDatasetByPandas("../datasets/LETOR4/MQ2007/Fold5/test.txt")
    testset, x = testset[['q', 'y']], testset[range(46)]
    pRank.test(testset, x)