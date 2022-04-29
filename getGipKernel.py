import numpy as np
import math



def kernel_to_distance(k):
    di = np.diag(k)
    dim = np.mat(di)  # 1 * 2 行向量
    dimt = dim.T  # 2 * 1 列向量
    d = np.tile(dim, (len(k), 1))  # 1*2,2,1     2*2
    d = d + np.tile(dimt, (1, len(k)))  # 2*1,1,2  2*2
    d = d - 2 * k
    return d


def getGipKernel(y):
    krnl = np.dot(y, y.T)  # y * y的转置
    krnl = krnl / np.mean(np.mat(np.diag(krnl)))  
    t = -1 * kernel_to_distance(krnl)
    krnl = np.exp(t)
    return krnl

