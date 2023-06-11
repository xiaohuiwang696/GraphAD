"""
实现LSTM-NDT的阈值+剪枝
"""
import numpy as np


def GetMaxList(score, th, p=0.01):
    """
    NDT剪枝
    :param score:预测值
    :param th: 阈值
    :param p: 比例
    :return: 剪枝后的预测序列
    """
    pred = score > th
    normal = np.where(score < th)[0]
    abnormal = np.where(score > th)[0]
    nindex, nl = ContinueSeq(normal)
    aindex, al = ContinueSeq(abnormal)
    l = nl if nl < al else al
    maxn = []
    maxa = []
    for i in range(l):
        nmax = np.max(score[nindex[i]])
        amax = np.max(score[aindex[i]])
        maxn.append(nmax)
        maxa.append(amax)
    rate = (np.array(maxa) - np.array(maxn)) / np.array(maxn)
    delete = np.where(rate < p)[0]
    for s in delete:
        pred[aindex[s]] = False
    return pred


def ContinueSeq(a):
    """
    统计连续数值子序列数量
    :param a: 总序列
    :return: 序列位置，序列数
    """
    s = []
    ss = []
    nseq = 0
    for i in a:
        if len(s) == 0 or s[-1] + 1 == i:
            s.append(i)  # 入栈
        else:
            if len(s) >= 2:
                ss.append(s)
                #                 print(s)
                nseq = nseq + 1
            s = []  # 清空
            s.append(i)  # 入栈
    # 最后一轮，需判断下
    if len(s) >= 2:
        nseq = nseq + 1
        ss.append(s)
    return ss, nseq


def NDT_threshold(score, z):
    """
    NDT阈值
    :param score: 异常分数
    :param z: 标准差乘积的列表
    :return:阈值
    """
    s = []
    mean = np.mean(score)
    print(mean)
    std = np.std(score)
    print(std)
    for i in z:
        th = mean + i * std
        normal = score[score < th]
        nmean = np.mean(normal)
        nstd = np.std(normal)
        anomalindex = np.where(score > th)
        anomalN = len(anomalindex[0])
        seqN = ContinueSeq(anomalindex[0].tolist())
        theta = (mean / nmean + std / nstd) / (anomalN + seqN)
        s.append(theta)
    thresh = mean + std * z[np.argmax(s)]
    return thresh