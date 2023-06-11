"""
对比不同的异常分数和阈值结果
"""
#
# np.save('loss.npy', stamplossFinal)
# np.save('label.npy', labelsFinal)
# result2, pred = pot_eval(stamplossTfinal, stamplossFinal, labelsFinal)
import numpy as np
import pandas as pd

# from utils.parser import args
from utils.pot import pot_eval


def getMatirxscore(loss, type='mean'):
    if type == 'mean':
        score = np.mean(loss, axis=1)
        score = np.mean(score, axis=1)
    elif type == 'max':
        score = np.max(loss, axis=1)
        score = np.max(score, axis=1)
    return score


def getTimeScore(tloss):
    """
    从重构损失中获得异常分数
    :param msematrix: (B,Channel,Length)
    :return:(B)
    """
    meanscore = getMatirxscore(tloss)

    score = tloss[:, :, -1]
    stampscore = np.mean(score, axis=1)

    maxscore = getMatirxscore(tloss)
    return meanscore, stampscore, maxscore


def getGraphScore(gloss):
    """
    从图重构损失中获得异常分数
    :param msematrix: (B,Channel,Channel)
    :return:(B)
    """
    meanscore = getMatirxscore(gloss)
    maxscore = getMatirxscore(gloss)

    return meanscore, maxscore


def getTimePOT(tloss0, tloss, label):
    tmeanscore0, tstampscore0, tmaxscore0 = getTimeScore(tloss0)
    tmeanscore, tstampscore, tmaxscore = getTimeScore(tloss)
    tmeanresult, pred = pot_eval(tmeanscore0, tmeanscore, label)
    tstampresult, pred = pot_eval(tstampscore0, tstampscore, label)
    print(tstampresult)
    np.save('tstampscore.npy', tstampscore)
    np.save('pred.npy', pred)
    tmaxresult, pred = pot_eval(tmaxscore0, tmaxscore, label)

    print("Time Mean Result\n", tmeanresult)
    print("Time Stamp Result\n", tstampresult)
    print("Time Max Result\n", tmaxresult)
    result = [tmeanresult, tstampresult, tmaxresult]
    return result


def getstampresult(tloss0, tloss, label):
    score0 = tloss0[:, :, -1]
    stampscore0 = np.mean(score0, axis=1)
    score = tloss[:, :, -1]
    stampscore = np.mean(score, axis=1)

    tstampresult, pred = pot_eval(stampscore0, stampscore, label)
    print(tstampresult)


def getGraphPOT(gloss0, gloss, label):
    gmeanscore0, gmaxscore0 = getGraphScore(gloss0)
    gmeanscore, gmaxscore = getGraphScore(gloss)
    gmeanresult, pred = pot_eval(gmeanscore0, gmeanscore, label)
    gmaxresult, pred = pot_eval(gmaxscore0, gmaxscore, label)
    print("Grpah Mean Result\n", gmeanresult)
    print("Grpah Max Result\n", gmaxresult)


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def mergePOT(gloss0, gloss, tloss0, tloss, label, gratio=0.5):
    tmeanscore0, tstampscore0, tmaxscore0 = getTimeScore(tloss0)
    tmeanscore, tstampscore, tmaxscore = getTimeScore(tloss)
    # gmeanscore0, gmaxscore0 = getGraphScore(gloss0)
    # gmeanscore, gmaxscore = getGraphScore(gloss)

    tmeanresult, pred = pot_eval(tmeanscore0, tmeanscore, label)
    tstampresult, pred = pot_eval(tstampscore0, tstampscore, label)
    np.save('gadtstampresult.npy', tstampscore)
    tmaxresult, pred = pot_eval(tmaxscore0, tmaxscore, label)

    # gmeanresult, pred = pot_eval(gmeanscore0, gmeanscore, label)
    gmaxresult, pred = pot_eval(gloss0, gloss, label)

    merge0 = 1 / (1 / normalization(tstampscore0) + 1 / normalization(gloss0))
    merge = 1 / (1 / normalization(tstampscore) + 1 / normalization(gloss))

    mergeresult, pred = pot_eval(merge0, merge, label)

    merge20 = normalization(tstampscore0) + normalization(gloss0)
    merge2 = normalization(tstampscore) + normalization(gloss)

    mergeresult2, pred = pot_eval(merge20, merge2, label)

    merge30 = normalization(tstampscore0) + gratio * normalization(gloss0)
    merge3 = normalization(tstampscore) + gratio * normalization(gloss)

    mergeresult3, pred = pot_eval(merge30, merge3, label)

    np.save('gadmerge.npy', merge)
    np.save('gadmerge3.npy', merge3)
    #
    # mergemax0 = 1 / (1 / tmaxscore0 + 1 / gmaxscore0)
    # mergemax = 1 / (1 / tmaxscore + 1 / gmaxscore)
    # mergeMaxresult, pred = pot_eval(mergemax0, mergemax, label)
    # return

    print("Time Mean Result\n", tmeanresult)
    print("Time Stamp Result\n", tstampresult)
    print("Time Max Result\n", tmaxresult)
    print("Grpah Max Result\n", gmaxresult)
    print("Merge Mean Result\n", mergeresult)
    print("Merge Mean Result2\n", mergeresult2)
    print("Merge Mean Result3\n", mergeresult3)
    result = [tmeanresult, tstampresult, tmaxresult, gmaxresult, mergeresult, mergeresult2, mergeresult3]
    return result

    # print("Merge Max Result\n", mergeMaxresult)
