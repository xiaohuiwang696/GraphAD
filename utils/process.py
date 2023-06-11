import dgl
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from dtw import dtw
import torch
import torch.nn.functional as F


def _slide_window(rows, sw_width, sw_steps):
    '''
    函数功能：
    按指定窗口宽度和滑动步长生成数据位置
    --------------------------------------------------
    参数说明：
    rows：单个文件中的行数；
    sw_width：滑动窗口的窗口宽度；
    sw_steps：滑动窗口的滑动步长；
    '''
    start = 0
    s_num = (rows - sw_width) // sw_steps  # 计算滑动次数
    new_rows = sw_width + (sw_steps * s_num)  # 完整窗口包含的行数，丢弃少于窗口宽度的采样数据；

    while True:
        if (start + sw_width) > new_rows:  # 如果窗口结束索引超出最大索引，结束截取；
            yield start, rows - 1
            return
        yield start, start + sw_width
        start += sw_steps


def covert2window(data, window_size=30, step=5):
    row = data.shape[0]
    window = []
    for start, end in _slide_window(row, window_size, step):
        '''
        row,sw,st

        '''
        if end - start < window_size:
            time_series = data.iloc[start:end, :]
            copy_len = window_size - (end - start)
            copy = data.iloc[end].values
            newarray = np.repeat([copy], copy_len, 0)
            seq = np.vstack((time_series.values, newarray))
        else:
            time_series = data.iloc[start:end, :]
            seq = time_series.values
        window.append(seq)
    return np.stack(window)


#
# namelist = ['\lb04.csv', '\lb24.csv', '\lb33.csv', '\lb41.csv', '\lb42.csv', '\lb47.csv', '\lb57.csv', '\lb70.csv',
#             '\lb74.csv', '\lb79.csv']


# namelist = ['\lb04.csv']
def allbv2window(data, namelist, window, step):
    allseq = []
    for name in namelist:
        bv = data.loc[data['vid'] == name]
        bvseq = covert2window(bv, window, step)
        allseq.append(bvseq)
    return (np.vstack(allseq))


def softmax(x):
    """ softmax function """

    # assert(len(x.shape) > 1, "dimension must be larger than 1")
    # print(np.max(x, axis = 1, keepdims = True)) # axis = 1, 行

    x -= np.max(x, axis=1, keepdims=True)  # 为了稳定地计算softmax概率， 一般会减掉最大的那个元素

    # print("减去行最大值 ：\n", x)

    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    return x


def ConstructGraph(window, topK=0, graphtype='DTW'):
    """
    用DTW从时间序列中构造出图
    :return: 时间序列的邻接矩阵
    """
    feature = window.shape[1]
    if topK == 0:
        topK = 3
    matrix = np.zeros((feature, feature), float)
    u = []
    v = []
    # 计算距离矩阵
    matrix = np.matmul(matrix, matrix.T)
    # for i in range(feature):
    #     for j in range(feature - 1, i, -1):
    #         # print(i,j)
    #         dis = dtw(window[:, i], window[:, j])
    #         matrix[i, j] = dis.distance
    #         matrix[j, i] = dis.distance
    # 对距离矩阵进行softmax
    # matrix = softmax(matrix)
    # 将距离矩阵进行排序，选择topK的序列为1
    for i in range(feature):
        dislist = matrix[i]
        sorted_id = sorted(range(len(dislist)), key=lambda k: dislist[k], reverse=False)
        # print(sorted_id)
        unew = [i for j in range(topK)]
        vnew = sorted_id[:topK]
        u = u + unew
        v = v + vnew
    # print((u,v))
    return (u, v)


def CosGraph(feature, topk=0):
    cos_ji_mat = torch.matmul(feature, feature.permute(1, 0))
    normed_mat = torch.matmul(feature.norm(dim=-1).view(-1, 1), feature.norm(dim=-1).view(1, -1))
    cos_ji_mat = cos_ji_mat / normed_mat
    cos_ji_mat = torch.where(torch.isnan(cos_ji_mat), torch.full_like(cos_ji_mat, 0), cos_ji_mat)
    # cos_ji_mat = F.softmax(cos_ji_mat, dim=0)
    # cos_ji_mat = torch.where(torch.isnan(cos_ji_mat), torch.full_like(cos_ji_mat, 0), cos_ji_mat)

    D, L = feature.shape
    if topk == 0:
        topk_num = 3
    # 选取topK置1 27X20
    topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]
    gated_i = torch.tensor(np.array(range(D)).repeat(topk_num), dtype=torch.int64)
    # print(gated_i)
    gated_j = topk_indices_ji.flatten()
    # print(gated_j)
    g = dgl.graph((gated_i, gated_j))
    g = dgl.add_self_loop(g)

    return g, cos_ji_mat


# 构造图数据集，输入一整段时间序列，滑动窗口分割序列，并对每个子序列构造邻接矩阵
class GraphDataset(Dataset):
    def __init__(self, data, window=6, step=1, datatype='graph'):
        self.windows = covert2window(pd.DataFrame(data), window, step)
        self.type = datatype

    def __getitem__(self, index):
        window = torch.from_numpy(self.windows[index]).T
        if self.type == 'time':
            return window.to(torch.float)
        else:
            graph, matrix = CosGraph(window)
            return window.to(torch.float), graph, matrix.to(torch.float)

    def __len__(self):
        return len(self.windows)




# 重写dataloader的colleate函数
def collate(samples):
    # 输入`samples` 是一个列表
    # 每个元素都是一个二元组 (图, 标签)
    feats, graphs, matrix = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return torch.stack(feats), batched_graph, torch.stack(matrix)

