import dgl
import numpy as np
import torch
# import torch as th
# from dgl.nn.pytorch.conv import GATConv
# from torch import nn
# from torch.nn import TransformerEncoderLayer, TransformerEncoder
# from torch.utils.data import DataLoader
# import time
# from models.TCN_tcn import TemporalConvNet
# from models.decoder import DecoderLayer
# from models.discriminator import Discriminator
# from models.model import DataEmbedding, GraphAD, TimeForecasting, TimeGAT, TimeTransformer, ConvBlock, CT_ED
#
# # # Case 1: Homogeneous graph
# # g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
# # g = dgl.add_self_loop(g)
# # feat = th.ones(6, 10)
# # gatconv = GraphEncoder(10,2)
# # # gatconv = GATConv(10, 2, num_heads=3)
# # res = gatconv(g, feat)
# # print(res)
# from models.model import FeatureSE
# from utils.parser import args
# from utils.process import GraphDataset, ConstructGraph, collate
#
# """
# 注意图和节点数相同
#
# """
# import os
#
# # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# # encode =torch.ones(1,4,16)
# # u, v = th.tensor([0, 0, 0, 1]), th.tensor([1, 2, 3, 3])
# # g = dgl.graph((u, v))
# # graphencoder = GraphEncoder(16,32)
# # o2 = graphencoder(g,encode)
# # print(o2.shape)
# "TimeForecasting测试 TCN测试"
# #
# # tcn = TemporalConvNet(22,[8,22])
# # x=torch.ones(22,30)
# # # model = TimeForecasting(22,16,[8,22],30)
# # # x = torch.ones(128,22,30)
# # # encode = torch.ones(128,22,16)
# # # o = model(x,encode)
# # o = tcn(x)
# # print(o.shape)
# """
# 测试graphforecasting
#
# g1 = dgl.graph(([0, 1], [1, 0]))
# g1.ndata['h'] = torch.tensor([1., 2.])
# g2 = dgl.graph(([0, 1], [1, 1]))
# g2.ndata['h'] = torch.tensor([1., 2.])
#
# dgl.readout_nodes(g1, 'h')
# # tensor([3.])  # 1 + 2
#
# bg = dgl.batch([g1, g2])
# dgl.readout_nodes(bg, 'h')
# # tensor([3., 6.])  # [1 + 2, 1 + 2 + 3]
# bg.ndata['h']
# # tensor([1., 2., 1., 2., 3.])
# """
# # 批次GAT
# # g1 = dgl.graph(([0, 1], [1, 1]))
# # g2 = dgl.graph(([0, 1], [1, 1]))
# # g = dgl.batch([g1,g2])
# # feat = torch.ones(2,2, 10)
# # model = GraphEncoder(10,3)
# # o = model(g,feat)
# # print(o.shape)
#
# # g1 = dgl.graph(([0, 1], [1, 1]))
# # g2 = dgl.graph(([0, 1], [1, 1]))
# # g = dgl.batch([g1, g2])
# # feat = torch.ones(2, 2, 10)
# # model = GraphAD(2, 10, [10, 2])
# # o = model(feat, g)
# # # print(o.shape)
# # print(o)
#
from torch.utils.data import DataLoader

from models.model import Graph_ED
from utils.process import GraphDataset, collate

# "测试backprop函数"
# #
# trainD = np.load('data/SMD/machine-1-1_train.npy')
# trainD=trainD[:100]
# print(trainD.shape)
# trainDataset = GraphDataset(trainD, 5, 1)
# train_dataloader = DataLoader(trainDataset, batch_size=16, collate_fn=collate)
# model = Graph_ED(38, 5)
# dataloader = next(iter(train_dataloader))
# feats, graph, matrix = dataloader
# print(feats.shape)
# print(graph.edges())
# print(matrix.shape)
# out = model(feats, graph)
# print(out.shape)
# print(torch.cuda.is_available())
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

#
#
# """
# transformer
# """
# # encoder_layers = TransformerEncoderLayer(d_model=16, nhead=4, dim_feedforward=8,
# #                                          dropout=0.1, batch_first=True)
# # timeEncoder = TransformerEncoder(encoder_layers, 1)
# #
# # x = torch.rand((128, 32, 16))
# # o = timeEncoder(x)
# # print(o.shape)
# """timeTransformer"""
# trainD = np.load('data/SMD/machine-1-1_train.npy')
# print(trainD.shape)
# trainDataset = GraphDataset(trainD, 5, 1, 'deepinfo')
# train_dataloader = DataLoader(trainDataset, batch_size=args.batch)
# dataloader = next(iter(train_dataloader))
# data1,data2 = dataloader
# print(data1.shape)
# print(data2.shape)
# # model = TimeTransformer(38, 5)
#
# # feats= dataloader
# # print(feats.shape)
# # o = model(feats)
# # print(o.shape)
# """测试discriminator"""
# # c = torch.rand((128,1,16))
# # h1 = torch.rand((128, 38, 16))
# # h2 = torch.rand((128, 38, 16))
# # c = torch.mean(h1, 1)
# # m = Discriminator(16)
# # o = m(c,h1,h2)
# # print(c.shape)
# # print(o.shape)
# """测试deepinfo"""
# """测试cted"""
# x = torch.rand((128,38,6))
# model = CT_ED(38)
# o = model(x)
# print(o.shape)
from utils.parser import params
"""
获取数据集的信息
"""
# data = ['ev24','ev242','ev24n']
# i='MIT'
#
# labels = np.load(f'./data/MIT/{i}_labels.npy')
# a = np.where(labels>0)
# b = len(a[0])
# c = len(labels)
# print(b/c)
# for i in data:
#     labels = np.load(f'./data/EV/{i}_labels.npy')
#     a = np.where(labels > 0)
#     b = len(a[0])
#     c = len(labels)
#     print(len(labels))
#     print(b / c)


"""
对MIT数据的结果进行分析
"""
