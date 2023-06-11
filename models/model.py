import math

import numpy as np
import torch.nn as nn
import torch
import dgl
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GATConv
from torch.nn import init, TransformerEncoderLayer, TransformerEncoder

from models.CT_ED import FeatureSE, mydecoder, ConvFTEncoder
from models.TCN_tcn import TemporalConvNet
from models.decoder import DecoderLayer
from models.discriminator import Discriminator


# from utils.parser import args


class DataEmbedding(nn.Module):
    def __init__(self, channel, in_feats, out_feats):
        super(DataEmbedding, self).__init__()
        self.embedding = nn.Linear(in_features=in_feats, out_features=out_feats, bias=True)
        self.att = FeatureSE(out_feats)

    def forward(self, x):
        """
        in:(128,22,30)===>(128,22,16)
        Linear嵌入，再对每个传感器进行调整
        :param x: 输入节点特征（B，N，L）
        :return: 返回嵌入数据（B，N，embed）
        """
        x = self.embedding(x)
        x = self.att(x)

        return x


class TimeForecasting(nn.Module):
    def __init__(self, x_feats, encode_feats, tcn_hidden, out_length):
        """
        对输入数据进行TCN重构，将graphencoder的输出也进行重构
        :param x_feats: 传感器数，原始数据的特征维度
        :param encode_feats: graphEncoder的输出维度，将其通过线性层转为out——length
        :param tcn_hidden: TCN的隐藏层参数列表，要求最后一个为特征维度【x，x，x，x_feats】
        :param out_length: 将graph输出编码为时间长度
        """
        super(TimeForecasting, self).__init__()
        self.TCN = TemporalConvNet(x_feats, tcn_hidden)
        self.timedecoder = nn.Linear(encode_feats, out_length)

        self.w = nn.Parameter(torch.ones(x_feats, out_length), requires_grad=True)
        # init.kaiming_uniform_(self.w.weight, a=math.sqrt(5))

    def forward(self, x, encode):
        """
        对时间序列的预测重构 对TCN时序嵌入和GAT的节点嵌入融合
        :param x: 原始序列，传入到TCN中(B.Cin,L)
        :param encode: 通过encoder编码的输出(B,Cin,Lgraph)
        :return: 返回重构的值(B,Cin,L)
        """
        out1 = self.TCN(x)
        out2 = self.timedecoder(encode)
        out = self.w * out1 + (1 - self.w) * out2
        return out


class GraphForecasting(nn.Module):
    def __init__(self, in_feats, out_feats, channel):
        super(GraphForecasting, self).__init__()
        self.channel = channel
        self.project = nn.Sequential(
            nn.Linear(in_features=in_feats, out_features=out_feats),

        )
        self.graph = nn.Linear(2 * out_feats, 1, False)
        self.w = nn.Parameter(torch.zeros(channel, channel), requires_grad=True)
        # init.kaiming_uniform_(self.w.weight, a=math.sqrt(5))

    def forward(self, x):
        """

        GAT的节点构造为图in:(B,N,E)===>(B,N,N)
        :param x:GAT输出
        :return:图矩阵
        """
        graph1 = torch.matmul(x, x.permute(0, 2, 1))
        h = self.project(x)
        N = self.channel  # N 图的节点数
        B = x.shape[0]

        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, -1), h.repeat(1, N, 1)], dim=2).view(B, N, N, -1)
        # [N, N, 2*out_features]
        graph2 = self.graph(a_input).squeeze(-1)
        graph = self.w * graph1 + (1 - self.w) * graph2
        # graph = F.softmax(graph, dim=2)

        return graph


class GraphAD(nn.Module):
    def __init__(self,
                 channel,  # 传感器数量
                 time,  # 滑动窗口长度
                 time_hidden,  # 列表要求最后一层为channel值
                 embed_out=16,  # 时间维度映射维度
                 encoder_out=4,  # GAT的输出维度
                 encoder_heads=4,  # GAT每层头数，中间层为head*hidden
                 graph_hidden=4):
        super(GraphAD, self).__init__()
        self.name = 'GraphAD'
        self.Embedding = DataEmbedding(channel=channel, in_feats=time, out_feats=embed_out)
        self.GraphEncoder = GATConv(in_feats=embed_out, out_feats=encoder_out, num_heads=encoder_heads)
        self.TimeForecasting = TimeForecasting(x_feats=channel, encode_feats=encoder_out * encoder_heads,
                                               tcn_hidden=time_hidden,
                                               out_length=time)
        self.GraphForecasting = GraphForecasting(in_feats=encoder_out * encoder_heads, out_feats=graph_hidden,
                                                 channel=channel)

    def forward(self, feature, graph):
        """
        :param feature: 节点特征（B，Cin,L）
        :param graph: dgl.graph
        :return: 返回重构的时间序列和邻接矩阵
        """
        B, C, L = feature.shape

        embed = self.Embedding(feature)
        embed = embed.view(B * C, -1)
        encode = self.GraphEncoder(graph, embed)
        encode = embed.view(B, C, -1)
        time = self.TimeForecasting(feature, encode)
        graph = self.GraphForecasting(encode)
        return time, graph


class GraphTransformer(nn.Module):
    def __init__(self,
                 channel,  # 传感器数量
                 time,  # 滑动窗口长度
                 time_hidden,
                 embed_out=16,  # 时间维度映射维度
                 encoder_out=4,  # GAT的输出维度
                 encoder_heads=4,  # GAT每层头数，中间层为head*hidden
                 graph_hidden=4):
        super(GraphTransformer, self).__init__()
        self.name = 'GraphTransformer'
        self.Embedding = DataEmbedding(channel=channel, in_feats=time, out_feats=embed_out)
        encoder_layers = TransformerEncoderLayer(d_model=embed_out, nhead=4, dim_feedforward=8,
                                                 dropout=0.1, batch_first=True)
        self.Encoder = TransformerEncoder(encoder_layers, 1)

        # self.Decoder = DecoderLayer(d_model=embed_out, n_head=4, d_ff=8)
        # self.TimeForecasting = nn.Linear(in_features=embed_out, out_features=time)
        self.TimeForecasting = TimeForecasting(x_feats=channel, encode_feats=embed_out,
                                               tcn_hidden=time_hidden,
                                               out_length=time)
        self.GraphForecasting = GraphForecasting(in_feats=embed_out, out_feats=graph_hidden,
                                                 channel=channel)

    def forward(self, feature):
        embed = self.Embedding(feature)
        encode = self.Encoder(embed)
        # decode = self.Decoder(embed, encode)
        time = self.TimeForecasting(feature, encode)
        graph = self.GraphForecasting(encode)
        return time, graph


class TimeGAT(nn.Module):
    def __init__(self,
                 channel,  # 传感器数量
                 time,  # 滑动窗口长度
                 time_hidden,  # 列表要求最后一层为channel值
                 embed_out=16,  # 时间维度映射维度
                 encoder_heads=4,  # GAT每层头数，中间层为head*hidden
                 encoder_hidden=4,  # GAT隐藏单元
                 ):
        super(TimeGAT, self).__init__()
        self.embed_out = embed_out
        self.name = 'TimeGAT'
        self.Embedding = DataEmbedding(channel=channel, in_feats=time, out_feats=embed_out)
        # encoder_layers = TransformerEncoderLayer(d_model=embed_out, nhead=4, dim_feedforward=embed_out,
        #                                          dropout=0.1, batch_first=True)
        # self.Encoder = TransformerEncoder(encoder_layers, 1)

        self.GraphEncoder = GATConv(in_feats=embed_out, out_feats=encoder_hidden, num_heads=encoder_heads,
                                    allow_zero_in_degree=True)
        self.TimeForecasting = TimeForecasting(x_feats=channel, encode_feats=embed_out,
                                               tcn_hidden=time_hidden,
                                               out_length=time)

    def forward(self, feature, graph):
        # matirx = torch.matmul(feature.T,feature)
        B, C, L = feature.shape
        # feature = feature.view(-1, L)
        embed = self.Embedding(feature)
        embed = embed.view(B * C, -1)
        encode = self.GraphEncoder(graph, embed)
        encode = encode.view(B, C, -1)
        # encode = self.Encoder(embed)
        time = self.TimeForecasting(feature, encode)
        return time


class GraphGAT(nn.Module):
    def __init__(self,
                 channel,  # 传感器数量
                 time,  # 滑动窗口长度
                 time_hidden,  # 列表要求最后一层为channel值
                 embed_out=16,  # 时间维度映射维度
                 encoder_heads=4,  # GAT每层头数，中间层为head*hidden
                 encoder_hidden=4,  # GAT隐藏单元
                 graph_hidden=4):
        super(GraphGAT, self).__init__()
        self.embed_out = embed_out
        self.name = 'GraphGAT'
        self.Embedding = DataEmbedding(channel=channel, in_feats=time, out_feats=embed_out)
        self.GraphEncoder = GATConv(in_feats=embed_out, out_feats=encoder_hidden, num_heads=encoder_heads,
                                    allow_zero_in_degree=True)
        self.GraphForecasting = GraphForecasting(in_feats=encoder_hidden * encoder_heads, out_feats=graph_hidden,
                                                 channel=channel)

    def forward(self, feature, graph):
        # matirx = torch.matmul(feature.T,feature)
        B, C, L = feature.shape
        # feature = feature.view(-1, L)
        embed = self.Embedding(feature)
        embed = embed.view(B * C, -1)
        encode = self.GraphEncoder(graph, embed)
        encode = encode.view(B, C, -1)
        # encode = self.Encoder(embed)
        graph = self.GraphForecasting(encode)
        return graph


class TimeTransformer(nn.Module):
    def __init__(self,
                 channel,  # 传感器数量
                 time,  # 滑动窗口长度
                 embed_out=16,  # 时间维度映射维度
                 ):
        super(TimeTransformer, self).__init__()
        self.embed_out = embed_out
        self.name = 'TimeTransformer'
        self.Embedding = DataEmbedding(channel=channel, in_feats=time, out_feats=embed_out)
        encoder_layers = TransformerEncoderLayer(d_model=embed_out, nhead=4, dim_feedforward=8,
                                                 dropout=0.1, batch_first=True)
        self.Encoder = TransformerEncoder(encoder_layers, 1)

        self.Decoder = DecoderLayer(d_model=embed_out, n_head=4, d_ff=8)
        self.output = nn.Linear(in_features=embed_out, out_features=time)

    def forward(self, feature):
        embed = self.Embedding(feature)
        encode = self.Encoder(embed)
        decode = self.Decoder(embed, encode)
        time = self.output(decode)
        # time = self.TimeForecasting(feature, encode)
        return time


class CT_ED(nn.Module):
    def __init__(self, cin, time, dmodel=16, nhead=4, nblock=1, d_layer=1, d_ff=16, dropout=0.0, reduction=2):
        super(CT_ED, self).__init__()
        self.name = 'CT_ED'
        self.encoder = ConvFTEncoder(cin=cin, dmodel=dmodel, nhead=nhead, d_ff=d_ff, dropout=dropout,
                                     reduction=reduction, nblock=nblock)

        # self.decoder = decoder(cin=cin, d_model=dmodel, n_heads=nhead, d_ff=d_ff, dropout=dropout, d_layers=d_layer)
        self.decoder = mydecoder(cin, dmodel, nhead, d_ff)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x_time_enc = self.encoder(x)
        x_out, hidden = self.decoder(x, x_time_enc)

        return x_out.permute(0, 2, 1)




class Graph_ED(nn.Module):
    def __init__(self, cin, time, dmodel=16, nhead=4, encoder_hidden=4, d_ff=4, dropout=0.0, reduction=2):
        super(Graph_ED, self).__init__()
        self.name = 'Graph_ED'
        self.Embedding = DataEmbedding(channel=cin, in_feats=time, out_feats=dmodel)
        self.GraphEncoder = GATConv(in_feats=dmodel, out_feats=encoder_hidden, num_heads=nhead,
                                    allow_zero_in_degree=True)

        # self.decoder = decoder(cin=cin, d_model=dmodel, n_heads=nhead, d_ff=d_ff, dropout=dropout, d_layers=d_layer)
        self.decoder = mydecoder(time, dmodel, nhead, d_ff)

    def forward(self, feature,graph):
        B, C, L = feature.shape
        # feature = feature.view(-1, L)
        embed = self.Embedding(feature)
        embed = embed.view(B * C, -1)
        encode = self.GraphEncoder(graph, embed)
        encode = encode.view(B, C, -1)
        # feature = feature.permute(0, 2, 1)
        # encode = encode.permute(0, 2, 1)

        x_out, hidden = self.decoder(feature, encode)

        return x_out