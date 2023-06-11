import math

import torch
from torch import nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder


import torch.nn.functional as F

from models.decoder import DecoderLayer
from models.embed import PositionalEmbedding, DataEmbedding


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=5, padding=2)

        self.w = nn.Parameter(torch.ones(3), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)

        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        w3 = torch.exp(self.w[2]) / torch.sum(torch.exp(self.w))

        out = conv1 * w1 + conv2 * w2 + conv3 * w3
        return torch.transpose(out, 1, 2).contiguous()


class FeatureSE(nn.Module):
    def __init__(self, channel, reduction=2):
        super(FeatureSE, self).__init__()
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """

        Args:
            x: 输入格式（B,L,Cin） 转换为（B,Cin，L）通过池化得到（B，Cin，1）
            通过线性层和激活函数得到特征维度的通道注意力，（B，Cin，L）对应元素相乘
            残差相加（B，Cin，L）
        Returns:转换为时间（B,L,Cin）

        """
        x = x.permute(0, 2, 1)
        channel = self.pooling(x).squeeze(-1)
        channel = self.fc(channel).unsqueeze(-1).expand_as(x)
        residual = x * channel
        x = x + residual
        return torch.transpose(x, 1, 2).contiguous()


class myDataEmbedding(nn.Module):
    def __init__(self, cin, dmodel, dropout=0.0):
        super(myDataEmbedding, self).__init__()
        self.value_embedding = ConvBlock(cin, dmodel)
        self.position_embedding = PositionalEmbedding(d_model=dmodel)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class FeatureTimeBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, reduction=2):
        super(FeatureTimeBlock, self).__init__()
        self.featureSE = FeatureSE(channel=d_model, reduction=reduction)
        encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
                                                 dropout=0.1, batch_first=True)
        self.timeEncoder = TransformerEncoder(encoder_layers, 1)

    def forward(self, x):
        x = self.featureSE(x)
        x = self.timeEncoder(x)

        return x


class ConvFTEncoder(nn.Module):
    def __init__(self, cin, dmodel=128, nhead=4, d_ff=64, dropout=0.0, reduction=2, nblock=1):
        super(ConvFTEncoder, self).__init__()

        self.data_embedding = myDataEmbedding(cin=cin, dmodel=dmodel, dropout=dropout)
        self.encoder = nn.ModuleList(
            [FeatureTimeBlock(d_model=dmodel, nhead=nhead, d_ff=d_ff, reduction=reduction) for i in range(nblock)]
        )

    def forward(self, x):
        x = self.data_embedding(x)
        for FeatureTimeBlock in self.encoder:
            x = FeatureTimeBlock(x)
        return x



class mydecoder(nn.Module):
    def __init__(self, cin, dmodel=16, nhead=4, d_ff=16, dropout=0.0, reduction=2):
        super(mydecoder, self).__init__()
        self.dec_embedding = DataEmbedding(cin, dmodel)
        self.decoder = DecoderLayer(dmodel, nhead, d_ff)
        self.rnn = nn.GRU(dmodel, cin, batch_first=True)

    def forward(self, x, cross):
        x_dec = self.dec_embedding(x)
        dec_out = self.decoder(x_dec, cross)
        out, hidden = self.rnn(dec_out)
        return out, hidden

