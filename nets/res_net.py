#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2022/4/20 上午11:51
# @Author  : Chaofan Pan
# @Email   : pan.chaofan@foxmail.com
# @File    : res_net.py
# @Software: PyCharm
import torch
from torch import nn
import torch.nn.functional as F


class ResNet_CNN(nn.Module):
    def __init__(self, embedding_size, steps):
        super().__init__()
        self.embedding_size = embedding_size
        self.input_size = steps * 2 + 1
        # 卷积层
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(self.input_size, 128, 4),
            # nn.Conv2d(121,120,3,padding=1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(inplace=True),
            # nn.Conv2d(256, 512, 3),
            # nn.ReLU(inplace=True)
        )
        # embedding
        self.embedding_layers = nn.Linear(256 * 9, self.embedding_size)

        self.out_layer = nn.Linear(self.embedding_size, 3)

    def forward(self, x):
        x = self.embedding(x)
        x = F.relu(x)
        return self.out_layer(x)

    def embedding(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        x = x.reshape(-1, self.input_size, 8, 8)
        x = self.cnn_layers(x)
        x = x.reshape(-1, 256 * 9)
        return self.embedding_layers(x)
