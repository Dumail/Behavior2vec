#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2021/12/7 下午2:21
# @Author  : Chaofan Pan
# @Email   : pan.chaofan@foxmail.com
# @File    : LvNet.py
# @Software: PyCharm
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from utils.utils import split_input


class LvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # embedding
        self.embedding_layers = nn.Sequential(
            nn.Linear(60, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),

        )
        self.out_layer = nn.Linear(128, 1)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        x = self.embedding_layers(x)
        x = F.relu(x)
        return self.out_layer(x)

    def embedding(self, x):
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.FloatTensor(x)
            return self.embedding_layers(x)


class LvNet_CNN(nn.Module):
    def __init__(self, embedding_size, view):
        super().__init__()
        self.view = view
        self.embedding_size = embedding_size
        # 卷积层
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(60, 128, 4),
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

        self.out_layer = nn.Linear(self.embedding_size, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = F.relu(x)
        return self.out_layer(x)

    def embedding(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)

        user_x, ai_x = split_input(x, -1)
        if self.view == 'player':
            x = user_x
        elif self.view == 'ai':
            x = ai_x

        x = x.reshape(-1, 60, 8, 8)
        x = self.cnn_layers(x)
        x = x.reshape(-1, 256 * 9)
        return self.embedding_layers(x)


class SingleNet(nn.Module):
    def __init__(self, embedding_size, out_size=1, class_num=3, predict_prob=False):
        super().__init__()
        if predict_prob:
            self.model = nn.Linear(embedding_size, class_num)
            # self.model = nn.Sequential(
                # nn.Linear(embedding_size, embedding_size),
                # nn.ReLU(),
                # nn.Linear(embedding_size, embedding_size),
                # nn.ReLU(),
                # nn.Linear(embedding_size, embedding_size),
                # nn.ReLU(),
                # nn.Linear(embedding_size, embedding_size),
                # nn.ReLU(),
                # nn.Linear(embedding_size, class_num)
            # )
        else:
            self.model = nn.Linear(embedding_size, out_size)

        self.predict_prob = predict_prob

    def forward(self, x):
        return self.model(x)

    def embedding(self, x):
        return self(x)
