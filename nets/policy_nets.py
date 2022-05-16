#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2022/3/8 下午2:25
# @Author  : Chaofan Pan
# @Email   : pan.chaofan@foxmail.com
# @File    : policy_nets.py
# @Software: PyCharm
import random
from typing import Iterable, Tuple

import torch

from torch import nn
from torchsummary import summary
from torch.nn import functional as F

from utils.utils import tensor_mask, split_input


class PolicyNet(nn.Module):
    def __init__(self, input_size, embedding_size, out_size, view, mask_ratio, cor_dim=1, is_out_ratio=False,
                 conv_depth=1, conv_width=32, fc_depth=1, rnn_layers=1):
        super(PolicyNet, self).__init__()
        self.view = view
        self.out_size = out_size
        self.mask_ratio = mask_ratio
        self.cor_dim = cor_dim  # 坐标的维度
        self.is_out_ratio = is_out_ratio
        input_size = int(input_size * mask_ratio)

        conv_blocks = [nn.Conv2d(input_size, conv_width, 3), nn.ReLU()]
        for d in range(conv_depth - 1):
            conv_blocks.append(nn.Conv2d(conv_width, conv_width, 3, padding=1))
            conv_blocks.append(nn.ReLU())
        self.encoder_cov = nn.Sequential(*conv_blocks)

        fc_blocks = [nn.Linear(conv_width * 6 * 6, embedding_size)]
        for d in range(fc_depth - 1):
            fc_blocks.append(nn.ReLU())
            fc_blocks.append(nn.Linear(embedding_size, embedding_size))
        self.encoder_fc = nn.Sequential(*fc_blocks)
        # input_size = int(input_size * mask_ratio)
        # self.states_encoders = nn.ModuleList([nn.Linear(64, 16) for _ in range(input_size)])
        # self.encoder = nn.Sequential(
        #     # nn.Linear(input_size * 16, embedding_size),
        #     nn.Linear(input_size, embedding_size),
        #     # nn.ReLU()
        # )
        # self.decoder = nn.Sequential(
        #     nn.Linear(embedding_size, self.out_size),
        #     nn.Sigmoid()
        # )
        self.decoder_rnn = nn.GRU(input_size=cor_dim, hidden_size=embedding_size, num_layers=rnn_layers,
                                  batch_first=True)
        self.decoder_fc = nn.Sequential(
            nn.Linear(embedding_size, cor_dim),
        )
        # actions_decoders = [nn.Linear(embedding_size, 64) for _ in range(self.out_size)]
        # self.actions_decoders = nn.ModuleList(actions_decoders)

    def forward(self, x):
        teacher_forcing_ratio = 0
        if isinstance(x, Tuple):  # 教师强迫
            x, y, teacher_forcing_ratio = x
        embedding, mask = self.embedding(x, True)
        embedding = F.relu(embedding)

        # return self.decoder(embedding)
        # actions = [self.actions_decoders[i](embedding) for i in range(self.out_size)]
        # return actions

        decoder_input = torch.zeros(x.shape[0], 1, self.cor_dim).to(x.device)  # 初始输入为0
        hidden = torch.unsqueeze(embedding, dim=0)  # 嵌入作为第一个隐藏层输入
        actions = []
        # TODO  mask
        for i in range(self.out_size):
            out, hidden = self.decoder_rnn(decoder_input, hidden)
            out = out[:, -1]
            out = self.decoder_fc(out)
            out = torch.unsqueeze(out, dim=1)
            teacher_force = random.random() < teacher_forcing_ratio
            decoder_input = y[:, i].view(x.shape[0], 1, self.cor_dim) if teacher_force else out
            actions.append(out)

        actions = torch.stack(actions, dim=1)
        return actions.view(x.shape[0], -1, self.cor_dim)

    def embedding(self, x, remain_mask=False):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, 8, 8)

        # ----划分数据
        if x.shape[1] > 60:
            if self.view == 'player':
                x = split_input(x, (x.shape[1] - 1) // 2)[0]  # 每一步有两个图，最后还有一层
            elif self.view == 'ai':
                x = split_input(x, (x.shape[1] - 1) // 2)[1]

        x = x.view(batch_size, -1, 8, 8)
        # embedding = [self.states_encoders[i](x[:, i]) for i in range(x.shape[1])]
        # embedding = torch.cat(embedding, dim=1)
        # embedding = self.encoder(embedding)
        mask = tensor_mask(x, self.mask_ratio)
        if self.mask_ratio != 1:
            x = torch.masked_select(x, mask)
        # embedding = self.encoder(x.view(batch_size, -1))

        embedding = self.encoder_cov(x.view(batch_size, -1, 8, 8))
        embedding = self.encoder_fc(embedding.view(batch_size, -1))

        if remain_mask:
            return embedding, mask
        else:
            return embedding
