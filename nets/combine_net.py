#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2022/3/14 下午4:51
# @Author  : Chaofan Pan
# @Email   : pan.chaofan@foxmail.com
# @File    : combine_net.py
# @Software: PyCharm
from torch import nn


class CombineNet(nn.Module):
    def __init__(self, embedding_net: nn.Module, projection_net: nn.Module):
        super(CombineNet, self).__init__()
        self.embedding_net = embedding_net
        self.projection_net = projection_net

    def forward(self, input_tensor):
        embedding = self.embedding_net.embedding(input_tensor)
        out = self.projection_net(embedding)
        return out
