#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2021/3/11 下午5:16
# @Author  : PCF
# @Email   : pan.chaofan@foxmail.com
# @File    : lvp.py
# @Software: PyCharm
import struct

import numpy as np
import torch
from torch import nn

from data_process import LINE_SIZE, BOARD_LINE_SIZE
# from data_process import deal_one_line
from nets.lv_net import LvNet, LvNet_CNN
from predictor import Predictor
from utils.utils import pos2nums, board_transform


class LvPredictor(Predictor):

    def __init__(self, net=None, board_data_type=False, learning_rate=0.004, embedding_size=128, weight_decay=0.08,
                 reduce_lr=False, view='player'):
        super(LvPredictor, self).__init__(net=net, data_shape=8 * 8 * 2 * 60 + 64 if board_data_type else 60,
                                          embedding_size=embedding_size, view=view,
                                          loss_fn=nn.MSELoss(), board_data_type=board_data_type,
                                          learning_rate=learning_rate, weight_decay=weight_decay, reduce_lr=reduce_lr)

    def create_net(self):
        if self.data_size > 60:
            self.net = LvNet_CNN(self.embedding_size, self.view)
        else:
            self.net = LvNet()  # 正常模型
        # temp_net = ThreeNet()
        # self.net = temp_net.to(self.device)  # 分段模型

    def pos2vector(self, pos_file, length):
        """将pos文件中前length行数据转换为向量"""
        data = np.zeros((length, self.data_size), np.float32)
        levels = np.zeros(length, np.uint8)
        with open(pos_file, 'rb') as f:
            for i in range(length):
                if self.board_data_type:
                    line = f.read(BOARD_LINE_SIZE)
                    line_data, ai_level = board_transform(line, self.data_size)
                else:
                    line = f.read(LINE_SIZE)
                    ad_id, ai_first, pos, ai_level, black_score, white_score = struct.unpack('36s?120sHHH', line)
                    line_data = pos2nums(pos)
                data[i] = line_data  # / 63.0  # 归一化
                levels[i] = ai_level  # / 60  # 预测级别

        data = torch.from_numpy(data)
        vector = self.net.embedding(data.to(self.device))
        return vector.cpu().detach().numpy(), levels
