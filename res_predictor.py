#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2022/4/20 上午11:54
# @Author  : Chaofan Pan
# @Email   : pan.chaofan@foxmail.com
# @File    : res_predictor.py
# @Software: PyCharm
from nets.res_net import ResNet_CNN
from predictor import Predictor
from torch import nn


class ResPredictor(Predictor):
    def __init__(self, net=None, board_data_type=True, learning_rate=0.004, embedding_size=128, weight_decay=0.08,
                 reduce_lr=False, steps=50):
        super(ResPredictor, self).__init__(net=net, data_shape=8 * 8 * 2 * 60 + 64 if board_data_type else 60,
                                          embedding_size=embedding_size, view='both', is_predict_result=True,
                                          loss_fn=nn.CrossEntropyLoss(), board_data_type=board_data_type, steps=steps,
                                          learning_rate=learning_rate, weight_decay=weight_decay, reduce_lr=reduce_lr)

    def create_net(self):
        self.net = ResNet_CNN(self.embedding_size, steps = self.steps)
