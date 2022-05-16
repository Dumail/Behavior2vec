#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2022/3/8 下午2:21
# @Author  : Chaofan Pan
# @Email   : pan.chaofan@foxmail.com
# @File    : policy_encoder.py
# @Software: PyCharm
import numpy as np
import torch

from torch.utils.data import TensorDataset
from torch import nn

from data_process import BOARD_LINE_SIZE
from nets.policy_nets import PolicyNet
from trainer import Trainer
from utils.utils import load_data_from_file, split_index, board_transform, split_input


class Behavior2vec(Trainer):
    def __init__(self, net=None, learning_rate=0.004, weight_decay=0, embedding_size=128, reduce_lr=False,
                 loss_fn=nn.MSELoss(), view='both', mask_ratio=0.7, teacher_force_ratio=1, cor_dim=1, steps=-1,
                 conv_depth=1, conv_width=32, fc_depth=3, rnn_layers=1, task='lv'):
        super(Behavior2vec, self).__init__("Behavior2vec", net, data_shape=8 * 8 * 2 * 60 + 64,
                                           learning_rate=learning_rate, weight_decay=weight_decay, loss_fn=loss_fn,
                                           embedding_size=embedding_size, teacher_force_ratio=teacher_force_ratio,
                                           reduce_lr=reduce_lr, view=view, mask_ratio=mask_ratio, cor_dim=cor_dim,
                                           steps=steps, conv_depth=conv_depth, conv_width=conv_width, down=task,
                                           fc_depth=fc_depth, rnn_layers=rnn_layers)

        # self.loss_fn = self.cross_loss_wrapper(self.loss_fn)
        # self.loss_fn = self.mask_loss_wrapper(self.loss_fn)

    def create_net(self):
        if self.steps == -1:
            input_size = 121 if self.view == 'both' else 60
            out_size = 60 if self.view == 'both' else 30
        else:
            input_size = self.steps * 2 + 1 if self.view == 'both' else self.steps
            out_size = self.steps if self.view == 'both' else self.steps // 2

        # input_size = self.data_size if self.view == 'both' else 8 * 8 * 2 * 30
        self.net = PolicyNet(input_size, self.embedding_size, out_size, self.view, self.mask_ratio,
                             cor_dim=self.cor_dim, conv_depth=self.conv_depth, conv_width=self.conv_width,
                             fc_depth=self.fc_depth, rnn_layers=self.rnn_layers)

        # self.net = PolicyNet(input_size, self.embedding_size, out_size, self.view, 1,
        #                      cor_dim=1, conv_depth=1, conv_width=128,
        #                      fc_depth=2, rnn_layers=self.rnn_layers)

    @staticmethod
    def states_split(board_data, view_steps):
        """
        将棋盘数据划分为不同视角的数据
        :param board_data: 棋盘数据，shape:batch_size*121*8*8
        :param view_steps: 视角，ai or player or both
        :return: 划分后的数据
        """
        view, steps = view_steps
        if not isinstance(board_data, torch.Tensor):
            board_data = torch.tensor(board_data)

        if view == 'player':
            board_data = split_input(board_data, steps)[0]
        elif view == 'ai':
            board_data = split_input(board_data, steps)[1]
        return board_data

    @staticmethod
    def actions_split(pos_data, ai_first_list, view):
        """
        将下棋位置数据划分为不同视角的数据
        :param pos_data: 下棋位置向量 shape:batch_size*60
        :param ai_first_list: 棋局是否是AI先手的列表 shape:batch_size
        :param view:  视角，ai or player or both
        :return: 划分后的数据
        """
        if not isinstance(pos_data, torch.Tensor):
            pos_data = torch.tensor(pos_data)

        if view == 'both':
            return pos_data

        actions_data = torch.zeros(pos_data.shape[0], pos_data.shape[1] // 2, pos_data.shape[2])

        if view == 'player':
            for b in range(pos_data.shape[0]):
                actions_data[b] = pos_data[b, ai_first_list[b]::2]
        elif view == 'ai':
            for b in range(pos_data.shape[0]):
                actions_data[b] = pos_data[b, not ai_first_list[b]::2]
        return actions_data

    def load_data(self, data_file_states, data_file_actions, length=-1, pickle_file_states=None,
                  pickle_file_actions=None):
        # 读取两种数据，棋盘的图像作为状态，用户下棋的位置作为动作
        data_states = load_data_from_file(data_file_states, length, self.data_size, True, False, pickle_file_states,
                                          process_fun=self.states_split, process_fun_params=[self.view, self.steps],
                                          steps=self.steps)
        data_actions = load_data_from_file(data_file_actions, length, 60 if self.cor_dim == 1 else 120, False,
                                           False, pickle_file_actions, process_fun=self.actions_split,
                                           process_fun_params=self.view, steps=self.steps)

        print(data_states.shape)
        print(data_actions.shape)

        # data_actions = data_actions / 64  # 归一化，行为范围是从0～63
        data_actions = data_actions / (7 if self.cor_dim == 2 else 63)  # 归一化

        # 划分训练测试集
        train_index, test_index = split_index(0.8, len(data_states), shuffle=True)  # 划分数据集和训练集

        train_states, train_actions = data_states[train_index], data_actions[train_index]
        test_states, test_actions = data_states[test_index], data_actions[test_index]

        self.train_dataset = TensorDataset(train_states, train_actions)
        self.test_dataset = TensorDataset(test_states, test_actions)

        self.is_load_data = True
        print("Load data completed!")

    def evaluate(self, test_loader=None, batch_size=256, show_out=False):
        test_loader = super(Behavior2vec, self).evaluate(test_loader, batch_size)
        total_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.net(x)
                # print("pred.shape:", pred.shape)
                # print("y.shape:", y.shape)
                loss = self.loss_fn(pred, y)
                total_loss += loss.item()

                # ~ 查看输出
                if show_out:
                    # pred = torch.stack(pred, dim=1)
                    # print("pred:", torch.nn.functional.softmax(pred[0],dim=1).argmax(dim=1))
                    # print("x:",x[0])
                    # print(str("Y:" + str(y[0] * 7) + "\r\n Pred:" + str(
                    #     torch.nn.functional.softmax(pred[0], dim=1).argmax(dim=1))))
                    return str("Y:" + str(y[0] * 7) + "\r\n Pred:" + str(pred[0] * 7))

        return total_loss / len(test_loader)

    def train_epoch(self, train_loader):
        loss_per_epoch = 0
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)

            if self.teacher_force_ratio > 0:
                pred = self.net((x, y, self.teacher_force_ratio))
                # self.teacher_force_ratio = self.teacher_force_ratio - (self.teacher_force_ratio / 20)
            else:
                pred = self.net(x)

            loss = self.loss_fn(pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_per_epoch += loss.item()
        return loss_per_epoch

    @staticmethod
    def mask_loss_wrapper(loss_fn):
        def fun(pred, y):
            batch_size = y.shape[0]
            pred_list, mask = pred
            print("mask.shape:", mask.shape)
            print("y.shape:", y.shape)
            mask_y = torch.masked_select(y, mask).view(batch_size, -1)
            return loss_fn(pred, y)

        return fun

    @staticmethod
    def cross_loss_wrapper(loss_fn):

        def fun(pred, y):
            loss = 0
            y = y.to(torch.int64)
            for i in range(len(y[0])):
                # print("y[i].shape:", y[:, i].shape)
                # print("pred[i].shape:", pred[i].shape)
                loss += loss_fn(pred[i], y[:, i])  # b x 60 , b
            return loss

        return fun
