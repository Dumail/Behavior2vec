#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2021/12/15 下午8:43
# @Author  : PCF
# @Email   : pan.chaofan@foxmail.com
# @File    : predictor.py
# @Software: PyCharm
import os

import fitlog
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

from nets.combine_net import CombineNet
from nets.lv_net import SingleNet
from trainer import Trainer
from utils.utils import load_data_from_file, split_index, calc_result

LEVEL_RANGE = 60


class Predictor(Trainer):
    """通过确定的监督任务对迁入网络进行评估"""

    def __init__(self, data_shape, board_data_type=True, embedding_size=128, net=None, learning_rate=0.001,
                 weight_decay=0, loss_fn=nn.MSELoss(), reduce_lr=False, embedding_net=None, log_dir="Predictor",
                 is_fine_tuning=False, is_predict_result=False, steps=-1, view='player'):
        super().__init__(log_dir, net, data_shape=data_shape, learning_rate=learning_rate,
                         is_predict_result=is_predict_result, steps=steps, view=view,
                         is_fine_tuning=is_fine_tuning, embedding_size=embedding_size, weight_decay=weight_decay,
                         loss_fn=loss_fn, reduce_lr=reduce_lr, embedding_net=embedding_net)
        # self.embedding_net = embedding_net  # 嵌入网络，输入原始数据输出向量
        self.data_shape = data_shape
        self.board_data_type = board_data_type  # 是否使用棋盘图像数据，否则只使用操作序列
        if is_predict_result:
            self.loss_fn = nn.CrossEntropyLoss()

    def create_net(self):
        if self.embedding_net is not None:
            if not self.is_fine_tuning:
                self.net = SingleNet(self.embedding_size, predict_prob=self.is_predict_result)
            else:
                self.net = CombineNet(self.embedding_net,
                                      SingleNet(self.embedding_size, predict_prob=self.is_predict_result))

    def transform_data(self, data, batch_size=256):
        """通过嵌入网络将数据转换为对应的向量"""
        data = data
        embedded_data = torch.zeros(len(data), self.embedding_size + 1)
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                # * 拆分为数据和标签两部分，前者通过嵌入网络转换，转换后与后者重组为新数据
                embedded_data[i:i + batch_size, :-1] = self.embedding_net.embedding(
                    data[i:i + batch_size, :-1].to(self.device)).detach().cpu()
                embedded_data[i:i + batch_size, -1] = data[i:i + batch_size, -1]
        return embedded_data

    @staticmethod
    def calc_results(board_data, reverse_result=False):
        assert board_data.dim() == 2 and board_data.shape[1] == 121 * 8 * 8
        batch_size = board_data.shape[0]
        results = torch.zeros(batch_size, 1)

        for b in range(batch_size):
            results[b] = calc_result(board_data[b])

        # 反转结果
        if reverse_result:
            results = results * -1

        board_data = torch.cat([board_data, results], dim=1)
        return board_data

    @staticmethod
    def steps_board_data(board_data, steps):
        # 截取前几步的数据
        assert board_data.dim() == 2 and board_data.shape[1] == 121 * 8 * 8 + 1
        if steps == -1: return
        board_data = torch.cat([board_data[:, :8 * 8 * 2 * steps], board_data[:, -65:]], dim=1)
        return board_data

    def load_data(self, data_file, length=-1, pickle_file=None, pickle_embedding=False, train_ratio=0.8):
        """
        载入训练数据,两种缓存机制：缓存处理后的数据或缓存嵌入后的数据
        :param data_file: 原始数据文件
        :param length: 数据条数
        :param pickle_file: 缓存文件，根据pickle_embedding的值表示不同的缓存文件
        :param pickle_embedding: 是否缓存嵌入后的数据，当嵌入网络在改变时应该设置为False
        :param train_ratio: 训练数据在数据集中的占比
        """
        if pickle_embedding and (pickle_file is None or self.embedding_net is None):
            raise AttributeError  # 参数矛盾

        if pickle_embedding and os.path.exists(pickle_file):
            # 尝试使用嵌入向量的缓存文件
            with open(pickle_file, 'rb') as f:
                data = torch.load(f)
                print("Pickle load data successfully.")

        else:
            if not self.is_predict_result:
                data = load_data_from_file(data_file, length, self.data_size, self.board_data_type, True,
                                           pickle_file if not pickle_embedding else None)  # 缓存嵌入向量时不缓存数据
            else:
                data = load_data_from_file(data_file, length, self.data_size, self.board_data_type, False,
                                           pickle_file if not pickle_embedding else None, process_fun=self.calc_results,
                                           process_fun_params=False)
                data = self.steps_board_data(data, self.steps)
            if self.embedding_net is not None:
                if not self.is_fine_tuning:
                    data = self.transform_data(data)
                    if pickle_embedding:
                        # 将嵌入向量缓存到文件
                        with open(pickle_file, 'wb') as f:
                            torch.save(data, f)

        # 划分训练测试集
        train_index, test_index = split_index(0.8, len(data), shuffle=True)  # 划分数据集和训练集

        train_num = int(len(data) * train_ratio)
        train_index = train_index[:train_num]

        x_train = data[train_index][:, :-1]
        x_test = data[test_index][:, :-1]

        y_train = data[train_index][:, -1:]
        y_test = data[test_index][:, -1:]

        if self.is_predict_result:
            y_train = torch.squeeze(y_train.to(torch.int64))  # 类别编号 0是黑胜，1是白胜
            y_test = torch.squeeze(y_test.to(torch.int64))
            print(torch.sum(y_train == 0))
            print(torch.sum(y_train == 1))
            print(torch.sum(y_train == 2))
        # else:
        #     y_train = y_train / LEVEL_RANGE  # 级别归一化
        #     y_test = y_test / LEVEL_RANGE

        self.train_dataset = TensorDataset(x_train, y_train)
        self.test_dataset = TensorDataset(x_test, y_test)

        self.is_load_data = True
        print("Load data completed!")

    def train_epoch(self, train_loader):
        loss_per_epoch = 0  # 每个epoch的总loss
        for x, y in train_loader:
            # 取出当前批次的特征x， 该批特征对应的序列长度列表l， lv为下一场对局的AI等级， y为预测分数
            # x, y = batch
            # 拷贝数据到运行设备端
            x, y = x.to(self.device), y.to(self.device)
            pred = self.net(x)  # 利用模型module预测输出

            # print(pred[0].max(),pred[0].min())

            loss = self.loss_fn(pred, y)  # 计算训练集损失
            self.optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            self.optimizer.step()  # apply gradients
            loss_per_epoch += loss.item()  # 将该批次的损失累加
        return loss_per_epoch

    def evaluate(self, test_loader=None, batch_size=256):
        test_loader = super(Predictor, self).evaluate(test_loader, batch_size)
        total_loss = 0  # 初始化 0
        accuracy = 0
        with torch.no_grad():  # 此作用域下的张量计算不会被反向传播
            for x, y in test_loader:
                # x, y = self.test_dataset.tensors  # 取出测试集数据，当前所有的 特征x， 该批 特征对应的序列长度列表 l， lv为下一场对局的AI等级， y 为预测分数
                x, y = x.to(self.device), y.to(self.device)  # 拷贝数据到运行设备端
                pred = self.net(x)  # 模型进行预测
                loss = self.loss_fn(pred, y)  # 计算损失，均方差
                total_loss += loss.item()  # 取出损失的数值

                # print("x:", x)
                # print("y:", y)
                # print("pred:", pred.argmax(dim=1))
                tmp = pred.argmax(dim=1)

                # 当任务是预测结果时，可以计算出准确率
                if self.is_predict_result:
                    accuracy += torch.sum(pred.argmax(dim=1) == y).cpu().item() / len(y)

        if self.is_predict_result:
            return accuracy / len(test_loader)
            # TODO 修改其他位置的test_loss为accuracy
        else:
            return total_loss / len(test_loader)

    @staticmethod
    def evaluate_predict(data_file, embedding_net, embedding_size, weight_file, epoch, pickle_file=None,
                         is_fine_tuning=False, data_length=-1, train_ratio=0.8, is_predict_result=False, steps=-1):
        """
        通过预测任务评估嵌入结果
        :param data_file:  原始数据文件
        :param embedding_net:  嵌入网络，需要实现embedding方法
        :param embedding_size: 嵌入后的向量大小
        :param weight_file: 嵌入网络的网络参数文件
        :param epoch: 单层全连接层的训练次数
        :param pickle_file: 数据缓存文件路径
        :param is_fine_tuning: 是否通过fine tuning方式进行评估，否则使用linear probing
        :param data_length: 用于评估的数据条数
        :param train_ratio: 训练数据在总数据集中的占比
        :param is_predict_result: 是否通过预测结果来评估，否则是预测胜负
        :return: 在测试集上的预测损失
        """
        method = "fine tuning" if is_fine_tuning else "linear probing"
        print(f">>>>>>>>>>Start {method}>>>>>>>>>>")
        fitlog.debug(flag=True)
        if weight_file is not None:
            embedding_net.load_state_dict(
                torch.load(weight_file, map_location=torch.device('cpu') if torch.cuda.is_available() else None))
        predictor = Predictor(data_shape=8 * 8 * 2 * 60 + 64, embedding_size=embedding_size, board_data_type=True,
                              embedding_net=embedding_net, is_fine_tuning=is_fine_tuning,
                              is_predict_result=is_predict_result, steps=steps)
        predictor.load_data(data_file, pickle_file=pickle_file, length=data_length, train_ratio=train_ratio)
        loss = predictor.train(epoch=epoch)[1][-1]
        fitlog.debug(flag=False)
        print(f"<<<<<<<<<<End {method}<<<<<<<<<<")
        return loss
