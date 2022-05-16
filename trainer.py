#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2021/3/12 下午2:50
# @Author  : PCF
# @Email   : pan.chaofan@foxmail.com
# @File    : trainer.py
# @Software: PyCharm
import os
from typing import Union, Tuple

import fitlog
import numpy as np
import torch
# 设置使用1号GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tqdm import trange
from visdom import Visdom

from data_process import BOARD_LINE_SIZE
from utils.utils import DataLoaderX, board_transform, load_data_from_file


class Trainer:
    def __init__(self, log_dir, net, data_shape: Union[int, list], learning_rate, weight_decay,
                 loss_fn, reduce_lr, embedding_size: int = 128, steps=-1, down='lv', **extra):
        """
        用来训练的类
        :param net: 神经网络
        :param data_shape: 单个用户单次对局的数据长度
        :param embedding_size: 嵌入向量大小
        :param learning_rate: 神经网络学习率
        """
        self.data_size = data_shape  # 每个用户每次对局的数据长度
        self.embedding_size = embedding_size
        self.steps = steps

        # 首先设定运行方式 为cpu
        self.device = torch.device('cpu')
        # 若gpu可行，则使用 gpu进行计算
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        print("Compute device: {}".format(self.device))

        self.__dict__.update(extra)  # 添加自定义的额外参数
        if down == 'res':
            self.teacher_force_ratio = 1
            self.mask_ratio = 1
            self.conv_width = 128
            self.fc_depth = 2

        # 初始化 模型 ，初始化模型得到参数 需在此前给出
        self.net = net
        if net is None:
            self.create_net()
        # 多GPU并行
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     self.net = nn.DataParallel(self.net)
        self.net = self.net.to(self.device)

        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        # 定义损失函数
        if loss_fn is not None:
            try:
                self.loss_fn = loss_fn.to(self.device)
            except AttributeError:
                self.loss_fn = loss_fn
        # self.loss_fn = nn.SmoothL1Loss()
        self.reduce_lr = reduce_lr  # 是否调整学习率
        if reduce_lr:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=3,
                                                                        verbose=True, min_lr=0.00005, cooldown=0)

        self.is_load_data = False
        self.train_dataset, self.test_dataset = None, None

        # log_dir = "logs/combine_logs"  # * 覆盖子类的设置，将日志同一记录在一个文件夹下
        log_dir = os.path.join("logs", log_dir)
        if not fitlog.is_debug() and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        fitlog.set_log_dir(log_dir, True)
        fitlog.create_log_folder()  # 创建日志目录

        # self.log_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d-%H%M"))
        # if not os.path.exists(self.log_dir):
        #     os.makedirs(self.log_dir)
        if not fitlog.is_debug():
            log_dir = fitlog.get_log_folder(True)
            self.model_dir = os.path.join(log_dir, 'model')  # 存储模型文件的目录
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)

    def create_net(self):
        """
            新建一个模型
        """
        raise NotImplementedError

    def load_data(self, data_file):
        """
        载入用户数据，从csv中导入数据，并保存在train_dataset与test_dataset中
        :param data_file: 预处理后的用户数据文件
        """
        raise NotImplementedError

    def train_init(self, use_vis):
        if not self.is_load_data:
            print("Please load data first.")
            raise ValueError

        vis = None
        if use_vis:
            # 可视化工具 实例化
            vis = Visdom()
            # 可视化需要先 开启 visdom 服务
            try:
                assert vis.check_connection()
            except AssertionError:
                print("Please start visdom server by 'python3 -m visdom.server'")
                raise ValueError
        return vis

    def train(self, batch_size=256, epoch=200, use_vis=False, train_sampler=None, test_sampler=None):
        """
        训练网络
        :param batch_size: 每个batch的大小
        :param epoch: 训练轮数
        :param use_vis: 是否使用visdom可视化训练过程
        :param train_sampler: 训练数据采样器
        :param test_sampler: 测试数据采样器
        :return 训练损失和测试损失
        """
        vis = self.train_init(use_vis)

        # 加载数据， train_loader 中的数据已经是分了批次的，shuffle 指定打乱数据，num_workers 指定读取数据线程为4
        # * 如果指定的批数据的采样器，则不能使用batch_size参数
        if train_sampler is None:
            train_loader = DataLoaderX(self.train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False,
                                       num_workers=4)
        else:
            train_loader = DataLoaderX(self.train_dataset, batch_sampler=train_sampler, pin_memory=False, num_workers=1)
        if test_sampler is None:
            test_loader = DataLoaderX(self.test_dataset, batch_size=batch_size, shuffle=True, pin_memory=False,
                                      num_workers=4)
        else:
            test_loader = DataLoaderX(self.test_dataset, batch_sampler=test_sampler, pin_memory=False, num_workers=1)

        losses, test_losses = [], []
        # with trange 可视化打印进度
        min_test_loss = 1000
        with trange(epoch) as t:
            for e in t:
                t.set_description("Train episode: %d" % e)  # 设置进度条题
                loss_per_epoch = self.train_epoch(train_loader)

                loss = loss_per_epoch / len(train_loader)  # 每个epoch的平均loss
                test_loss = self.evaluate(test_loader, batch_size)  # 计算测试集损失
                t.set_postfix(test_loss=test_loss, train_loss=loss)  # 进度条相关信息

                if self.reduce_lr:
                    self.scheduler.step(test_loss)
                losses.append(loss)
                test_losses.append(test_loss)

                if use_vis:
                    # 绘图
                    if e == 0:
                        vis.line([[loss, test_loss]], [e], win='train', update='replace',
                                 opts=dict(legend=["train_loss", "test_loss"], showlegend=True, xlabel='epoch',
                                           ylabel='mse'))
                    else:
                        vis.line([[loss, test_loss]], [e], win='train', update='append',
                                 opts=dict(legend=["train_loss", "test_loss"], showlegend=True, xlabel='epoch',
                                           ylabel='mse'))
                # 写入记录
                # self.writer.add_scalar("train/loss", loss, e)
                # self.writer.add_scalar("test/loss", test_loss, e)
                fitlog.add_loss(loss, name="Train_loss", step=e)
                fitlog.add_metric(test_loss, name="Test_loss", step=e)
                if test_loss < min_test_loss:
                    # 存储测试效果最好的模型
                    min_test_loss = test_loss
                    fitlog.add_best_metric(min_test_loss, "Min_test_loss")
                    if not fitlog.is_debug():
                        self.save_net(os.path.join(self.model_dir, 'best.pth'))
                # 存储最终的模型
                if not fitlog.is_debug():
                    self.save_net(os.path.join(self.model_dir, 'final.pth'))
        return losses, test_losses

    def train_epoch(self, train_loader):
        """从数据加载器训练所有数据一轮，返回这一轮的损失和"""
        raise NotImplemented

    def evaluate(self, test_loader, batch_size):
        """
            网络评估
            :param test_loader: 测试数据加载器，为空则根据batch_size创建
            :param batch_size: 创建的数据加载器每一个batch的大小
            :return 评估损失
        """
        if test_loader is None:
            test_loader = DataLoaderX(self.test_dataset, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, num_workers=4)

        return test_loader
        # * 子类需要计算测试损失值并返回

    def reset_params(self):
        """
        重置网络参数
        TODO 只需重置参数而不需要重新重设网络
        """
        self.create_net()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

    # 暂未使用
    def empty_data(self):
        """
        清空模型数据
        :return:
        """
        self.train_dataset, self.test_dataset = None, None
        self.is_load_data = False

    def save_net(self, path='weights/module.pth'):
        """
        存储网络参数，以.pth 存储模型
        :param path: 参数文件路径
        """
        torch.save(self.net.state_dict(), path)

    def load_net(self, path='weights/module.pth'):
        """
        加载网络参数
        :param path: 参数文件路径
        """
        new_state_dict = torch.load(path, map_location=torch.device('cpu') if not torch.cuda.is_available() else None)
        self.net.load_state_dict(new_state_dict)
        print('Loaded net successfully')

    def pos2vec(self, data_file, length, pickle_file: Tuple[str, str] = None):
        # """将文件中length行数据转换为向量"""
        if pickle_file is not None and os.path.exists(pickle_file[0]):
            with open(pickle_file[0], 'rb') as f:
                out_data = torch.load(f)
            with open(pickle_file[1], 'rb') as f:
                out_labels = np.load(f)
            print("Pickle load data successfully.")
            vector = self.net.embedding(out_data.to(self.device)).cpu().detach().numpy()
            return vector, out_labels

        data = load_data_from_file(data_file, 100000, self.data_size, True, True, None, include_player=True,
                                   steps=self.steps)
        indices = np.random.permutation(len(data))
        data = data[indices]  # shuffle

        all_labels = []
        group = []
        out_labels = np.zeros(length, np.uint8)
        out_data = np.zeros((length, data.shape[1] - 2))
        pos = 0

        for i, d in enumerate(data):
            lv, id = d[-1], d[-2]

            if self.view == 'player':
                if id not in all_labels:
                    all_labels.append(id)
                    group.append([])
                id_idx = all_labels.index(id)
                group[id_idx].append(i)
            else:
                if lv not in all_labels:
                    all_labels.append(lv)
                    group.append([])
                lv_idx = all_labels.index(lv)
                group[lv_idx].append(i)

        label_num = len(group)
        print(f"There are {label_num} class.")

        num_per_label = length // label_num

        idx_in_group = 0
        l = 0
        while True:
            for indices in group:
                if idx_in_group >= len(indices):
                    continue
                out_data[l] = data[indices[idx_in_group]][:-2]
                if self.view == 'player':
                    out_labels[l] = data[indices[idx_in_group]][-2]
                else:
                    out_labels[l] = data[indices[idx_in_group]][-1]
                l += 1
                if l >= length:
                    break
            else:
                idx_in_group += 1
                continue
            break

        out_data = torch.from_numpy(out_data)
        if pickle_file is not None:
            with open(pickle_file[0], 'wb') as f:
                torch.save(out_data, f)
            with open(pickle_file[1], 'wb') as f:
                np.save(f, out_labels)
        vector = self.net.embedding(out_data.to(self.device)).cpu().detach().numpy()
        return vector, out_labels
