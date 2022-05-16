#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2022/1/11 下午8:18
# @Author  : Chaofan Pan
# @Email   : pan.chaofan@foxmail.com
# @File    : board_plot.py
# @Software: PyCharm
from typing import Union

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt

BOARD_CHANNEL = 2
BOARD_SIZE = 8


def board_img_array(board_data: Union[torch.Tensor, np.ndarray]):
    """处理棋盘图像数组"""

    # 数据形状检查
    if len(board_data.shape) == 1:
        # 数据是向量时尝试转换为张量
        try:
            board_data = board_data.reshape(-1, BOARD_CHANNEL, BOARD_SIZE, BOARD_SIZE)
        except ValueError:
            print("Data shape not suite.", board_data.shape)
            return
    elif len(board_data.shape) == 4:
        # 数据是张量时验证各个维度的大小
        try:
            assert board_data.shape[-1] == BOARD_SIZE and board_data.shape[-2] == BOARD_SIZE and board_data.shape[
                -3] == BOARD_CHANNEL
        except AssertionError:
            print("Data shape not suite.", board_data.shape)
            return
    else:
        board_data = board_data.view(-1, BOARD_CHANNEL, BOARD_SIZE, BOARD_SIZE)

    if isinstance(board_data, torch.Tensor):
        board_data = board_data.detach().cpu().numpy()
    length = board_data.shape[0]  # 对局长度

    # 存储绘图的矩阵
    img_array = np.zeros((length, BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)

    for i in range(length):
        img_array[i][np.where(board_data[i][0] == 1)] = 1  # 用户棋子表示为黑子
        img_array[i][np.where(board_data[i][1] == 1)] = 2  # AI棋子表示为白子

    assert 2 >= img_array.all() >= 0  # 只能有0,1,2三种数
    return img_array


def board_plot(img_array: np.ndarray):
    """绘制棋盘图像"""

    # * 通过数组绘制图片的方式来画棋盘，数组每个值表示一个颜色
    color1 = (247 / 255, 220 / 255, 111 / 255)  # 背景色
    color2 = (1, 1, 1)  # 黑子颜色
    color3 = (0, 0, 0)  # 白子颜色
    color4 = (0.5, 0.5, 0.5)  # 棋盘分割线颜色

    # 最多有60步棋，因此使用6行5列来显示棋盘 10 6
    mat = np.zeros((8 * 10 + 9, 8 * 6 + 5))

    for col in range(6):
        for row in range(10):
            mat[row * 9:row * 9 + 8, col * 9:col * 9 + 8] = img_array[row * 6 + col]

    # 绘制棋盘分割线
    mat[:, [i * 8 + i - 1 for i in range(1, 6)]] = 4
    mat[[i * 8 + i - 1 for i in range(1, 10)], :] = 4

    # 数组中有四种元素，显示成四种颜色，通过颜色映射函数进行绘图
    my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('my_camp', [color1, color2, color3, color4], 4)
    cs = plt.imshow(mat, cmap=my_cmap)

    # plt.xticks(np.linspace(0, 8, 8, endpoint=False), ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'), fontsize=20)
    # plt.yticks(np.linspace(0, 8, 8, endpoint=False), ('1', '2', '3', '4', '5', '6', '7', '8'), fontsize=20)
    plt.tick_params(bottom=False, left=False, labeltop=True, labelright=True)
    plt.show()
