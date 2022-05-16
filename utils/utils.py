#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/12/26 下午5:10
# @Author  : PCF
# @Email   : pan.chaofan@foxmail.com
# @File    : utils.py
# @Software: PyCharm
import os
import random
import struct

import numpy
import numpy as np
import torch
from matplotlib import pyplot as plt
from prefetch_generator import BackgroundGenerator
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from data_process import BOARD_LINE_SIZE, LINE_SIZE
from utils.board_plot import board_img_array, board_plot


def split_index(train_ratio, total_length, shuffle=False):
    """
    划分数据为训练集，验证集和测试集
    @param train_ratio: 训练集的比例
    @param total_length: 序列总长度
    @param shuffle: 是否打乱
    """
    # index=np.random.permutation(total_length)
    # Data shuffled by preprocessing during the generation
    if shuffle:
        index = np.random.permutation(total_length)  # 打乱数据
    else:
        index = np.array(range(total_length))
    # Get split
    train_num = int(total_length * train_ratio)
    train_index = index[:train_num]
    test_index = index[train_num:total_length]
    return train_index, test_index


class DataLoaderX(DataLoader):
    """
    更好的数据载入器
    """

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def loss_plot(epoch_num, train_losses, test_losses, save_path='img/plot.pdf'):
    """
    绘制训练误差和测试误差的变化图
    @param epoch_num: 训练轮数
    @param train_losses: 每轮的训练误差列表
    @param test_losses: 每轮的测试误差列表
    @param save_path: 存储结果图的位置
    """
    epochs = [i for i in range(epoch_num)]
    train_losses = np.sqrt(train_losses)
    test_losses = np.sqrt(test_losses)
    plt.plot(epochs, train_losses, label='train loss')
    plt.plot(epochs, test_losses, label='test loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def average_loss_plot(trainer, file, epoch_num=130, average_num=50):
    """
    绘制平均loss变化图
    :param epoch_num:训练的epoch数量和
    :param average_num:进行多少次实验求平均值
    @return:
    """
    total_losses, total_test_losses = numpy.zeros(epoch_num), numpy.zeros(epoch_num)
    trainer.load_data(data_file=file)

    # 多次训练取得平均值
    for i in range(average_num):
        print('Experiment number: {}:'.format(i))
        trainer.reset_params()
        losses, test_losses = trainer.train(epoch=epoch_num, use_vis=False)
        total_losses += numpy.array(losses)
        total_test_losses += numpy.array(test_losses)

    total_losses /= average_num
    total_test_losses /= average_num

    loss_plot(epoch_num, total_losses, total_test_losses)


def pos2num(pos, keep_dim=False):
    """将位置转换为数字0-63"""
    assert len(pos) == 2
    row = pos[0] - int.from_bytes(b'a', byteorder='big')
    col = pos[1] - int.from_bytes(b'1', byteorder='big')
    if not keep_dim:
        return row * 8 + col
    else:
        return row, col


def pos2nums(pos, keep_dim=False):
    """将操作序列转换为0-63的数字序列"""
    pos_nums = np.empty(len(pos) // 2) if not keep_dim else np.empty(len(pos))
    for i in range(0, len(pos), 2):
        if not keep_dim:
            pos_nums[i // 2] = pos2num(pos[i:i + 2])
        else:
            pos_nums[i:i + 2] = pos2num(pos[i:i + 2], keep_dim)
    return pos_nums


def plot_tsne(data, color, file_path=""):
    """绘制t-sne图"""
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(data)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], label='t-SNE', c=color, cmap='rainbow')
    plt.axis('off')
    plt.savefig(file_path + "t-sne.eps")
    plt.show()


def load_data_from_file(data_file, length, data_size=8 * 8 * 2 * 60 + 64, board_data_type=False, include_ai=False,
                        pickle_file=None, include_player=False, process_fun=None, group_by_user=False,
                        save_before_process=False, process_fun_params=None, shuffle=False, steps=-1):
    """
    从文件载入数据，可以利用上一次载入过的数据缓存快速载入
    :param data_file: 数据文件
    :param length: 数据长度，-1表示文件大小
    :param data_size: 每条数据的大小
    :param board_data_type: 是否转换为棋盘数据
    :param include_ai: 是否添加AI级别信息
    :param include_player: 是否包含用户信息
    :param pickle_file: 序列化的缓存文件位置
    :param process_fun: 读取数据后的预处理函数  numpy -> tensor
    :param group_by_user: 是否按用户id进行分组，否则按ai级别进行分组
    :param save_before_process: 是否在预处理之前缓存序列化数据，否则缓存处理后的数据 TODO
    :param process_fun_params: 预处理函数的额外参数
    :param shuffle: shuffle the output data order
    :param steps: 使用前多少步 -1使用所有步
    :return: torch类型的数据
    """
    keep_vec_dim = True
    if include_player:
        # ~ 需要记录用户信息时则同时记录AI信息，级别代表不同AI，id代表不同用户
        include_ai = True
    if not include_player and include_ai:
        group_by_user = False

    if steps != -1 and pickle_file is not None:
        pickle_file = '_'.join([pickle_file[:-3], "step", str(steps), pickle_file[-3:]])

    try:  # 载入数据较为耗时，因此设计缓存机制，将数据序列化，第二次直接可以使用序列化的文件
        if (length > 10000 or length <= -1) and pickle_file is not None and os.path.exists(pickle_file):
            # 数据长度小于10000没必要用该机制节约时间
            with open(pickle_file, 'rb') as f:
                data = torch.load(f)
                if shuffle:
                    indices = list(range(len(data)))
                    random.shuffle(indices)
                    data = data[indices]
            print("Pickle load data successfully.")
            return data
    except IOError as e:
        print("Pickle load error:", e)

    # print("Start data reading...")
    size = os.path.getsize(data_file)
    line_size = BOARD_LINE_SIZE if board_data_type else LINE_SIZE
    if length <= 0:
        length = size // line_size

    user_ids = []
    ai_lvs = []
    groups = []  # 记录每个组的id对应的所有位置索引
    ai_first_list = []

    if steps == -1:
        out_data_size = data_size
    else:
        if data_size == 60:
            out_data_size = steps
        else:
            out_data_size = 8 * 8 * 2 * steps + 64

    if include_player:
        data = np.zeros((length, out_data_size + 2), np.float32)
    else:
        data = np.zeros((length, out_data_size + 1 if include_ai else out_data_size), np.float32)
    with open(data_file, 'rb') as f:
        for i in range(length):
            line = f.read(line_size)
            if board_data_type:  # 使用棋盘作为输入
                if include_player:
                    ad_id, line_data, ai_level = board_transform(line, data_size, reserve_id=True)  # 监督数据有标签
                else:
                    line_data, ai_level = board_transform(line, data_size)  # 监督数据有标签
                if steps != -1:
                    # 截取前s步数据
                    line_data = np.concatenate([line_data[:8 * 8 * 2 * steps], line_data[-64:]])

            else:  # 使用向量作为输入
                ad_id, ai_first, pos, ai_level, black_score, white_score = struct.unpack('36s?120sHHH', line)
                line_data = pos2nums(pos, keep_dim=True if data_size == 120 else False)
                ai_first_list.append(ai_first)
                if steps != -1:
                    line_data = line_data[:steps]

            if include_player:
                if ad_id not in user_ids:
                    user_ids.append(ad_id)
                    if group_by_user:
                        groups.append([])
                if ai_level not in ai_lvs:
                    ai_lvs.append(ai_level)
                    if not group_by_user:
                        groups.append([])

                id_idx = user_ids.index(ad_id)
                lv_idx = ai_lvs.index(ai_level)
                data[i][:-2] = line_data
                data[i][-2] = id_idx
                data[i][-1] = lv_idx
                if group_by_user:
                    groups[id_idx].append(i)
                else:
                    groups[lv_idx].append(i)

            elif include_ai:
                data[i][:-1] = line_data
                data[i][-1] = ai_level
            else:
                data[i] = line_data

            if i % 10000 == 0:
                print("Load data {} lines.".format(i))

    # ---数据预处理
    data = torch.from_numpy(data)
    if not board_data_type:
        if data_size == 120:
            data = data.view(data.shape[0], steps, 2)  # cow, rol
        elif data_size == 60:
            data = data.view(data.shape[0], steps, 1)
    if process_fun is not None:
        if not board_data_type:
            data = process_fun(data, ai_first_list, process_fun_params)
        elif include_player:
            data = process_fun(data, groups, process_fun_params)
        else:
            data = process_fun(data, process_fun_params)

    # ---序列化数据
    try:
        if (length > 10000 or length <= -1) and pickle_file is not None:
            with open(pickle_file, 'wb') as f:
                torch.save(data, f)
    except IOError as e:
        print("Pickle dump error:", e)

    if shuffle:
        indices = list(range(length))
        random.shuffle(indices)
        data = data[indices]
    return data


def board_transform(line, data_size, reserve_id=False):
    """
    将数据转换为棋盘格式
    :param line: 每一局棋的原始数据，表示为二进制的字符串
    :param data_size: 每一行原始数据的大小
    :param reserve_id: 是否保留用户id
    :return: 棋盘格式的数据和AI的级别 (以及用户id)
    """
    board_data = np.zeros(data_size)
    ad_id, *states, ai_level = struct.unpack('36s121Qi', line)
    for j in range(121):
        board_str = bin(states[j])[2:]
        start = 8 * 8 - len(board_str)
        for k in range(len(board_str)):
            if board_str[k] == '1':
                board_data[j * 8 * 8 + start + k] = 1
    if reserve_id:
        return ad_id, board_data, ai_level
    else:
        return board_data, ai_level


def flip180(arr):
    """
    旋转numpy矩阵180度
    @param arr: 选择的二维矩阵
    @return: 旋转后的二维矩阵
    """
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    return new_arr


def flip90_left(arr):
    """
    逆时针旋转numpy矩阵90度
    @param arr: 选择的二维矩阵
    @return: 旋转后的二维矩阵
    """
    new_arr = np.transpose(arr)
    new_arr = new_arr[::-1]
    return new_arr


def flip90_right(arr):
    """
    顺时针旋转numpy矩阵90度
    @param arr: 选择的二维矩阵
    @return: 旋转后的二维矩阵
    """
    new_arr = arr[::-1]
    new_arr = np.transpose(new_arr)
    return new_arr


def board_rotate(data, angle):
    """
    顺时针旋转棋盘
    :param data:原始数据
    :param angle:旋转角度
    """
    if angle == 0:
        return data
    if angle not in [90, 180, 270]:
        print("Error: 不支持的旋转角度!")
        exit(1)

    # 去掉数据尾的级别信息
    # if data.shape[0] != 121 * 8 * 8:
    #     data = data[:-1]
    data = data.numpy()
    # 处理数据为矩阵形式
    data = data.reshape(-1, 8, 8)
    board_num = data.shape[0]
    # 额外生成的其他三种数据
    data_flip = np.zeros(data.shape, dtype=np.int32)
    # 对每个棋盘进行选择
    for i in range(board_num):
        if angle == 90:
            data_flip[i, :] = flip90_right(data[i, :])
        elif angle == 180:
            data_flip[i, :] = flip180(data[i, :])
        else:
            data_flip[i, :] = flip90_left(data[i, :])
    # 恢复原来的数据形状
    data = data.flatten()
    data_flip = data_flip.flatten()

    assert len(data_flip) == 8 * 8 * board_num
    return torch.tensor(data_flip)


def tensor_mask(x, ratio):
    """
    数据遮罩，随机遮住某些数据

    :param x: 输入数据
    :param ratio: 遮住率
    :return: 遮罩掩码
    """
    batch_size = x.shape[0]
    length = x.shape[1]
    masked_num = int(length * ratio)  # 最后得到的数据序列长度

    bool_mask = torch.empty((batch_size, length), dtype=torch.bool)
    for b in range(batch_size):
        mask = np.hstack([
            np.zeros(length - masked_num),
            np.ones(masked_num)
        ])
        np.random.shuffle(mask)
        mask = torch.from_numpy(mask)
        bool_mask[b] = mask.to(torch.bool)
    return bool_mask.view(batch_size, length, 1, 1).to(x.device)


def nt_xent_loss(batch_data_aug1, batch_data_aug2, hidden_norm=False, temperature=1.0):
    device = batch_data_aug1.device
    batch_size = batch_data_aug1.shape[0]
    assert batch_size == batch_data_aug2.shape[0]
    if hidden_norm:
        batch_data_aug1 = torch.norm(batch_data_aug1, 2, dim=-1)
        batch_data_aug2 = torch.norm(batch_data_aug2, 2, dim=-1)
    labels = torch.arange(0, batch_size).to(device)
    masks = torch.nn.functional.one_hot(torch.arange(0, batch_size), batch_size).to(device)
    # labels = tf.one_hot(tf.range(batch_size), batch_size * 2)  # [batch_size,2*batch_size]
    # masks = tf.one_hot(tf.range(batch_size), batch_size)  # [batch_size,batch_size]
    logits_aa = torch.matmul(batch_data_aug1, batch_data_aug1.T) / temperature  # [batch_size,batch_size]
    logits_bb = torch.matmul(batch_data_aug2, batch_data_aug2.T) / temperature  # [batch_size,batch_size]
    # logits_aa = torch.matmul(batch_data, batch_data, =True) / temperature  # [batch_size,batch_size]
    # logits_bb = torch.matmul(batch_data_aug, batch_data_aug, transpose_b=True) / temperature  # [batch_size,batch_size]
    logits_aa[masks == 1] = -torch.inf  # remove the same samples in batch_data
    logits_bb[masks == 1] = -torch.inf  # remove the same samples in batch_data_aug
    # logits_aa = logits_aa - masks * INF
    # logits_bb = logits_bb - masks * INF
    logits_ab = torch.matmul(batch_data_aug1, batch_data_aug2.T) / temperature
    logits_ba = torch.matmul(batch_data_aug2, batch_data_aug1.T) / temperature
    loss_a = torch.nn.functional.cross_entropy(torch.cat([logits_ab, logits_aa], 1), labels)
    loss_b = torch.nn.functional.cross_entropy(torch.cat([logits_ba, logits_bb], 1), labels)
    # loss_a = tf.losses.softmax_cross_entropy(
    #     labels, tf.concat([logits_ab, logits_aa], 1))
    # loss_b = tf.losses.softmax_cross_entropy(
    #     labels, tf.concat([logits_ba, logits_bb], 1))
    loss = loss_a + loss_b
    return loss, logits_ab


# def get_actions(boards):
#     """
#     得到状态下对应的下一步行为
#     :param boards: batch*board_seqs*action_space
#     :return: 对应行为, batch*batch_seqs*1
#     """
#     for step in range(boards.shape[1]):
#

def calc_result(board_data):
    """
    计算棋盘结果
    :param board_data:
    :return: 1(先手/黑方胜利), -1(后手/白方胜利) or 0(平局)
    """
    assert board_data.shape == (121, 8, 8) or board_data.shape[0] == 121 * 8 * 8
    if board_data.dim() == 1:
        board_data = board_data.reshape(121, 8, 8)

    # * 先手是黑，但数据体中白色为第一层
    final_board_black = board_data[119]
    final_board_white = board_data[118]
    num_pieces_black = torch.sum(final_board_black)
    num_pieces_white = torch.sum(final_board_white)
    try:
        assert num_pieces_black + num_pieces_white == 64
    except AssertionError:
        array = board_img_array(board_data[:-1])
        board_plot(array)
        print("final_board_white:", board_data[0], board_data[1])
        print("num_pieces_first:", num_pieces_black)
        print("num_pieces_second:", num_pieces_white)
        exit(1)

    if num_pieces_black > num_pieces_white:
        return 0
    elif num_pieces_black < num_pieces_white:
        return 1
    else:
        return 2

def split_input(x, steps=-1):
    # 将输入划分为用户和AI的各60张图，划分依据是最后一张图
    new_steps = 60 if steps == -1 else steps
    x = x.reshape(-1, new_steps * 2 + 1, 64)

    device = x.device
    user_first = torch.all(x[:, -1] == 0, dim=-1)
    user_x, ai_x = torch.zeros(len(x), new_steps * 8 * 8, device=x.device), torch.zeros(len(x), new_steps * 8 * 8,
                                                                                        device=x.device)
    for i in range(len(x)):  # TODO: remove loop
        user_x[i] = x[i, list(range([1, 0][user_first[i]], new_steps * 2, 2))].reshape(new_steps * 8 * 8)
        ai_x[i] = x[i, list(range([0, 1][user_first[i]], new_steps * 2, 2))].reshape(new_steps * 8 * 8)
    return user_x.to(device), ai_x.to(device)
