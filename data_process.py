#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2021/8/1 下午7:27
# @Author  : PCF
# @Email   : pan.chaofan@foxmail.com
# @File    : data.py
# @Software: PyCharm
import json
import logging
import os
import struct
import traceback
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

import board

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(ch)
LINE_SIZE = 164
BOARD_LINE_SIZE = 1012


def json_line_process(line: str, line_no: int):
    """
    处理数据为更加精简的形式，去掉无用的数据
    @param line:  数据行
    @param line_no: 数据行号
    @return: 二进制数据行
    """
    res = None
    # 第1个过滤条件: 过滤掉 人-人 棋局，即不是Person VS AI(PVA)的暂定为无效
    if not line.__contains__('"IsAI": true,'):
        logger.debug(f"Line {line_no}: No AI, skip")
        return res

    # 第2个过滤条件: 过滤掉 非法的行，即 不是json格式的行，或者有问题的行
    # json.loads(line)报错， 或者load完了返回的js为None, 说明此行是烂行
    # sj为解析好的json对象
    sj = json.loads(line)
    if sj is None:
        logger.debug(f"Line {line_no}: Bad line, skip")
        return res

    # 第4个过滤条件：用户AdvertisingId为空的剔除掉
    advertising_id: str = sj["data"]["AdvertisingId"]  # 用户的id号
    if advertising_id == "":
        logger.debug(f"Line {line_no}: AdvertisingId empty, skip")
        return res

    # 过滤掉未完成的棋局
    if sj['data']['BlackScore'] + sj['data']['WhiteScore'] != 64:
        logger.debug(f"Line {line_no}: Game not completed.")
        return res

    # 拿到所有的AllOperations，用opts接收
    opts = sj["data"]["AllOperations"]  # opts为一个list,包含每一步
    ai_first = opts[0]['IsAI']  # 用于判断第一步是用户下棋 还是 AI下棋
    pos: str = sj["data"]["Pos"]  # 棋盘棋谱
    ai_level: int = sj["data"]["AILevel"]  # 该盘棋ai的等级
    is_tips = False
    is_repentance = False
    black_score = sj['data']['BlackScore']
    white_score = sj['data']['WhiteScore']

    if ai_level < 0 or ai_level > 59:
        return res

    # 过滤掉未完成的棋局
    if black_score + white_score != 64:
        logger.debug(f"Line:{line_no}: Game not completed.")
        return res

    for opt in opts:
        if opt["OperateType"] == 1:
            is_repentance = True
        elif opt["OperateType"] == 2:
            is_tips = True

    if is_tips or is_repentance:
        return res

    # 拼接结果
    # res = f"{advertising_id},{is_tips},{is_repentance},{ai_first},{pos},{ai_level}"
    res = struct.pack('36s?120sHHH', advertising_id.encode('utf-8'), ai_first,
                      pos.encode('utf-8'), ai_level, black_score, white_score)
    return res


def json_files_process(json_path, outfile='data/pos_data', append=False):
    """
    处理json文件，将其保存为二进制数据文件
    @param json_path: json文件目录
    @param outfile: 输出数据文件
    @param append: 是否追加到数据文件
    """
    # 每次处理覆盖旧文件内容
    out = open(outfile, 'ab' if append else 'wb')
    data = set()

    for root, dirs, file_name in os.walk(json_path):
        # root 表示当前正在访问的文件夹路径; dirs 表示文件夹下的子目录名list; file_name 表示该文件下的文件名list
        for i in range(0, len(file_name)):  # 遍历文件
            if '.json' in file_name[i]:
                # 对一个文件进行处理
                infile = root + '/' + file_name[i]
                print(f"File {infile} processing...")

                line_no = 0  # 设置line_no初始值

                valid_game_count = 0  # 用于统计有效的棋盘数, 初始化为0

                with open(infile, mode='r', encoding='utf-8') as fs:
                    while True:
                        try:
                            # 读取一行
                            line_no += 1
                            line = fs.readline()

                            # 如过行数不存在或者该行为空读取文件结束，跳出循环
                            if line is None or line == "":
                                break

                            # 调用deal_one_line对一行数据进行处理
                            new_line = json_line_process(line, line_no)

                            # 如果返回的new_line是None或者空字符串，则不会写入文件中，继续处理下一行
                            if not new_line:
                                continue

                            # 将处理好的一行数据加入数据集合
                            data.add(new_line)
                            valid_game_count += 1

                        except Exception:
                            # 以error级别打印异常日志
                            print("Line {}: {}".format(line_no, traceback.format_exc()))

                print(f"Valid game count: {valid_game_count}")
                print("File" + infile + "processing completed.")  # 打印结束
            else:
                continue

    for d in data:
        # 将数据集合中的数据写入文件，同时起到去重作用
        out.write(d)
    print(f"Total valid {len(data)} lines.")
    out.close()


def duplicate_removal_pos(pos_file):
    """
    pos 数据文件去重
    @param pos_file:
    @return:
    """
    data = set()
    size = os.path.getsize(pos_file)
    with open(pos_file, 'rb') as fi:
        for i in range(size // LINE_SIZE):
            line = fi.read(LINE_SIZE)
            data.add(line)

    with open(pos_file, 'wb') as fo:
        for d in data:
            fo.write(d)
    print(f"Remove {size // LINE_SIZE - len(data)} data, remaining {len(data)} data.")


def check_board(black_score, white_score, boards):
    """
    检查生成的最终棋盘是否和实际结果相同
    """
    # for i in range(2):
    #     for j in range(8):
    #         for k in range(8):
    #             print(board[i * 64 + j * 8 + k], end=',')
    #         print()
    #     print("............")
    # --------检查初始状态，一开始有四个棋子，第一步后有5个棋子
    init_board = boards[0]
    init_num = init_board[0:64].count(1) + init_board[64:].count(1)
    if init_num != 5:
        return False

    # --------检查最终状态，与游戏信息提供的一致
    finial_board = boards[59]
    board_black_score = finial_board[64:].count(1)
    board_white_score = finial_board[0:64].count(1)
    # print("Real score: B{}-W{}".format(black_score, white_score))
    # print("Real score: B{}-W{}".format(board_black_score, board_white_score))
    if board_black_score != black_score or board_white_score != white_score:
        return False
    else:
        return True


def generate_board_data(pos_data_file='data/pos_data', board_data_file='data/board_data', predict_user=False,
                        limit=10000, new_pos_data_file='data/new_pos_data'):
    """
    生成棋盘数据并存储为二进制数据文件，并限制每个级别的数据量
    :param new_pos_data_file: 为了保证生成的棋盘数据与行为数据对应，重新存储行为数据文件
    :param pos_data_file: 棋谱数据文件
    :param board_data_file: 棋盘数据文件
    :param limit: 每个级别的数据不超过此限制
    :param predict_user: 预测级别的对象是否是用户
    :return: 棋盘数据文件
    """
    size = os.path.getsize(pos_data_file)
    print("pos data file size is", size)
    line_no = 0
    saved_levels = [0] * 60  # 已保存的级别数量

    out_pos = open(new_pos_data_file, 'wb')
    out = open(board_data_file, 'wb')
    with open(pos_data_file, 'rb') as p_file:
        for i in range(size // LINE_SIZE):
            line_no += 1
            line = p_file.read(LINE_SIZE)
            ad_id, ai_first, pos, ai_level, black_score, white_score = struct.unpack(
                '36s?120sHHH', line)

            if saved_levels[ai_level] > limit:
                continue

            try:
                boards = board.get_state_from_board(pos.decode('utf-8'))
            except ValueError:
                continue

            # for n in range(5):
            #     for i in range(16):
            #         print(boards[n][i * 8:i * 8 + 8])
            #     print()

            # 检查棋盘状态生成是否错误
            if not check_board(black_score, white_score, boards):
                # print(f"Warning: board check failed at line {line_no}.")
                logger.debug(f"Board check failed at line {line_no}")
                continue

            if line_no % 1000 == 0:
                print(f"Line {line_no} processing...")

            boards_result = [0] * 120
            # 将每个棋盘状态表示为一个数
            for i in range(len(boards)):
                board_str = ''.join([str(b) for b in boards[i]])
                # pos_result[i] = str(pos_result[i]).replace('[', '').replace(']', '').replace(" ", "").replace(",","")
                boards_result[i * 2] = int(board_str[0:64], 2)
                boards_result[i * 2 + 1] = int(board_str[64:], 2)

            # 64比特用来表示预测级别的对象是否是先手，全1表示是先手的级别已知，全0表示后手级别已知
            if predict_user:
                user_first = int(('0' if ai_first else '1') * 64, 2)
            else:
                # 另外,在该场景下,已知级别都是AI级别,全1也可表示先手是AI，全0表示先手是玩家
                user_first = int(('1' if ai_first else '0') * 64, 2)

            out.write(struct.pack('36s121Qi', ad_id, *boards_result, user_first, ai_level))
            out_pos.write(line)
            saved_levels[ai_level] += 1
    out.close()

    # 绘制柱状图
    x = np.arange(60)
    y1 = np.array(saved_levels)
    plt.figure(figsize=(10, 10))

    plt.title('AI Levels')
    plt.xlabel('Level')
    plt.ylabel('Number')
    plt.bar(x, y1, width=0.5)
    plt.show()


def player_count(file_name='data/pos_data', low_num=10, high_num=100, level_max_num=20):
    """
    按用户进行数据划分

    :param file_name: 数据文件路径
    :param low_num: 用户最小数据量
    :param high_num: 用户最大数据量
    :param level_max_num: 每个用户每个级别最大对局数
    :return: 数据字典，用户id是键，除id外的二进制数据列表是值，列表长度在给定范围内
    """
    all_data = defaultdict(list)
    all_level_num = defaultdict(dict)
    value_data = {}
    size = os.path.getsize(file_name)
    print("pos data length is", size // LINE_SIZE)
    with open(file_name, 'rb') as f:
        for i in range(size // LINE_SIZE):
            line = f.read(LINE_SIZE)
            player_id, ai_first, pos, ai_level, black_score, white_score = struct.unpack('36s?120sHHH', line)
            if ai_level not in all_level_num[player_id]:
                all_level_num[player_id][ai_level] = 1
            elif all_level_num[player_id][ai_level] >= level_max_num:
                continue
            else:
                all_level_num[player_id][ai_level] += 1
            if len(all_data[player_id]) < high_num:
                # 数据条数不超过指定值
                all_data[player_id].append(line)

    # 筛选数量大于一定值的数据
    for k, v in all_data.items():
        if len(v) >= low_num:
            value_data[k] = v
    return value_data

# if __name__ == '__main__':
#     # json_files_process('/home/pcf/Documents/PyProject/Reversi/json', append=True)
#     # duplicate_removal_pos('data/pos_data')
#     data = player_count()
#     print(len(data))
#     for k in data.keys():
#         for line in data[k]:
#             ai_first, pos, ai_level, black_score, white_score = struct.unpack('?120sHHH', line)
#             print(ai_first, pos, ai_level, black_score, white_score)
#         break
