#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2021/4/2 下午1:46
# @Author  : PCF
# @Email   : pan.chaofan@foxmail.com
# @File    : board.py
# @Software: PyCharm
import numpy as np

from board_tools import BoardTools


def str2index(string):
    mark = 1
    if len(string) != 2:
        mark = 0
    a = 'a'
    # 计算数组下标
    row = ord(string[0]) - ord(a)
    col = int(string[1])
    index = row * 8 + col - 1

    if index > 63 or index < 0:
        mark = 0
    if mark == 0:
        print("Error of pos index.")
        return -1
    return index


class Board:
    def __init__(self):
        # 1为黑子，2为白子，0为空
        self.board = np.zeros(64)
        # 初始棋盘有四颗子，左上右下白，左下右上黑
        self.board[27] = 2
        self.board[28] = 1
        self.board[35] = 1
        self.board[36] = 2

    def try_change_board(self, index, piece_type):
        """
        尝试下这步棋
        @param index:
        @param piece_type:
        @return:
        """
        mark = False  # 如果发生吃棋，则置 mark 为 true，代表下一步棋 piece_type 不冲突
        # 落子为黑子时
        if self.board[index] == 1 or self.board[index] == 2:
            return False
        if piece_type == "black":
            tool = BoardTools(piece_type)

            # 朝左边遍历，添加至栈中
            tool.clear()
            temp = index % 8
            start = index
            eat_num = 0
            i = 0
            while i < temp:
                eat_num = tool.addElement(self.board[start - i - 1])
                i += 1
            if eat_num > 0:
                self.board[start - eat_num] = 1  # 可以吃掉则下掉这颗棋子
                mark = True
            while eat_num > 0:
                self.board[start - eat_num] = 1  # 从最远处吃回来，白子变为黑子
                eat_num -= 1

            # 朝右边遍历，添加至栈中
            tool.clear()  # 重置工具
            temp = 8 - index % 8 - 1  # 右边剩下的个数要剪一个
            start = index
            eat_num = 0
            i = 0
            while i < temp:
                eat_num = tool.addElement(self.board[start + i + 1])
                i += 1
            if eat_num > 0:
                self.board[start] = 1  # 可以吃掉则下掉这颗棋子
                mark = True
            while eat_num > 0:
                self.board[start + eat_num] = 1  # 从最远处吃回来，白子变为黑子
                eat_num -= 1

            # 朝上边遍历，添加至栈中
            tool.clear()  # 重置工具
            temp = index // 8
            start = index
            eat_num = 0
            i = 0
            while i < temp:
                eat_num = tool.addElement(self.board[start - 8 * (i + 1)])
                i += 1
            if eat_num > 0:
                self.board[start] = 1
                mark = True
            while eat_num > 0:
                self.board[start - 8 * eat_num] = 1
                eat_num -= 1

            # 朝下边遍历，添加至栈中
            tool.clear()
            temp = 8 - index // 8 - 1
            start = index
            eat_num = 0
            i = 0
            while i < temp:
                eat_num = tool.addElement(self.board[start + 8 * (i + 1)])
                i += 1
            if eat_num > 0:
                self.board[start] = 1
                mark = True
            while eat_num > 0:
                self.board[start + 8 * eat_num] = 1
                eat_num -= 1

            # 朝右上边遍历，添加至栈中
            tool.clear()
            temp1 = index // 8
            temp2 = 8 - index % 8 - 1
            temp = min(temp1, temp2)
            start = index
            eat_num = 0
            i = 0
            while i < temp:
                eat_num = tool.addElement(self.board[start - 7 * (i + 1)])
                i += 1
            if eat_num > 0:
                self.board[start] = 1
                mark = True
            while eat_num > 0:
                self.board[start - 7 * eat_num] = 1
                eat_num -= 1

            # 朝左上边遍历，添加至栈中
            tool.clear()
            temp1 = index // 8
            temp2 = index % 8
            temp = min(temp1, temp2)
            start = index
            eat_num = 0
            i = 0
            while i < temp:
                eat_num = tool.addElement(self.board[start - 9 * (i + 1)])
                i += 1
            if eat_num > 0:
                self.board[start] = 1
                mark = True
            while eat_num > 0:
                self.board[start - 9 * eat_num] = 1
                eat_num -= 1

            # 朝左下边遍历，添加至栈中
            tool.clear()
            temp1 = 8 - index // 8 - 1
            temp2 = index % 8
            temp = min(temp1, temp2)
            start = index
            eat_num = 0
            i = 0
            while i < temp:
                eat_num = tool.addElement(self.board[start + 7 * (i + 1)])
                i += 1
            if eat_num > 0:
                self.board[start] = 1
                mark = True
            while eat_num > 0:
                self.board[start + 7 * eat_num] = 1
                eat_num -= 1

            # 朝右下边遍历，添加至栈中
            tool.clear()
            temp1 = 8 - index // 8 - 1
            temp2 = 8 - index % 8 - 1
            temp = min(temp1, temp2)
            start = index
            eat_num = 0
            i = 0
            while i < temp:
                eat_num = tool.addElement(self.board[start + 9 * (i + 1)])
                i += 1
            if eat_num > 0:
                self.board[start] = 1
                mark = True
            while eat_num > 0:
                self.board[start + 9 * eat_num] = 1
                eat_num -= 1

        elif piece_type == "white":
            tool = BoardTools(piece_type)

            # 朝左边遍历，添加至栈中
            tool.clear()
            temp = index % 8
            start = index
            eat_num = 0
            i = 0
            while i < temp:
                eat_num = tool.addElement(self.board[start - i - 1])
                i += 1
            if eat_num > 0:
                self.board[start - eat_num] = 2
                mark = True
            while eat_num > 0:
                self.board[start - eat_num] = 2
                eat_num -= 1

            # 朝右边遍历，添加至栈中
            tool.clear()
            temp = 8 - index % 8 - 1
            start = index
            eat_num = 0
            i = 0
            while i < temp:
                eat_num = tool.addElement(self.board[start + i + 1])
                i += 1
            if eat_num > 0:
                self.board[start] = 2
                mark = True
            while eat_num > 0:
                self.board[start + eat_num] = 2
                eat_num -= 1

            # 朝上边遍历，添加至栈中
            tool.clear()
            temp = index // 8
            start = index
            eat_num = 0
            i = 0
            while i < temp:
                eat_num = tool.addElement(self.board[start - 8 * (i + 1)])
                i += 1
            if eat_num > 0:
                self.board[start] = 2
                mark = True
            while eat_num > 0:
                self.board[start - 8 * eat_num] = 2
                eat_num -= 1

            # 朝下边遍历，添加至栈中
            tool.clear()
            temp = 8 - index // 8 - 1
            start = index
            eat_num = 0
            i = 0
            while i < temp:
                eat_num = tool.addElement(self.board[start + 8 * (i + 1)])
                i += 1
            if eat_num > 0:
                self.board[start] = 2
                mark = True
            while eat_num > 0:
                self.board[start + 8 * eat_num] = 2
                eat_num -= 1

            # 朝右上边遍历，添加至栈中
            tool.clear()
            temp1 = index // 8
            temp2 = 8 - index % 8 - 1
            temp = min(temp1, temp2)
            start = index
            eat_num = 0
            i = 0
            while i < temp:
                eat_num = tool.addElement(self.board[start - 7 * (i + 1)])
                i += 1
            if eat_num > 0:
                self.board[start] = 2
                mark = True
            while eat_num > 0:
                self.board[start - 7 * eat_num] = 2
                eat_num -= 1

            # 朝左上边遍历，添加至栈中
            tool.clear()
            temp1 = index // 8
            temp2 = index % 8
            temp = min(temp1, temp2)
            start = index
            eat_num = 0
            i = 0
            while i < temp:
                eat_num = tool.addElement(self.board[start - 9 * (i + 1)])
                i += 1
            if eat_num > 0:
                self.board[start] = 2
                mark = True
            while eat_num > 0:
                self.board[start - 9 * eat_num] = 2
                eat_num -= 1

            # 朝左下边遍历，添加至栈中
            tool.clear()
            temp1 = 8 - index // 8 - 1
            temp2 = index % 8
            temp = min(temp1, temp2)
            start = index
            eat_num = 0
            i = 0
            while i < temp:
                eat_num = tool.addElement(self.board[start + 7 * (i + 1)])
                i += 1
            if eat_num > 0:
                self.board[start] = 2
                mark = True
            while eat_num > 0:
                self.board[start + 7 * eat_num] = 2
                eat_num -= 1

            # 朝右下边遍历，添加至栈中
            tool.clear()
            temp1 = 8 - index // 8 - 1
            temp2 = 8 - index % 8 - 1
            temp = min(temp1, temp2)
            start = index
            eat_num = 0
            i = 0
            while i < temp:
                eat_num = tool.addElement(self.board[start + 9 * (i + 1)])
                i += 1
            if eat_num > 0:
                self.board[start] = 2
                mark = True
            while eat_num > 0:
                self.board[start + 9 * eat_num] = 2
                eat_num -= 1

        return mark

    def change_board(self, pos, piece_type):
        """
        输入下一步棋的字符串，更改棋盘面貌
        @param pos:
        @param piece_type:
        @return:
        """
        index = str2index(pos)  # 根据坐标 转化为 数组下标
        keep = self.try_change_board(index, piece_type)
        # print("Insert success: ", keep)
        if not keep:
            if piece_type == "black":
                piece_type = "white"
                next_type = "black"
            else:
                piece_type = "black"
                next_type = "white"
            self.try_change_board(index, piece_type)
        else:
            if piece_type == "black":
                next_type = "white"
                self.board[index] = 1
            else:
                next_type = "black"
                self.board[index] = 2
        return next_type

    def print_board(self):
        """
        获取棋盘状态
        @return: 棋盘状态，用大小为2*8*的列表
        """
        # return self.board
        temp_board = [0] * 8 * 8 * 2
        for i in range(8):
            for j in range(8):
                if self.board[i * 8 + j] == 2:
                    temp_board[j * 8 + i] = 1
                    temp_board[64 + j * 8 + i] = 0
                elif self.board[i * 8 + j] == 1:
                    temp_board[64 + j * 8 + i] = 1
                    temp_board[j * 8 + i] = 0
                # temp_board[j*8+i]= self.board[i*8+j]
        return temp_board


def get_state_from_board(pos: str):
    """
    由棋谱信息得到一系列棋盘状态
    @param pos:  棋谱
    @return:  棋盘状态序列，大小为60的列表，每个元素为2*8*8的状态
    """
    state = []
    board = Board()
    arrays = []
    piece_type = "black"
    # board.print_board()
    for i in range(60):
        arrays.append(pos[i * 2:i * 2 + 2])
        # print(f"............{i + 1}..........{arrays[i]}..........{piece_type}...")
        piece_type = board.change_board(arrays[i], piece_type)
        # board_array = board.print_board()
        state.append(board.print_board())
    return state


if __name__ == "__main__":
    pos_test = """f5f6e6f4c3b2e3d6c6b6g5g3e7g7f3d3c2c5d7f8f7g6h8c4h7g8e8d2\
a6c7d1e1f1a5a4h5h3h6h4f2e2c8d8g4b8g2h2h1b7a8b4c1a2g1b5a7a1b3b1a3"""
    state = get_state_from_board(pos_test)
    print(len(state))
    print(len(state[0]))
    for i in range(8 * 2):
        for j in range(8):
            print(state[59][i * 8 + j], end=',')
        print()
