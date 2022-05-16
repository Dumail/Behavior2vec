#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2021/4/2 下午1:46
# @Author  : PCF
# @Email   : pan.chaofan@foxmail.com
# @File    : board_tools.py
# @Software: PyCharm
import numpy as np


class BoardTools:
    def __init__(self, piece_type):
        self.stack = np.zeros(8)
        self.piece_type = 0
        self.end = 0
        if piece_type == 'black':
            self.piece_type = 1
        elif piece_type == 'white':
            self.piece_type = 2
        self.stack[0] = self.piece_type

    def addElement(self, element):
        """
        每添加一个元素，就可以返回可以吃掉多少个子
        @param element:
        @return:
        """
        self.end += 1
        self.stack[self.end] = element
        return self.check()

    def check(self):
        rs = 0
        # 碰到空或者相同类型的都应该结束
        for i in range(1, self.end + 1):
            if self.stack[i] == 0:  # 判断到该方向的最后一个棋子
                if i == 1:  # 挨着第一个为空
                    rs = 0  # 不能吃子，直接返回吃0个，即不能吃子
                return rs
            # 匹配到该方向第一个相同的棋子
            if self.stack[i] == self.piece_type:
                if i == 1:  # 挨着第一个相同类型
                    rs = 0  # 不能吃子，直接返回吃0个，即不能吃子
                else:
                    rs = i - 1
                return rs
            # 匹配到该方向的不同类型棋子
            # 匹配到不同的棋子，直接看下一个
        return rs

    def clear(self):
        """
        重置对象数据
        """
        self.stack = np.array([self.piece_type, 0, 0, 0, 0, 0, 0, 0])
        self.end = 0
