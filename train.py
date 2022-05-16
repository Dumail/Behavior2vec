import getopt
import logging
import os
import random
import sys
import time
import traceback
from copy import deepcopy
from typing import Tuple

import fitlog
import numpy as np
import torch
from torch import nn

from lvpredictor import LvPredictor
from behavior2vec import Behavior2vec
from predictor import Predictor
from res_predictor import ResPredictor
from utils.utils import nt_xent_loss, plot_tsne

fh = logging.FileHandler("logs/main.log")
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(fh)
logger.addHandler(ch)
random.seed(79)
np.random.seed(79)
torch.manual_seed(79)
torch.cuda.manual_seed(79)
torch.backends.cudnn.deterministic = True


# fitlog.commit(__file__)
# fitlog.debug()

def choose_evaluate(data_file, pickle_file_predict_lv, pickle_file_predict_res, pickle_file_clustering, embedding_net,
                    embedding_size, weight_file, evaluate_epoch, tuning_epoch, tuning_ratio, view, task):
    if task == "lv":
        evaluate_predict_lv(data_file, pickle_file_predict_lv, embedding_net, embedding_size, weight_file,
                            evaluate_epoch, tuning_epoch, tuning_ratio)
        fitlog.finish(0)
    elif task == 'res':
        evaluate_predict_result(data_file, embedding_net, embedding_size, weight_file, evaluate_epoch, tuning_epoch,
                                pickle_file_predict_res, tuning_ratio)
        fitlog.finish(0)


def evaluate_predict_lv(data_file, pickle_file_predict_lv, embedding_net, embedding_size, weight_file, evaluate_epoch,
                        tuning_epoch, tuning_ratio):
    """
    预测级别
    """
    lv_predict_loss_linear = Predictor.evaluate_predict(data_file, embedding_net, embedding_size, weight_file,
                                                        evaluate_epoch, pickle_file_predict_lv)
    lv_predict_loss_tuning = Predictor.evaluate_predict(data_file, embedding_net, embedding_size, weight_file,
                                                        tuning_epoch, pickle_file_predict_lv, is_fine_tuning=True,
                                                        train_ratio=tuning_ratio)
    fitlog.add_best_metric(lv_predict_loss_linear, "lv_loss_l")
    fitlog.add_best_metric(lv_predict_loss_tuning, "lv_loss_t")

    logging.info("".join(
        ["Evaluate linear probing loss for level prediction is ", str(lv_predict_loss_linear), ", fine tuning loss is ",
         str(lv_predict_loss_tuning)]))


def evaluate_predict_result(data_file, embedding_net, embedding_size, weight_file, evaluate_epoch, tuning_epoch,
                            pickle_file_predict_res, tuning_ratio):
    """
    预测胜负
    """
    result_predict_accuracy_linear = Predictor.evaluate_predict(data_file, embedding_net, embedding_size, weight_file,
                                                                evaluate_epoch, pickle_file_predict_res,
                                                                is_predict_result=True, steps=params["steps"])
    result_predict_accuracy_tuning = Predictor.evaluate_predict(data_file, embedding_net, embedding_size, weight_file,
                                                                tuning_epoch, pickle_file_predict_res,
                                                                is_predict_result=True, is_fine_tuning=True,
                                                                train_ratio=tuning_ratio, steps=params["steps"])

    fitlog.add_best_metric(result_predict_accuracy_linear, "res_acc_l")
    fitlog.add_best_metric(result_predict_accuracy_tuning, "res_acc_t")

    logging.info("".join(
        ["Evaluate linear probing loss for result prediction is ", str(result_predict_accuracy_linear),
         ", fine tuning loss is ", str(result_predict_accuracy_tuning)]))


def train_predictor_board():
    # - 通过级别预测进行训练，输入为棋盘序列
    logging.info("Start training lv predictor of board...")
    predictor = LvPredictor(learning_rate=params["learning_rate"], weight_decay=params["weight_decay"],
                            board_data_type=True, view=params["view"],
                            reduce_lr=params["reduce_lr"], embedding_size=params["embedding_size"])
    # fitlog.add_hyper(params)
    # fitlog.add_hyper("lv_predict", "method")

    try:
        predictor.load_data('data/board_data', pickle_file='tmp/data_eva.pt', train_ratio=params['tuning_ratio'])
        predictor.train(batch_size=params["batch_size"], epoch=params["tuning_epoch"])
    except ValueError:
        fitlog.finish(1)

    logging.info("Supervised evaluate loss is " + str(predictor.evaluate()))

    # data, level = predictor.pos2vec("data/board_data", 1000, ("tmp/plot.pt", "tmp/labels.npy"))
    # plot_tsne(data, level, "imgs/")


def train_res_predictor():
    logging.info("Start training res predictor of board...")
    predictor = ResPredictor(learning_rate=params["learning_rate"], weight_decay=params["weight_decay"],
                             board_data_type=True, steps=params["steps"],
                             reduce_lr=params["reduce_lr"], embedding_size=params["embedding_size"])
    # fitlog.add_hyper(params)
    # fitlog.add_hyper("lv_predict", "method")

    try:
        predictor.load_data('data/board_data', pickle_file='tmp/data_res.pt', train_ratio=params['tuning_ratio'])
        predictor.train(batch_size=params["batch_size"], epoch=params["tuning_epoch"])
    except ValueError:
        fitlog.finish(1)

    logging.info("Supervised evaluate accuracy is " + str(predictor.evaluate()))


def train_b2v():
    logging.info("Start training PolicyEncoder")
    encoder = Behavior2vec(learning_rate=params["learning_rate"], weight_decay=params["weight_decay"],
                           reduce_lr=params["reduce_lr"], steps=params["steps"],
                           embedding_size=params["embedding_size"], loss_fn=nn.MSELoss(),
                           view=params["view"], mask_ratio=params["mask_ratio"], task=params['task'])
    fitlog.add_hyper(params)
    fitlog.add_hyper("policy", "method")
    encoder.load_data('data/board_data', 'data/pos_data', -1,
                      'tmp/policy_states_' + params["view"] + '.pt',
                      'tmp/policy_actions_' + params["view"] + '.pt')
    try:
        encoder.train(batch_size=params["batch_size"], epoch=params["epoch"])
        out = encoder.evaluate(batch_size=params["batch_size"], show_out=True)
        # fitlog.add_to_line(out)
        # pass
    except ValueError as e:
        logger.error(traceback.format_exc())
        fitlog.finish(1)
        sys.exit(-1)
    encoder.empty_data()

    weight_file = os.path.join(fitlog.get_log_folder(True), 'model/best.pth')
    choose_evaluate(data_file='data/board_data',
                    pickle_file_predict_lv='tmp/data_eva.pt',
                    pickle_file_predict_res='tmp/data_res.pt',
                    pickle_file_clustering='tmp/data_cluser.pt',
                    embedding_size=params["embedding_size"],
                    embedding_net=encoder.net,
                    weight_file=weight_file,
                    evaluate_epoch=params["evaluate_epoch"],
                    tuning_epoch=params["tuning_epoch"],
                    tuning_ratio=params["tuning_ratio"], view=params["view"], task=params["task"])

    # data, level = encoder.pos2vec("data/board_data", 1000, ("tmp/data.npy", "tmp/labels.npy"))
    # plot_tsne(data, level, "imgs/")


if __name__ == '__main__':

    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "h", ["view=", "embedding_size=", "mask_ratio=", "task=", "steps="])
    except:
        logging.error("Args error!")
        exit(1)

    # 以下为超参数
    params = {"view": "both", "batch_size": 1024, "epoch": 20, "evaluate_epoch": 50, "tuning_epoch": 10,
              "learning_rate": 0.0008, "weight_decay": 0, "embedding_size": 1024, "mask_ratio": 0.7,
              "reduce_lr": True, "tuning_ratio": 0.01, "task": "res", "steps": 44, "conv_depth": 1}

    for opt, arg in opts:
        if opt in ['--view']:
            params["view"] = arg
        if opt in ['--embedding_size']:
            params["embedding_size"] = int(arg)
        if opt in ['--mask_ratio']:
            mask_ratio = float(arg)
        if opt in ['--task']:
            params["task"] = arg
        if opt in ['--steps']:
            params["steps"] = int(arg)

    train_b2v()
    if params['task'] == "lv":
        train_predictor_board()
    else:
        train_res_predictor()
    logging.info("Program finished!")
