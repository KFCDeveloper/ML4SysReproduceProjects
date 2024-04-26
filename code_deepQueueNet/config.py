import pandas as pd
import numpy as np


# 在第一个 jupyter notebook里面，这个4-port switch/FIFO被加载后，拿来train了。
# 但是这里又没有表明是在FatTree16，所以这个应该是通用的
# 有点好奇，这里面的配置改了后，还能直接用之前的模型吗
# time steps 也不知道是个什么玩意，论文里没有对 time step 做描述，但是也出现了time step

class BaseConfig:
    test_size = 0.2  # train_test_split ratio
    sub_rt = 0.005  # subsampling for Eval.
    TIME_STEPS = 42
    BATCH_SIZE = 32 * 8
    modelname = '4-port switch/FIFO'
    no_of_port = 4
    no_of_buffer = 1
    ser_rate = 2.5 * 1024 ** 2  # unit conversion b to MB
    sp_wgt = 0.
    seed = 0    # todo 不知道是什么，可能是随机数的随机种子
    window = 63  # window size to cal. average service time.
    no_process = 15  # multi-processing:no of processes used.
    epochs = 6
    n_outputs = 1
    learning_rate = 0.001
    l2 = 0.1
    lstm_params = {'layer': 2, 'cell_neurons': [200, 100], 'keep_prob': 1}
    att = 64.  # attention output layer dim
    mul_head = 3
    mul_head_output_nodes = 32


# Path config. To access the existing model in the directory.
class modelConfig:
    scaler = './trained/scaler'
    model = './trained/model'
    md = 341
    train_sample = './trained/sample/train.h5'
    test1_sample = './trained/sample/test1.h5'
    test2_sample = './trained/sample/test2.h5'
    bins = 100
    errorbins = './trained/error'
    error_correction = False
