#!/usr/bin/env python3

import argparse
import math
import numpy as np
import os
import pickle
import statistics
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from AL_module_trunk_part import *
import time
from torch.autograd import Variable, Function
from scipy.stats import t
from common.discretization import *
from common.loss import *
from common.utils import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

print("USING CUDA:", use_cuda)

is_log = False

##  Feature vector for 2-rack TCP (18 features)
##      0 1 2 3 - Congestion state
##      4 5 6 7 - Server
##      8 9 - Agg
##      10 11 - Agg interface
##      12 13 - ToR
##      14 - Time since last packet
##      15 - EWMA
##      16 - Last drop prediction
##      17 - Last latency prediction

class NetworkApproxLSTM(nn.Module):
    def __init__(self, input_size=18, num_layers=2, window_size=10):
        super(NetworkApproxLSTM, self).__init__()

        self.input_size = input_size
        # remember window_size packets
        self.hidden_size = input_size * window_size
        self.num_layers = num_layers
        self.window_size = window_size
        self.output_size = 1
        self.variant = "TCP"

        # batch_first – If True, then the input and output tensors are provided
        # as (batch, seq, feature) or (batch, num_layers, hidden_size).
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers,
                            batch_first = True) 

        self.linearL = nn.Linear(self.hidden_size, 1)
        self.linearD = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def init_hidden(self, batch_size = 1):
        h_t = torch.zeros(batch_size, self.num_layers, self.hidden_size,
                          dtype=torch.double).to(device)
        c_t = torch.zeros(batch_size, self.num_layers, self.hidden_size,
                          dtype=torch.double).to(device)
        self.hidden_state = (h_t, c_t)

    def forward(self, data):
        X = data.view(-1, self.window_size, self.input_size)
        batch_size = X.shape[0]
        lstm_out, self.hidden_state = self.lstm(X, (self.hidden_state[0].view(self.num_layers, batch_size, -1), self.hidden_state[1].view(self.num_layers, batch_size, -1))) ## Qizhen: to tune hidden state

        l_output = self.linearL(lstm_out[:,-1,:].view(batch_size, -1))
        d_output = self.sigmoid(self.linearD(lstm_out[:,-1,:].view(batch_size, -1)))
        return d_output, l_output


def discretize_features(X, y_l, granu = 1000):
    Last, meta_last = discretize((X.T[-2]).astype(float), granu)
    EWMA, meta_ewma = discretize((X.T[-1]).astype(float), granu)
    Latency, meta_latency = discretize(y_l.astype(float), granu)
    Last = np.array([[l] for l in Last]).astype(int)
    EWMA = np.array([[e] for e in EWMA]).astype(int)
    Latency = np.array(Latency).astype(int)

    data = np.concatenate((X.T[:-2].T, Last, EWMA), axis = 1)
    
    return data, Latency, meta_last, meta_ewma, meta_latency

def format_data(X, y_d, y_l, num_servers = 4, degree = 2,
                network_states = 4, include_label = False):
    Congestion = convertToOneHot((X.T[0]).astype(int), network_states)
    Server = convertToOneHot((X.T[1]).astype(int), num_servers)
    Agg = convertToOneHot(X.T[2].astype(int), degree)
    Agg_intf = convertToOneHot(X.T[3].astype(int), degree)
    ToR = convertToOneHot(X.T[4].astype(int), degree)

    data = np.concatenate((Congestion, Server, Agg, Agg_intf, ToR, X.T[5:].T), \
                          axis = 1)

    if not include_label:
        return data, y_d, y_l

    data = np.concatenate((data[1:], np.stack((y_d[:-1], y_l[:-1]), axis=1)), axis=1)
    return data, y_d[1:], y_l[1:]

def pad_zeros(X, y_d, y_l, window_size):
    pad_size = window_size - 1
    print ("X shape", X.shape)
    empty_X = np.zeros((pad_size, len(X[0]))).astype(int)
    print ("empty X shape", empty_X.shape)
    empty_y = np.zeros((pad_size)).astype(int)
    return np.concatenate((empty_X, X)), np.concatenate((empty_y, y_d)), \
           np.concatenate((empty_y, y_l))

def train_both_data_generator(large_train_samples, large_train_d_targets,
                              large_train_l_targets, window_size, index,
                              batch_size):
    if index + window_size + (batch_size - 1) > len(large_train_samples):
        remaining_samples = len(large_train_samples) - index - window_size + 1
        print("remain sample:")
        print(remaining_samples)
        return None
    train_batch = [large_train_samples[index+i : index+i+window_size] for i in range(batch_size)]
    train_batch = np.array(train_batch).reshape(len(train_batch), len(train_batch[0]), len(train_batch[0][0]))
    d_label_batch = [large_train_d_targets[index+i+window_size - 1] for i in range(batch_size)]
    l_label_batch = [large_train_l_targets[index+i+window_size - 1] for i in range(batch_size)]
    d_label_batch = np.array(d_label_batch).reshape(len(d_label_batch), 1)
    l_label_batch = np.array(l_label_batch).reshape(len(l_label_batch), 1)
    return train_batch, d_label_batch, l_label_batch

# def train_both_data_generator(large_train_samples, large_train_d_targets,
#                               large_train_l_targets, window_size, index,
#                               batch_size):
#     if index + window_size + (batch_size - 1) > len(large_train_samples):
#         return None
#     train_batch = [large_train_samples[index+i : index+i+window_size] for i in range(batch_size)]
#     if len(train_batch) < batch_size:
#         return None
#     train_batch = np.array(train_batch).reshape(len(train_batch), len(train_batch[0]), len(train_batch[0][0]))
#     d_label_batch = [large_train_d_targets[index+i+window_size - 1] for i in range(batch_size)]
#     l_label_batch = [large_train_l_targets[index+i+window_size - 1] for i in range(batch_size)]
#     d_label_batch = np.array(d_label_batch).reshape(len(d_label_batch), 1)
#     l_label_batch = np.array(l_label_batch).reshape(len(l_label_batch), 1)
#     return train_batch, d_label_batch, l_label_batch

def train(model, data, target_drop, target_latency, optimizer, alpha=0.33,
          drop_weight=0.5, latency_loss='huber', discretized_max=1000):
    model.init_hidden(data.shape[0])
    data = data.to(device, dtype=torch.double)
    target_drop = target_drop.to(device, dtype=torch.double)
    target_latency = target_latency.to(device, dtype=torch.double)

    pred_drop, pred_latency = model(data)

    pred_drop = pred_drop.view(-1, 1)
    pred_latency = pred_latency.view(-1, 1)
    target_drop = target_drop.view(-1, 1)
    target_latency = target_latency.view(-1, 1)
    loss_drop = weighted_binary_cross_entropy(pred_drop, target_drop, weight0=(1-drop_weight))
    
    uncertainties = get_uncertainty(target_latency,pred_latency,target_drop,pred_drop)

    if latency_loss == 'huber':
        loss_latency = rescaled_huber_loss(pred_latency, target_latency, discretized_max = discretized_max)
    elif latency_loss == 'mse':
        loss_latency = rescaled_mean_squared_error(pred_latency, target_latency, discretized_max = discretized_max)
    elif latency_loss == 'male':
        loss_latency = rescaled_mean_absolute_log_error(pred_latency, target_latency, discretized_max = discretized_max)
    elif latency_loss == 'mae':
        loss_latency = rescaled_mean_absolute_error(pred_latency, target_latency, discretized_max = discretized_max)
    else:
        loss_latency = mean_absolute_error(pred_latency, target_latency)

    loss = alpha*loss_latency + (1-alpha)*loss_drop

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    uncertainties_flat = [item for sublist in uncertainties for item in sublist]
    
    
    return loss.item(),sum(uncertainties_flat)

def test(model, data, target_drop, target_latency, alpha=0.33, drop_weight=0.5,
         latency_loss='huber', discretized_max=1000):
    model.init_hidden(data.shape[0])
    data = data.to(device, dtype=torch.double)
    target_drop = target_drop.to(device, dtype=torch.double)
    target_latency = target_latency.to(device, dtype=torch.double)

    pred_drop, pred_latency = model(data)

    pred_drop = pred_drop.view(-1, 1)
    pred_latency = pred_latency.view(-1, 1)
    target_drop = target_drop.view(-1, 1)
    target_latency = target_latency.view(-1, 1)
    lossDrop = weighted_binary_cross_entropy(pred_drop, target_drop, weight0=(1-drop_weight))

    if latency_loss == 'huber':
        lossLatency = rescaled_huber_loss(pred_latency, target_latency, discretized_max = discretized_max)
    elif latency_loss == 'mse':
        lossLatency = rescaled_mean_squared_error(pred_latency, target_latency, discretized_max = discretized_max)
    elif latency_loss == 'male':
        lossLatency = rescaled_mean_absolute_log_error(pred_latency, target_latency, discretized_max = discretized_max)
    elif latency_loss == 'mae':
        lossLatency = rescaled_mean_absolute_error(pred_latency, target_latency, discretized_max = discretized_max)
    else:
        lossLatency = mean_absolute_error(pred_latency, target_latency)

    loss = alpha*lossLatency + (1-alpha)*lossDrop

    return loss.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_dict", type=str,
                        help="Data dictionary pickle file")
    parser.add_argument("degree", type=int,
                        help="Number of ToRs/Aggs/AggUplinks per cluster")

    parser_saveload = parser.add_argument_group('model saving/loading')
    parser_saveload.add_argument("--model_name", type=str,
                                 help="Output model filename")
    parser_saveload.add_argument("--direction", type=str,
                                 choices=['INGRESS', 'EGRESS'],
                                 help="direction hint to prepend to model name")
    parser_saveload.add_argument("--load_model", type=str,
                                 help="path to a pretrained model")

    parser_model = parser.add_argument_group('model and feature options')
    parser_model.add_argument("--num_layers", type=int,
                              help="number of layers in lstm")
    parser_model.add_argument("--double_type", dest="double_type",
                              action="store_true",
                              help="make the model double type")
    parser_model.add_argument("--window_size", type=int,
                              help="number of packets in the window")
    parser_model.add_argument("--include_label", dest="include_label",
                              action="store_true",
                              help="Include previous predictions of drop and " \
                                   "latency in feature set")
    parser_model.add_argument("--exclude_label", dest="include_label",
                              action="store_false",
                              help="Exclude previous predictions of drop and " \
                                   "latency in feature set")
    parser_model.add_argument("--disc_factor", type=float,
                              help="Discretization factor")

    parser_loss = parser.add_argument_group('loss options')
    parser_loss.add_argument("--latency_loss", type=str,
                             choices=['huber', 'mse', 'male', 'mae',
                                      'mae_norescale'],
                             help="Loss function for latency")
    parser_loss.add_argument("--drop_weight", type=float,
                             help="weight on drops [0-1]")
    parser_loss.add_argument("--alpha", type=float,
                             help="Weight parameter for latency influence")

    parser_train = parser.add_argument_group('training options')
    parser_train.add_argument("--num_epochs", type=int,
                              help="number of epoch to run for")
    parser_train.add_argument("--batch_size", type=int,
                              help="How many examples per batch")
    parser_train.add_argument("--learning_rate", type=float,
                              help="Learning rate constant")
    parser_train.add_argument("--train_size", type=int,
                              help="number of packets for training " \
                                   "(-1 means all data)")
    parser_train.add_argument("--train_size_prop", type=float,
                              help="Proportion of packets for training (do " \
                                   "not set this for all data)")
    parser_train.add_argument("--enable_validation", dest="enable_validation",
                              action="store_true",
                              help="split 10%% of data to be validation data")

    parser.set_defaults(direction="INGRESS")

    parser.set_defaults(num_layers=2)
    parser.set_defaults(double_type=True)
    parser.set_defaults(window_size=12)
    parser.set_defaults(include_label=True)
    parser.set_defaults(disc_factor=1000)

    parser.set_defaults(latency_loss='huber')
    parser.set_defaults(drop_weight=0.9)
    parser.set_defaults(alpha=0.5)

    parser.set_defaults(num_epochs=10)
    parser.set_defaults(batch_size=128)
    parser.set_defaults(learning_rate=1e-4)
    parser.set_defaults(train_size=-1)
    parser.set_defaults(enable_validation=False)

    args = parser.parse_args()

    data_dict = args.data_dict
    degree = args.degree

    direction = args.direction

    num_layers = args.num_layers
    double_type = args.double_type
    window_size = args.window_size
    include_label = args.include_label
    disc_factor = args.disc_factor

    latency_loss = args.latency_loss
    drop_weight = args.drop_weight
    alpha = args.alpha

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    train_size = args.train_size
    enable_validation = args.enable_validation
    print(f"enable_validation {enable_validation}", flush=True)
    torch.cuda.manual_seed_all(22)
    torch.manual_seed(22)

    pdata = pickle.load(open(data_dict, 'rb'))
    X, y_d, y_l = pdata['X'], pdata['y_d'], pdata['y_l']

    print("Discretizing with factor %s..." % disc_factor)
    X, y_l, dis_meta_last, dis_meta_ewma, dis_meta_latency = \
            discretize_features(X, y_l, disc_factor)
    print("meta_last =", dis_meta_last, "meta_ewma =", dis_meta_ewma,
          "meta_latency =", dis_meta_latency)

    # Trim for training
    if train_size != -1:
        X = X[:train_size]
        y_d = y_d[:train_size]
        y_l = y_l[:train_size]
    if args.train_size_prop is not None:
        X = X[:round(len(X) * args.train_size_prop)]
        y_d = y_d[:round(len(y_d) * args.train_size_prop)]
        y_l = y_l[:round(len(y_l) * args.train_size_prop)]

    print("Formatting...")
    X, y_d, y_l = format_data(X, y_d, y_l, degree = degree,
                              include_label = include_label)

    print("Priming...")
    X, y_d, y_l = pad_zeros(X, y_d, y_l, window_size)
    print("Feature vector size: ", X.shape)
    

    if args.model_name is None:
        if args.train_size_prop is None:
            model_name = "%s_LSTM_Pytorch_Double%s_SW%s_LAYER%s_FEAT%s_BATCH%s_WIN%s_Alpha%s_DWeight%s_LatLoss%s_EPOCH%s" \
                       % (str(direction), str(double_type), str(degree),
                          str(num_layers), str(X.shape[1]), str(batch_size),
                          str(window_size), str(alpha), str(drop_weight),
                          str(latency_loss), str(num_epochs))
        else:
            model_name = "%s_LSTM_Pytorch_Double%s_SW%s_FEAT%s_BATCH%s_WIN%s_Alpha%s_DWeight%s_EPOCH%s_PROP%s" \
                       % (str(direction), str(double_type), str(degree),
                          str(X.shape[1]), str(batch_size), str(window_size),
                          str(alpha), str(drop_weight), str(num_epochs),
                          str(args.train_size_prop))
    else:
        model_name = args.model_name

    print("Model name =", model_name)

    start_epoch = 0

    if args.load_model:
        checkpoint = torch.load(args.load_model,
                                map_location=lambda storage, loc:storage)
        assert checkpoint["variant"] == "TCP"

        input_size = checkpoint["input_size"]
        window_size = checkpoint["window_size"]
        start_epoch = checkpoint["start_epoch"]
        print("Loading model from checkpoint:", args.load_model,
              "with", (start_epoch + 1), "epochs")
        model = NetworkApproxLSTM(input_size=input_size, window_size=window_size).to(device)
        if double_type:
            model.double()
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model = NetworkApproxLSTM(input_size=X.shape[1], num_layers=num_layers,
                                  window_size=window_size).to(device)
        if double_type:
            model.double()

    print(model)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print ("batch size = %s" % batch_size)

    rs = [_ for _ in range(math.floor(X.shape[0]/batch_size) - 1)]
    
    
    
    

    # batches_in_epoch = math.floor(X.shape[0]/batch_size) - 1 - window_size
    start_time = time.time()
    print('start_time {}'.format(start_time)) 
    initialize = True

    # uncertainties = []
    # index = 0
    # with torch.no_grad():
    #     while window_size + index <= X.shape[0]:
    #         cur_step = int(index / batch_size)
    #         data = train_both_data_generator(X, y_d, y_l, window_size, index, batch_size)
    #         if data is None:
    #             break
    #         X_sample, y_d_sample, y_l_sample = data
    #         X_sample = torch.from_numpy(X_sample).to(device, dtype=torch.double)
    #         y_d_sample = torch.from_numpy(y_d_sample).to(device, dtype=torch.double)
    #         y_l_sample = torch.from_numpy(y_l_sample).to(device, dtype=torch.double)
            
    #         model.init_hidden(X_sample.shape[0])

    #         pred_drop, pred_latency = model(X_sample)
    #         pred_latency = pred_latency.view(-1, 1)
    #         target_latency = y_l_sample.view(-1, 1)
            
    #         pred_drop = pred_drop.view(-1, 1)
    #         target_drop = y_d_sample.view(-1, 1)
            
    #         # variances = (0.5*(pred_latency - target_latency)+0.5*(pred_drop - target_drop)) ** 2
    #         pred_latency_detached = pred_latency.detach()
    #         pred_drop_detached = pred_drop.detach()

    #         confidence_level = 0.95
    #         t_value = t.ppf((1 + confidence_level) / 2, len(pred_latency) - 1)
    #         intervals_latency = []
    #         for i in range(pred_latency_detached.shape[0]):  # Loop through each sample in the batch
    #             # Prediction interval for latency
    #             interval_latency = t_value * torch.sqrt(
    #                 1 + (1 / len(pred_latency_detached)) + 
    #                 ((pred_latency_detached[i] - pred_latency_detached.mean(dim=0))**2).sum() / ((pred_latency_detached - pred_latency_detached.mean(dim=0))**2).sum()
    #             )
    #             intervals_latency.append(interval_latency.cpu().numpy())
           
    #         uncertainties.append(intervals_latency)
    #         index += batch_size
    # with open("uncertainties.txt", "w") as f:
    #     for interval_list in uncertainties:
    #         f.write(" ".join(map(str, interval_list)) + "\n")
    
    
    val_size = (int)(X.shape[0] * 0.1)
    X_test = X[-val_size:, :]
    y_d_test = y_d[-val_size:]
    y_l_test = y_l[-val_size:]

    X_train_data = X[:-val_size, :]
    y_d_train_data = y_d[:-val_size]
    y_l_train_data = y_l[:-val_size]
    
    
    n_chunks = 100
    chunk_size = len(X_train_data) // n_chunks
    
    # print("start chunk")
    # X_chunks = np.array_split(X_train_data, n_chunks)
    # y_d_chunks = np.array_split(y_d_train_data, n_chunks)
    # y_l_chunks = np.array_split(y_l_train_data, n_chunks)

    # Save the chunk indices (start and end index) in a separate list
    chunk_indices = [(i * chunk_size, (i + 1) * chunk_size) for i in range(n_chunks)]

    # For handling cases where len(X_train_data) is not perfectly divisible by n_chunks
    chunk_indices[-1] = (chunk_indices[-1][0], len(X_train_data))
    print("Chunk indices:", chunk_indices)
    
    
    

    initialize_trunk_indice = int(0.1 * len(chunk_indices))
    print("trainsize:")
    print(initialize_trunk_indice)

    indices = np.random.permutation(len(chunk_indices))
    
    
    active_indices = []
    active_trunk_indices = np.sort(indices[:initialize_trunk_indice])
    print("active_trunk_indices:")
    print(active_trunk_indices)

    # For each selected trunk, add the range of indices from that chunk to active_indices
    for trunk_idx in active_trunk_indices:
        start_idx, end_idx = chunk_indices[trunk_idx]
        active_indices.extend(range(start_idx, end_idx))
    
    # active_indices = indices[:initial_train_size]
    
    X_stable = X_train_data[active_indices]
    y_d_stable = y_d_train_data[active_indices]
    y_l_stable = y_l_train_data[active_indices]
    
    random_indices_simulation = np.random.choice(active_indices, size=1152, replace=False)
    # Select the sampled data points
    X_simulation = X_train_data[random_indices_simulation]
    y_d_simulation = y_d_train_data[random_indices_simulation]
    y_l_simulation = y_l_train_data[random_indices_simulation]
    
    remaining_indices = np.setdiff1d(active_indices, random_indices_simulation)
    
    # X_select = X_stable[active_indices]
    # y_d_select = y_d_stable[active_indices]
    # y_l_select = y_l_stable[active_indices]
    X_select = X_train_data[remaining_indices]
    y_d_select = y_d_train_data[remaining_indices]
    y_l_select = y_l_train_data[remaining_indices]
    
    pool_trunk_indices = list(set(range(len(chunk_indices))) - set(active_trunk_indices))
    pool_indices = list(set(range(len(X_train_data))) - set(active_indices))
    
    
    print("pool_indices:")
    print(len(pool_indices))
    print("pool_trunk_indices:")
    print(pool_trunk_indices)
    #select 10%
    cost_utility = 0
    #smooth
    prev_smoothed_delta_cost_ratio = None
    prev_smoothed_diff_sum_cost_ratio = None
    smooth_delta_cost_ratios = []  # List to store (delta / iter_training_cost) values
    smooth_diff_sum_cost_ratios = []
    window_size_al = 1
    #initialzie value
    iter_training_cost = 0
    current_uncertainty = 0
    last_uncertainty = 100
    #simulation choice
    sample_size = 1152  # Fixed sample size of 100
    dataset_size = len(X_train_data)
    apply_check = False
    simulation_epoch = 0
    iteration = 0
    window_diff = []
    window_delta = []
    
    for i in range(1000):
        print('EPOCH: ', (i + start_epoch))
        if apply_check == True:
            apply_check = False
            simulation_epoch = 0
            best_mape = 100
            
            #momentum
            prev_smoothed_delta_cost_ratio = None
            prev_smoothed_diff_sum_cost_ratio = None
            smooth_delta_cost_ratios = []  # List to store (delta / iter_training_cost) values
            smooth_diff_sum_cost_ratios = []
            
            #sample window
            window_diff = []
            window_delta = []
            
            
            window_size = 12
            #initialzie value
            iter_training_cost = 0
            current_uncertainty = 0
            last_uncertainty = 100
        ##
        epoch_time_start = time.time()
        uncertainties = []
        ##
        if is_log:
            train_log.write("\nEPOCH: %s\n\n" % (i + start_epoch))

        index = 0
        steps = (X_select.shape[0] - window_size) / batch_size

        loss_list = []
        loss_count = []

        while index + window_size <= X_select.shape[0]:
            cur_step = (int)(index/batch_size)

            data = \
                    train_both_data_generator(X_select, y_d_select, y_l_select, window_size, \
                                              index, batch_size)
            if data is None:
                break
            X_train, Y_d_train, Y_l_train = data
            X_train, Y_d_train, Y_l_train = \
                    torch.from_numpy(X_train), \
                    torch.from_numpy(Y_d_train), \
                    torch.from_numpy(Y_l_train)



            loss_value,uncertainty = train(model, X_train, Y_d_train, Y_l_train,
                               optimizer, alpha, drop_weight, latency_loss,
                               disc_factor)
            #
            uncertainties.append(uncertainty)

            #
            loss_list.append(loss_value)
            loss_count.append(len(X_train))
            if cur_step % 1000 == 0 or steps - cur_step <= 1:
                print('STEP: ', cur_step, '/', steps,
                         ' last loss: ', loss_value,
                         ' min loss: ', min(loss_list),
                         ' max loss: ', max(loss_list),
                         ' avg loss:', sum(loss_list)/ len(loss_list),
                         ' med loss:', statistics.median(loss_list))
            # if enable_validation:
            #     val_loss_value = test(model, X_test, y_d_test , y_l_test ,
            #                           alpha, drop_weight, latency_loss,
            #                           discretized_max = disc_factor)
            #     val_loss_list.append(val_loss_value)
            #     val_loss_count.append(len(X_test))
            index +=  batch_size
        epoch_time_end = time.time() 
        
        test_start_time = time.time()
        model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            val_loss_list = []
            val_loss_count = []
            val_index = 0
            while val_index + window_size <= X_test.shape[0]:
                data = train_both_data_generator(
                    X_test, y_d_test, y_l_test, window_size, val_index, batch_size)
                if data is None:
                    break
                X_val_batch, Y_d_val_batch, Y_l_val_batch = data
                X_val_batch = torch.from_numpy(X_val_batch)
                Y_d_val_batch = torch.from_numpy(Y_d_val_batch)
                Y_l_val_batch = torch.from_numpy(Y_l_val_batch)

                batch_size_actual = X_val_batch.size(0)
                
                val_loss_value = test(
                    model, X_val_batch, Y_d_val_batch, Y_l_val_batch,
                    alpha, drop_weight, latency_loss, disc_factor)
                val_loss_list.append(val_loss_value)
                val_loss_count.append(len(X_val_batch))
                
                total_loss += val_loss_value * batch_size_actual
                total_samples += batch_size_actual
                
                val_index += batch_size
        test_end_time = time.time()
        print('test_start_time {}'.format(test_start_time)) 
        print('test_end_time {}'.format(test_end_time)) 
        print('time duration {}'.format(test_end_time - test_start_time)) 
        avg_val_loss = sum(val_loss_list)/sum(val_loss_count)
        print(f"Epoch {i}: Validation Loss = {avg_val_loss}")
        avg_val_loss = total_loss / total_samples
        print(f"Epoch {i}: Validation Loss_3 = {avg_val_loss}")
        
        
        model.train()
        print('end_time {}'.format(epoch_time_end)) 
        print('start_time {}'.format(epoch_time_start)) 
        iter_training_cost = get_time_cost(iter_training_cost,epoch_time_end,epoch_time_start)

        
        
        print('iter_training_cost {}'.format(iter_training_cost))  
        print('length of current_uncertainty {}'.format(len(uncertainties)))
        
        percentile_95 = np.percentile(uncertainties, 95)
        capped_uncertainties = [min(u, percentile_95) for u in uncertainties]
        uncertainties = capped_uncertainties
        avg_val_loss = sum(val_loss_list) / len(val_loss_list)
        print(f"Epoch {i}: Validation Loss = {avg_val_loss}")
        current_uncertainty = sum(uncertainties)
        data_uncertainty,data_cost,new_labeled,sample_scale = uncertainties_simulation(X_train_data, y_d_train_data, y_l_train_data,dataset_size,sample_size,iter_training_cost,pool_indices,model,enable_validation,index,batch_size,window_size,len(X_select),random_indices_simulation)
        print('data_uncertainty {}'.format(data_uncertainty))
        print('data select in simulation {}'.format(new_labeled))
        # result, window_delta, window_diff = calculate_and_compare_metrics_sample_window(
        #     uncertainties,
        #     last_uncertainty,
        #     current_uncertainty,
        #     dataset_size,
        #     new_labeled,
        #     iter_training_cost,
        #     data_uncertainty,
        #     data_cost,
        #     sample_scale,
        #     window_diff,
        #     window_delta,
        #     window_size_al,
        #     simulation_epoch
        # )
        result, prev_smoothed_delta_cost_ratio, prev_smoothed_diff_sum_cost_ratio = calculate_and_compare_metrics(
            uncertainties,
            last_uncertainty,
            current_uncertainty,
            dataset_size,
            new_labeled,
            iter_training_cost,
            data_uncertainty,
            data_cost,
            sample_scale,
            prev_smoothed_delta_cost_ratio,
            prev_smoothed_diff_sum_cost_ratio,
            smooth_delta_cost_ratios,
            smooth_diff_sum_cost_ratios,
            window_size_al,
            simulation_epoch
        )
        simulation_epoch+=1
        if result == False:
            if iteration>=6:
                break
            number,new_indices,cost_utility = active_learning_iteration(cost_utility,iteration,model, X, y_d, y_l, pool_indices,chunk_indices,pool_trunk_indices)
            iteration+=1
            apply_check = True
            # active_indices = active_indices.append(new_indices)
            # active_trunk_indices.append(new_indices)
            active_trunk_indices = active_trunk_indices.tolist()
            active_trunk_indices.extend(new_indices)
            active_trunk_indices = np.array(active_trunk_indices)
            print("unsort_active_trunk_indices:")
            print(active_trunk_indices)
            active_trunk_indices = np.sort(active_trunk_indices)
            print("len of trunk_selected")
            print(len(active_trunk_indices))
            print("active_trunk_indices:")
            print(active_trunk_indices)
            active_indices = []
            for trunk_idx in active_trunk_indices:
                start_idx, end_idx = chunk_indices[trunk_idx]
                active_indices.extend(range(start_idx, end_idx))
            
            # active_indices = indices[:initial_train_size]
            
            
            # X_select = X_train_data[active_indices]
            # y_d_select = y_d_train_data[active_indices]
            # y_l_select = y_l_train_data[active_indices]
            
            pool_trunk_indices = list(set(range(len(chunk_indices))) - set(active_trunk_indices))
            pool_indices = list(set(range(len(X_train_data))) - set(active_indices))
            
            random_indices_simulation = np.random.choice(active_indices, size=1152, replace=False)
            remaining_indices = np.setdiff1d(active_indices, random_indices_simulation)
            
            
            X_select = X_train_data[remaining_indices]
            y_d_select = y_d_train_data[remaining_indices]
            y_l_select = y_l_train_data[remaining_indices]
            
            print("len of newdata")
            print(len(active_indices))
                        
            # with open(os.path.join(f'active_indices{iteration}.txt'), 'w') as f:
            #     for index in active_indices:
            #         f.write(f"{index}\n")
            directory = '/mydata/MimicNet/cost_test/normalize_0.1_al_cost'
            file_path = os.path.join(directory, f'active_indices{iteration}.txt')

            # Ensure the directory exists
            os.makedirs(directory, exist_ok=True)
            
            with open(file_path, 'w') as f:
                for index in active_indices:
                    f.write(f"{index}\n")
            
            
        else:
            last_uncertainty = current_uncertainty
        
        
        print("Saving model...")
        save_ckpt(model_name + "_epoch" + str(start_epoch+i+1) + ".ckpt",
                  model, start_epoch + i,
                  dis_meta_last, dis_meta_ewma, dis_meta_latency)
        save_hdf5(model_name + "_epoch" + str(start_epoch+i+1) + ".hdf5",
                  model, device, start_epoch + i,
                  dis_meta_last, dis_meta_ewma, dis_meta_latency)

        print("Current train loss:", sum(loss_list)/sum(loss_count))

    
    end_time = time.time() 
    print('end_time {}'.format(end_time)) 
    
    print("Saving final model...")
    save_ckpt(model_name + ".ckpt", model, start_epoch + num_epochs,
              dis_meta_last, dis_meta_ewma, dis_meta_latency)
    save_hdf5(model_name + ".hdf5", model, device, start_epoch + num_epochs,
              dis_meta_last, dis_meta_ewma, dis_meta_latency)

    if is_log:
        train_log.close()
