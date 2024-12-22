from utils import *
from params import *
from snail import SnailFewShot, SnailFewShotOnSizeless, Sizeless, RNNFewShot
from torchmetrics import MeanAbsolutePercentageError
import argparse
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import time
from blocks import *
import matplotlib.pyplot as plt
import torch.nn.functional as F
import csv
import random
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset, Subset
from scipy.stats import t

mean_abs_percentage_error = MeanAbsolutePercentageError()

def init_model(opt, device):
    # X-shot regression: num_cls = 1, num_shots = X
    num_features = NUM_FEATURES_PER_SHOT * opt.num_shots + NUM_CONFIG_PARAMS
    print('Total number of features:', num_features)
    if opt.skip_embedding:
        # reproduce Sizeless, which is 1-shot regression
        print('Initializing Sizeless model...')
        model = Sizeless(num_features)
    elif opt.sizeless:
        # SNAIL + sizeless
        print('Initializing SNAIL-based meta-learning model on Sizeless...')
        model = SnailFewShotOnSizeless(opt.num_cls, opt.num_shots, opt.dataset, use_cuda=opt.cuda)
    elif opt.rnn:
        # RNN + fully-connected NN
        print('Initializing RNN-based meta-learning model...')
        model = RNNFewShot(opt.num_cls, opt.num_shots, opt.dataset, device)
    else:
        # SNAIL + fully-connected NN
        print('Initializing SNAIL-based meta-learning model...')
        model = SnailFewShot(opt.num_cls, opt.num_shots, opt.dataset, use_cuda=opt.cuda)
    model = model.to(device)
    return model


def AL_intrain(sampled_uncertainties, budget, sampled_costs):
    selected_uncertainties = []
    total_cost = 0
    budget = budget
    sqrt_sampled_costs = sampled_costs
    # Calculate utility (uncertainty/cost) for each sample
    utilities = [u / c if c != 0 else 0 for u, c in zip(sampled_uncertainties, sampled_costs)]
    
    # Sort utilities and their corresponding uncertainties and costs by utility value
    sorted_indices = np.argsort(utilities)[::-1]  # Sort in descending order of utility
    sorted_utilities = np.array(utilities)[sorted_indices]
    sorted_uncertainties = np.array(sampled_uncertainties)[sorted_indices]
    
    sorted_costs = np.array(sqrt_sampled_costs)[sorted_indices]
    
    num_exclude = int(0.05 * len(sorted_utilities))  # Calculate the number of samples to exclude (top 5%)
    sorted_utilities = sorted_utilities[num_exclude:]
    sorted_uncertainties = sorted_uncertainties[num_exclude:]
    sorted_costs = sorted_costs[num_exclude:]
    
    i = 0
    # Select based on utility without exceeding the budget
    for utility, uncertainty, cost in zip(sorted_utilities, sorted_uncertainties, sorted_costs):
        if total_cost + cost <= budget:
            selected_uncertainties.append(uncertainty)
            total_cost += cost
            i += 1
        else:
            remaining = budget - total_cost
            fraction = remaining / cost
            selected_uncertainties.append(uncertainty * fraction)
            total_cost += remaining
            i+=fraction
            break
        print('total_cost {}'.format(total_cost))
    print('number of data select in sample {}'.format(i))
    # Return the sum of the selected uncertainties
    return sum(selected_uncertainties),total_cost,i

def uncertainty_select(model, dataloader, opt,i):
    model.eval()
    uncertainties = []
    prediction_list = []
    
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x_validate = x[:, :-1] if len(x.shape) > 1 else x[:-1]
            x_validate, y, last_targets = batch_for_few_shot(opt, x_validate, y)
            model_output = model(x_validate, y)
            last_model = model_output[:, -1, :].squeeze(0)
            predictions = last_model.view(-1)
            prediction_list.extend(predictions.cpu().tolist())
        
        # Convert x_list to a PyTorch tensor
        prediction_list = torch.tensor(prediction_list)
        
        confidence_level = 0.95
        t_value = t.ppf((1 + confidence_level) / 2, len(prediction_list) - 1)

        intervals = []
        for i in range(prediction_list.shape[0]):  # Loop through each sample in the batch
            # Prediction interval for latency
            interval = t_value * torch.sqrt(
                1 + (1 / len(prediction_list)) + 
                ((prediction_list[i] - prediction_list.mean(dim=0)) ** 2).sum() / ((prediction_list - prediction_list.mean(dim=0)) ** 2).sum()
            )
            intervals.append(interval.cpu().numpy())

        uncertainties.extend(intervals)
        # with open("uncertainties_prediction_{}.txt".format(i), "w") as f:
        #     for interval in uncertainties:
        #         f.write(str(interval) + "\n")
    
    return uncertainties

def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)

def batch_for_few_shot(opt, x, y):
    last_targets = y.view(-1, 1).float()  # Convert to FloatTensor
    x, y = x.float(), y.float()  # Ensure x and y are also FloatTensors
    x, y = Variable(x), Variable(y)
    if opt.cuda:
        x, y = x.cuda(), y.cuda()
        last_targets = last_targets.cuda()
    return x, y, last_targets

def get_mape(last_model, last_targets):
    # Ensure both inputs are PyTorch tensors (keep them in tensor form)
    if not isinstance(last_model, torch.Tensor):
        last_model = torch.tensor(last_model)
    if not isinstance(last_targets, torch.Tensor):
        last_targets = torch.tensor(last_targets)

    if last_model.dim() == 0:
        last_model = last_model.unsqueeze(0)
    if last_targets.dim() == 0:
        last_targets = last_targets.unsqueeze(0)
    
    # Flatten both tensors if necessary
    last_model = last_model.view(-1)
    last_targets = last_targets.view(-1)
    
    # Ensure shapes match after flattening
    if last_model.shape != last_targets.shape:
        raise ValueError(f"Shape mismatch: last_model shape {last_model.shape}, last_targets shape {last_targets.shape}")

    # Calculate MAPE using PyTorch operations
    mape = torch.mean(torch.abs((last_targets - last_model) / last_targets)) * 100
    return mape.item()


# def test(opt, test_dataloader, model):
#     model.eval()
#     avg_mape = list()
#     for epoch in range(10):
#         test_iter = iter(test_dataloader)
#         for batch in test_iter:
#             x, y = batch
#             x, y, last_targets = batch_for_few_shot(opt, x, y)
#             model_output = model(x, y)
#             # last_model = model_output[:, -1, :].squeeze(0)
#             avg_mape.append(get_mape(last_model, last_targets))
#             last_model = model_output.squeeze(1)  # Adjust output shape
#             batch_mape = get_mape(last_model.detach().cpu().numpy(), last_targets.detach().cpu().numpy())
#     avg_mape = np.mean(batch_mape)
#     print('Test Avg MAPE: {}'.format(avg_mape))

#     return avg_mape
def test(opt, test_dataloader, model):
    # avg_mape = list()
    # for epoch in range(10):
    #     test_iter = iter(test_dataloader)
    #     for batch in test_iter:
    #         x, y = batch
    #         x_validate = x[:, :-1] if len(x.shape) > 1 else x[:-1]
    #         x_validate, y, last_targets = batch_for_few_shot(opt, x_validate, y)
    #         model_output = model(x_validate, y)
    #         last_model = model_output[:, -1, :].squeeze(0)
    #         avg_mape.append(get_mape(last_model, last_targets))
    # avg_mape = np.mean(avg_mape)
    # print('Test Avg MAPE: {}'.format(avg_mape))

    # return avg_mape
    model.eval()
    with torch.no_grad():
        avg_mape_first_epoch = []
        for batch_test in test_dataloader:
            x, y = batch_test
            x_validate = x[:, :-1] if len(x.shape) > 1 else x[:-1]
            x_validate, y, last_targets = batch_for_few_shot(opt, x_validate, y)
            model_output = model(x_validate, y)
                            
            last_model = model_output.squeeze(1)  # Adjust output shape
                            
                            # Calculate MAPE for the current batch and append to avg_mape
            batch_mape = get_mape(last_model.cpu().numpy(), last_targets.cpu().numpy())
            avg_mape_first_epoch.append(batch_mape)

        # Compute average MAPE over all batches
        avg_mape_first_epoch = np.mean(avg_mape_first_epoch)
        print(f'Test Set MAPE = {avg_mape_first_epoch:.4f}')
    return avg_mape_first_epoch



def original_train(opt, tr_dataloader, model, optim, val_dataloader=None):
    best_state = None  # Initialize best_state to None

    train_loss = []
    train_mape = []
    val_loss = []
    val_mape = []
    best_mape = 500

    train_loss_per_epoch = []
    train_mape_per_epoch = []

    best_model_path = os.path.join(opt.exp, 'best_model.pth')
    last_model_path = os.path.join(opt.exp, 'last_model.pth')
    with open('training_metrics_1.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
            
            # Write the header of the CSV file
        writer.writerow(['Epoch', 'Avg Val MAPE'])

        loss_fn = nn.MSELoss()

        for epoch in range(5):
            print('=== Epoch: {} ==='.format(epoch))
            tr_iter = iter(tr_dataloader)
            model.train()
            model = model.cuda() if opt.cuda else model
            for batch in tqdm(tr_iter):
                optim.zero_grad()
                x, y = batch
                x, y, last_targets = batch_for_few_shot(opt, x, y)
                model_output = model(x, y)
                last_model = model_output.squeeze(1)  # Remove the singleton dimension

                loss = loss_fn(last_model, last_targets)
                loss.backward()
                optim.step()
                train_loss.append(loss.item())
                train_mape.append(get_mape(last_model, last_targets))
            avg_loss = np.mean(train_loss[-opt.iterations:])
            avg_mape = np.mean(train_mape[-opt.iterations:])
            print('Avg Train Loss: {}, Avg Train MAPE: {}'.format(avg_loss, avg_mape))

            if val_dataloader is None:
                continue
            val_iter = iter(val_dataloader)
            model.eval()
            for batch in val_iter:
                x, y = batch
                x, y, last_targets = batch_for_few_shot(opt, x, y)
                model_output = model(x, y)
                last_model = model_output.squeeze(1)

                loss = loss_fn(last_model, last_targets)
                val_loss.append(loss.item())
                val_mape.append(get_mape(last_model, last_targets))
                
            avg_loss = np.mean(val_loss[-opt.iterations:])
            avg_mape = np.mean(val_mape[-opt.iterations:])
            postfix = ' (Best)' if avg_mape <= best_mape else ' (Best: {})'.format(best_mape)
            print('Avg Val Loss: {}, Avg Val MAPE: {}{}'.format(avg_loss, avg_mape, postfix))
            train_loss_per_epoch.append(avg_loss)
            train_mape_per_epoch.append(avg_mape)
            if avg_mape <= best_mape:
                torch.save(model.state_dict(), best_model_path)
                best_mape = avg_mape
                best_state = model.state_dict()
            for name in ['train_loss', 'train_mape', 'val_loss', 'val_mape', 'train_loss_per_epoch', 'train_mape_per_epoch']:
                save_list_to_file(os.path.join(opt.exp, name + '.txt'), locals()[name])
            writer.writerow([epoch, avg_mape])

        torch.save(model.state_dict(), last_model_path)
    
    return best_state, best_mape, train_loss_per_epoch, train_mape_per_epoch


def train_on_test_dataloader(opt, test_training_dataloader, test_test_dataloader, model, optim, freeze_layers=True):
    loss_fn = nn.MSELoss()
    all_epoch_losses = []
    test_accuracies = []

    # Optionally freeze certain layers
    if freeze_layers:
        for param in model.fc1.parameters():  # Example: Freeze the first fully connected layer
            param.requires_grad = False

    for epoch in range(30):  # Train on test_training_dataloader for 50 epochs
        model.train()
        epoch_loss = []
        
        for batch in test_training_dataloader:
            optim.zero_grad()
            x, y = batch
            x, y, last_targets = batch_for_few_shot(opt, x, y)
            model_output = model(x, y)
            last_model = model_output.squeeze(1)  # Adjust output shape
            loss = loss_fn(last_model, last_targets)
            loss.backward()
            optim.step()
            epoch_loss.append(loss.item())

        avg_loss = np.mean(epoch_loss)
        all_epoch_losses.append(avg_loss)
        print(f'Epoch {epoch + 1}: Avg Training Loss on Test Data = {avg_loss:.4f}')
        
        # Every 5 epochs, evaluate MAPE on test_test_dataloader
        if (epoch + 1) % 5 == 0:

            model.eval()
            with torch.no_grad():
                avg_mape = []
                for batch in test_test_dataloader:
                    x, y = batch
                    x, y, last_targets = batch_for_few_shot(opt, x, y)
                    model_output = model(x, y)
                    last_model = model_output.squeeze(1)  # Adjust output shape
                    
                    # Calculate MAPE for the current batch and append to avg_mape
                    batch_mape = get_mape(last_model.cpu().numpy(), last_targets.cpu().numpy())
                    avg_mape.append(batch_mape)

                # Compute average MAPE over all batches
                avg_mape_epoch = np.mean(avg_mape)
                test_accuracies.append(avg_mape_epoch)
                print(f'Epoch {epoch + 1}: Test Set MAPE = {avg_mape_epoch:.4f}')

        # Optionally reduce the learning rate dynamically
        if epoch > 0 and epoch % 10 == 0:
            for param_group in optim.param_groups:
                param_group['lr'] *= 0.1  # Reduce the learning rate by 10x after every 10 epochs

    # Save the results to CSV
        with open("test.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Test Accuracy"])  # Header for the CSV file
            for epoch_idx, accuracy in enumerate(test_accuracies, start=5):
                writer.writerow([epoch_idx * 5, accuracy])  # Write accuracy only for every 5th epoch

    return model, test_accuracies


def uncertainty(model, dataloader, opt):
    model.eval()
    uncertainties = []
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x_validate= x[:, :-1] if len(x.shape) > 1 else x[:-1]
            x_validate, y, last_targets = batch_for_few_shot(opt, x_validate, y)
            model_output = model(x_validate, y)
            last_model = model_output[:, -1, :].squeeze(0)
            true_values = y.view(-1)
            predictions = last_model.view(-1)
            variances = (true_values - predictions) ** 2
            uncertainties.extend(variances.cpu().tolist())
    
    return uncertainties


def train(simulation_indices,i,opt, tr_dataloader, model,whole_dataloader,pool_indices, optim, test_test_dataloader,val_dataloader=None):
   
    best_state = model.state_dict()
    train_loss = []
    train_mape = []
    val_loss = []
    val_mape = []
    best_mape = 1000
    check = True
    train_loss_per_epoch = []
    train_mape_per_epoch = []
    iter_training_cost = 0
    current_uncertainty = 0
    last_uncertainty = 100
    prev_smoothed_delta_cost_ratio = None
    prev_smoothed_diff_sum_cost_ratio = None
    smooth_delta_cost_ratios = []  # List to store (delta / iter_training_cost) values
    smooth_diff_sum_cost_ratios = []
    best_model_path = os.path.join(opt.exp, f'best_model_{i}.pth')
    last_model_path = os.path.join(opt.exp, f'last_model_{i}.pth')
    
    window_size = 1  # Size of the moving average window
    delta_cost_ratios = []  # List to store (delta / iter_training_cost) values
    diff_sum_cost_ratios = []
    
    counter = 0
    loss_fn = nn.MSELoss()
    sample_size = len(simulation_indices)  # Fixed sample size of 100
    dataset_size = len(tr_dataloader.dataset)
    # sampled_indices = random.sample(range(dataset_size), sample_size)
    # sampled_clients = [tr_dataloader.dataset[i] for i in sampled_indices]

    for epoch in range(opt.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        epoch_time_start = time.time()
        uncertainties = []
        tr_iter = iter(tr_dataloader)
        model.train()
        model = model.cuda() if opt.cuda else model
        
        if epoch != 0:
            last_uncertainty = current_uncertainty
        
        for batch in tqdm(tr_iter):
            optim.zero_grad()
            x, y = batch
            
            # x, y, last_targets = batch_for_few_shot(opt, x, y)
            # model_output = model(x, y)
            x_train = x[:, :-1] if len(x.shape) > 1 else x[:-1]
            x_train, y, last_targets = batch_for_few_shot(opt, x_train, y)
            model_output = model(x_train, y)
            
            last_model = model_output[:, -1, :].squeeze(0)
            
            true_values = y.view(-1)
            predictions = last_model.view(-1)
            variances = (true_values - predictions) ** 2
            uncertainties.extend(variances.cpu().tolist())
            
            
            loss = loss_fn(last_model, last_targets)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            train_mape.append(get_mape(last_model, last_targets))
        avg_loss = np.mean(train_loss[-opt.iterations:])
        avg_mape = np.mean(train_mape[-opt.iterations:])
        print('Avg Train Loss: {}, Avg Train MAPE: {}'.format(avg_loss, avg_mape))
        train_loss_per_epoch.append(avg_loss)
        train_mape_per_epoch.append(avg_mape)
        
        current_uncertainty = sum(uncertainties)
        epoch_time_end = time.time()  
        epoch_duration = epoch_time_end - epoch_time_start 
        epoch_duration_hours = epoch_duration / 3600 
        iter_training_cost += epoch_duration_hours * 0.8   
        print('epoch_duration: {},: iter_training_cost {}'.format(epoch_duration, iter_training_cost))
        
        if val_dataloader is None:
            continue
        val_iter = iter(val_dataloader)
        model.eval()
        for batch in val_iter:
            x, y = batch
            x, y, last_targets = batch_for_few_shot(opt, x, y)
            model_output = model(x, y)
            # x_validate= x[:, :-1] if len(x.shape) > 1 else x[:-1]
            # x_validate, y, last_targets = batch_for_few_shot(opt, x_validate, y)
            # model_output = model(x_validate, y)
            last_model = model_output[:, -1, :].squeeze(0)
            loss = loss_fn(last_model, last_targets)
            val_loss.append(loss.item())
            val_mape.append(get_mape(last_model, last_targets))
        avg_loss = np.mean(val_loss[-opt.iterations:])
        avg_mape = np.mean(val_mape[-opt.iterations:])
        postfix = ' (Best)' if avg_mape <= best_mape else ' (Best: {})'.format(best_mape)
        print('Avg Val Loss: {}, Avg Val MAPE: {}{}'.format(avg_loss, avg_mape, postfix))
        
        val_end_time = time.time()
        print(f"Validation ended at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(val_end_time))}")
        
        if avg_mape <= best_mape:
            torch.save(model.state_dict(), best_model_path)
            best_mape = avg_mape
            best_state = model.state_dict()
            
            
        mean_uncertainty = np.mean(uncertainties)
        delta =  last_uncertainty - current_uncertainty
        print('current_uncertainty: {},: last_uncertainty {}'.format(current_uncertainty, last_uncertainty))
        ##Random select
                
        sample_scale = dataset_size / sample_size
        budget = iter_training_cost / sample_scale
        
        
        ###cost for data
        sampled_costs = []
        # for l in sampled_indices:
        #     row = df.iloc[l]
        #     cost = calculate_total_cost(row)
        #     sampled_costs.append(cost)
        # sampled_costs = [whole_dataloader.dataset[idx][0] for idx in sampled_indices]
        # for idx in simulation_indices:
        #     fake_cost = whole_dataloader.dataset[idx][0]  # [1] accesses the cost label
        #     sampled_costs.append(fake_cost[9]+fake_cost[20]+fake_cost[31])
        for idx in simulation_indices:
            fake_cost = whole_dataloader.dataset[idx][0]  # Get the feature vector
            sampled_costs.append(fake_cost[-1])  # Get the last column value as cost

            
        
        
        ### uncertainty
        sampled_uncertainties = []
        model.eval()

        with torch.no_grad():
            for w in simulation_indices:
                x_sample, y_sample = whole_dataloader.dataset[w]
                
                # Convert x_sample to PyTorch tensor and exclude last column
                if isinstance(x_sample, np.ndarray):
                    x_sample = torch.from_numpy(x_sample).float()
                elif not isinstance(x_sample, torch.Tensor):
                    x_sample = torch.tensor(x_sample).float()
                    
                # Handle dimensionality - check if tensor is 1D or 2D
                if x_sample.dim() == 1:
                    x_sample = x_sample[:-1]  # For 1D tensor
                else:
                    x_sample = x_sample[:, :-1]  # For 2D tensor
                
                # Convert y_sample to PyTorch tensor
                if isinstance(y_sample, (np.ndarray, np.float64, np.float32)):
                    y_sample = torch.tensor(y_sample).float()
                elif isinstance(y_sample, float):
                    y_sample = torch.tensor([y_sample]).float()
                elif not isinstance(y_sample, torch.Tensor):
                    y_sample = torch.tensor(y_sample).float()
                
                x_sample = x_sample.unsqueeze(0)  # Add batch dimension
                y_sample = y_sample.unsqueeze(0)  # Add batch dimension
                
                x_sample, y_sample, _ = batch_for_few_shot(opt, x_sample, y_sample)
                model_output = model(x_sample, y_sample)
                last_model_sample = model_output[:, -1, :].squeeze(0)
                true_values_sample = y_sample.view(-1)
                predictions_sample = last_model_sample.view(-1)
                variance_sample = (true_values_sample - predictions_sample) ** 2
                sampled_uncertainties.append(variance_sample.mean().item())
                
        if (epoch + 1) % 1 == 0:
            model.eval()
            with torch.no_grad():
                avg_mape = []
                for batch in test_test_dataloader:
                    x, y = batch
                    # x, y, last_targets = batch_for_few_shot(opt, x, y)
                    # model_output = model(x, y)
                    x_validate = x[:, :-1] if len(x.shape) > 1 else x[:-1]
                    x_validate, y, last_targets = batch_for_few_shot(opt, x_validate, y)
                    model_output = model(x_validate, y)
                    last_model = model_output.squeeze(1)  # Adjust output shape
                    
                    # Calculate MAPE for the current batch and append to avg_mape
                    batch_mape = get_mape(last_model.cpu().numpy(), last_targets.cpu().numpy())
                    avg_mape.append(batch_mape)

                # Compute average MAPE over all batches
                avg_mape_epoch = np.mean(avg_mape)
                print(f'Epoch {epoch + 1}: Test Set MAPE = {avg_mape_epoch:.4f}')
        
        data_uncertainty,data_cost,new_labeled = AL_intrain(sampled_uncertainties,budget,sampled_costs)
        
        ##
        print('budget: {}'.format(budget))
        print('delta: {}'.format(delta))
        print('sample_scale: {}'.format(sample_scale))
        print('data_uncertainty: {},: mean_uncertainty {}'.format(data_uncertainty, mean_uncertainty))
        print('i: {}'.format(i))
 

        mean_simulation = data_uncertainty / new_labeled
        data_cost = data_cost / new_labeled
        
        diff = delta + (mean_simulation-mean_uncertainty) * new_labeled * sample_scale
        size_increase = (dataset_size + new_labeled) / dataset_size
        print('size_increase: {},: data_cost {}'.format(size_increase, data_cost))
        sum_cost = iter_training_cost*size_increase + data_cost * new_labeled * sample_scale
        print('diff: {},: sum_cost {}'.format(diff, sum_cost))
        print('first: {},: second {}'.format(delta / iter_training_cost, diff / sum_cost))
        # if (epoch + 1) % 10 == 0 and (delta / iter_training_cost) < (diff / sum_cost):
        delta_cost_ratio = delta / iter_training_cost
        delta_cost_ratios.append(delta_cost_ratio) #first
        diff_sum_cost_ratio = diff / sum_cost
        diff_sum_cost_ratios.append(diff_sum_cost_ratio) #second
        
        # if len(delta_cost_ratios) >= window_size:
        #     # Calculate moving averages
        #     avg_delta_cost_ratio = np.mean(delta_cost_ratios[-window_size:])
        #     avg_diff_sum_cost_ratio = np.mean(diff_sum_cost_ratios[-window_size:])

        #     print(f'Moving Avg (delta/cost): {avg_delta_cost_ratio}, Moving Avg (diff/sum_cost): {avg_diff_sum_cost_ratio}')

        #     # Check condition using moving averages
        #     if avg_delta_cost_ratio < avg_diff_sum_cost_ratio:
        #         print("Exit by less delta (moving average)")
        #         break

        # # If we don't have enough values yet, print a message
        # else:
        #     print(f'Collecting data for moving average. Current window: {len(delta_cost_ratios)}/{window_size}')
        # # # if (delta / iter_training_cost) < (diff / sum_cost):
        #     counter+=1
        #     if counter == 5:
        #         print("exit by less delta")
        #         break 
        #     else:
        #         counter = 0   

            
        if epoch != 0:
            if prev_smoothed_delta_cost_ratio is None:
                smoothed_delta_cost_ratio = delta_cost_ratio
            else:
                smoothed_delta_cost_ratio = 0.9 * prev_smoothed_delta_cost_ratio + (1-0.9)*delta_cost_ratio
            if prev_smoothed_diff_sum_cost_ratio is None:
                smoothed_diff_sum_cost_ratio = diff_sum_cost_ratio
            else:
                smoothed_diff_sum_cost_ratio = 0.9 * prev_smoothed_diff_sum_cost_ratio + (1-0.9)*diff_sum_cost_ratio
            
            smooth_delta_cost_ratios.append(smoothed_delta_cost_ratio) #first
            smooth_diff_sum_cost_ratios.append(smoothed_diff_sum_cost_ratio) #second
            
            prev_smoothed_delta_cost_ratio = smoothed_delta_cost_ratio
            prev_smoothed_diff_sum_cost_ratio = smoothed_diff_sum_cost_ratio
            
            
            if len(smooth_delta_cost_ratios) >= window_size:
                # Calculate moving averages
                avg_delta_cost_ratio = np.mean(smooth_delta_cost_ratios[-window_size:]) #first
                avg_diff_sum_cost_ratio = np.mean(smooth_diff_sum_cost_ratios[-window_size:]) #second
                print("delta_cost_ratios[-window_size:]:", smooth_delta_cost_ratios[-window_size:])
                print("diff_sum_cost_ratios[-window_size:]:", smooth_diff_sum_cost_ratios[-window_size:])
                print(f'Moving Avg (delta/cost): {avg_delta_cost_ratio}, Moving Avg (diff/sum_cost): {avg_diff_sum_cost_ratio}')

                # Check condition using moving averages
                if avg_delta_cost_ratio < avg_diff_sum_cost_ratio:
                    print("Exit by less delta (moving average)")
                    break
            else:
                print(f'Collecting data for moving average. Current window: {len(smooth_delta_cost_ratios)}/{window_size}')
            
        for name in ['train_loss', 'train_mape', 'val_loss', 'val_mape', 'train_loss_per_epoch', 'train_mape_per_epoch']:
            save_list_to_file(os.path.join(opt.exp, name + '.txt'), locals()[name])

    torch.save(model.state_dict(), last_model_path)

    return check,best_state, best_mape, train_loss, train_mape, val_loss, val_mape, train_loss_per_epoch, train_mape_per_epoch



def active_learning_iteration(cost_utility, i, model, dataloader, pool_indices, batch_size, options):
    print(f"Starting active learning iteration {i}", flush=True)
    pool_loader = DataLoader(Subset(dataloader.dataset, pool_indices), batch_size=batch_size, shuffle=False)
    print("Calculating uncertainty values...", flush=True)
    uncertainty_values = uncertainty_select(model, pool_loader, options,i)
   
    print("Calculating costs...", flush=True)
    
    costs = []
    for idx in pool_indices:
        fake_cost = dataloader.dataset[idx][0]  # Get the feature vector
        costs.append(fake_cost[-1])  # Get the last column value
    print("First 10 cost:", costs[:10], flush=True)
    # costs = [calculate_total_cost(df.iloc[idx]) for idx in pool_indices]
    
    print(f"pool_indices: {len(pool_indices)}", flush=True)
    print(f"Costs: {len(costs)}", flush=True)
    print(f"uncertainty_values: {len(uncertainty_values)}", flush=True)
    print("Sample of Uncertainty values:", flush=True)
    print("First 10:", uncertainty_values[:10], flush=True)
    assert len(uncertainty_values) == len(costs), "Mismatch between number of uncertainty values and costs"
    
    print("Calculating uncertainty-cost ratios...", flush=True)
    uncertainty_cost_ratios = [u / c for u, c in zip(uncertainty_values, costs)]
    
    num_to_select = int(0.1 * len(dataloader.dataset))
    print(f"Number to select: {num_to_select}", flush=True)
    number_cluster = int(np.sqrt(10*num_to_select))
    print(f"number_cluster: {number_cluster}", flush=True)
    if i == 0:
        print("First iteration (i == 0)", flush=True)
        print("Sorting indices...", flush=True)
        sorted_indices = np.argsort(uncertainty_cost_ratios)[::-1][:10*num_to_select]
        print("Selecting data for clustering...", flush=True)
        selected_data = [dataloader.dataset[pool_indices[idx]][0] for idx in sorted_indices]
        
        print("Starting K-means clustering...", flush=True)
        kmeans = KMeans(n_clusters=number_cluster, max_iter=300, random_state=0).fit(selected_data)
        print("K-means clustering completed", flush=True)
        print(f"Cluster labels: {kmeans.labels_}", flush=True)
        print("Calculating distances...", flush=True)
        distances = cdist(selected_data, kmeans.cluster_centers_)
        print("Sorting cluster indices...", flush=True)
        sorted_cluster_indices = []
        for cluster_id in range(number_cluster):
            cluster_points = np.where(kmeans.labels_ == cluster_id)[0]
            print(f"Cluster {cluster_id}: {len(cluster_points)} points", flush=True)
            if len(cluster_points) > 0:
                sorted_cluster_points = sorted(cluster_points, key=lambda x: distances[x, cluster_id])
                sorted_cluster_indices.append([sorted_indices[idx] for idx in sorted_cluster_points])
        
        print("Selecting data points...", flush=True)
        actual_selected = []
        iteration_count = 0
        total_cost = 0
        while True:
            for cluster in sorted_cluster_indices:
                if len(actual_selected) >= num_to_select:
                    break
                if len(cluster) > 0:
                    idx = cluster.pop(0)
                    actual_selected.append(pool_indices[idx])
                    cost_index = pool_indices.index(pool_indices[idx])
                    print(f"Selected idx: {pool_indices[idx]}, Cost: {costs[cost_index]}", flush=True)
                    total_cost += costs[cost_index]
            
            iteration_count += 1
            print(f"Iteration count: {iteration_count}, selected so far: {len(actual_selected)}, total_cost: {total_cost}", flush=True)
            if len(actual_selected) >= num_to_select:
                break
        
        print("Selection completed", flush=True)
        print("Selected indices:", [idx for idx in actual_selected], flush=True)
        print("Corresponding normalized uncertainty-cost ratios:", [uncertainty_cost_ratios[pool_indices.index(idx)] for idx in actual_selected])  
        cost_utility = sum(costs[pool_indices.index(idx)] for idx in actual_selected)
    else:
        print(f"Iteration {i}", flush=True)
        print("Sorting indices...", flush=True)
        sorted_indices = np.argsort(uncertainty_cost_ratios)[::-1]
        print("Selecting data for clustering...", flush=True)
        selected_data = [dataloader.dataset[pool_indices[idx]][0] for idx in sorted_indices]
        print("Starting K-means clustering...", flush=True)
        kmeans = KMeans(n_clusters=number_cluster, max_iter=300, random_state=0).fit(selected_data)
        print("K-means clustering completed", flush=True)
        print("Calculating distances...", flush=True)
        distances = cdist(selected_data, kmeans.cluster_centers_)
        
        print("Sorting cluster indices...", flush=True)
        sorted_cluster_indices = []
        for cluster_id in range(number_cluster):
            cluster_points = np.where(kmeans.labels_ == cluster_id)[0]
            if len(cluster_points) > 0:
                sorted_cluster_points = sorted(cluster_points, key=lambda x: distances[x, cluster_id])
                sorted_cluster_indices.append([sorted_indices[idx] for idx in sorted_cluster_points])
        
        print("Selecting data points...", flush=True)
        cumulative_utility = 0
        actual_selected = []
        iteration = 0
        # while True:
        #     for cluster in sorted_cluster_indices:
        #         # print(f"Cluster length: {len(cluster)}",flush=True)
        #         if len(cluster) > 0:
        #             idx = cluster.pop(0)
        #             if cumulative_utility + costs[idx] > cost_utility or len(actual_selected) > num_to_select:
        #                 break
        #             cumulative_utility += costs[idx]
        #             iteration += 1
        #             print(f"iteration: {iteration}, idx: {idx}, cost: {costs[idx]}, cumulative_utility: {cumulative_utility}", flush=True)
        #             actual_selected.append(pool_indices[idx])
        #     else:
        #         continue
        while True:
            all_empty = True  # Flag to check if all clusters are empty
            for cluster in sorted_cluster_indices:
                if len(cluster) > 0:
                    all_empty = False  # At least one cluster has data
                    idx = cluster.pop(0)
                    if cumulative_utility + costs[idx] > cost_utility or len(actual_selected) > num_to_select:
                        break
                    cumulative_utility += costs[idx]
                    iteration += 1
                    print(f"iteration: {iteration}, idx: {idx}, cost: {costs[idx]}, cumulative_utility: {cumulative_utility}", flush=True)
                    actual_selected.append(pool_indices[idx])
            else:
                if all_empty:
                    break  # Exit the while loop if all clusters are empty
                continue
            break
        
    print(f"Final cost_utility: {cost_utility}", flush=True)
    print(f"Number of selected points: {len(actual_selected)}", flush=True)
    return num_to_select, actual_selected, cost_utility
def active_learning_iteration_no_clustering(cost_utility, i, model, dataloader, pool_indices, batch_size, options):
    print(f"Starting active learning iteration {i}", flush=True)
    pool_loader = DataLoader(Subset(dataloader.dataset, pool_indices), batch_size=batch_size, shuffle=False)
    print("Calculating uncertainty values...", flush=True)
    uncertainty_values = uncertainty_select(model, pool_loader, options,i)
   
    print("Calculating costs...", flush=True)
    
    costs = []
    for idx in pool_indices:
        fake_cost = dataloader.dataset[idx][0]  # Get the feature vector
        costs.append(fake_cost[-1])  # Get the last column value
    print("First 10 cost:", costs[:10], flush=True)
    # costs = [calculate_total_cost(df.iloc[idx]) for idx in pool_indices]
    
    print(f"pool_indices: {len(pool_indices)}", flush=True)
    print(f"Costs: {len(costs)}", flush=True)
    print(f"uncertainty_values: {len(uncertainty_values)}", flush=True)
    print("Sample of Uncertainty values:", flush=True)
    print("First 10:", uncertainty_values[:10], flush=True)
    assert len(uncertainty_values) == len(costs), "Mismatch between number of uncertainty values and costs"
    
    print("Calculating uncertainty-cost ratios...", flush=True)
    uncertainty_cost_ratios = [u / c for u, c in zip(uncertainty_values, costs)]
    
    num_to_select = int(0.1 * len(dataloader.dataset))
    print(f"Number to select: {num_to_select}", flush=True)

    total_utility = sum(uncertainty_cost_ratios)
    probabilities = [u / total_utility for u in uncertainty_cost_ratios]
    
    if i == 0:
        print("First iteration (i == 0)", flush=True)
        print("Sorting indices...", flush=True)
        sorted_indices = np.argsort(uncertainty_cost_ratios)[::-1][:10*num_to_select]
        print("Selecting data for clustering...", flush=True)
        selected_data = [dataloader.dataset[pool_indices[idx]][0] for idx in sorted_indices]
        
        actual_selected = []    
        current_cost = 0
        while len(actual_selected) < num_to_select:
            selected_pos = random.choices(range(len(pool_indices)), weights=probabilities, k=1)[0]
            selected_idx = pool_indices[selected_pos]
            if selected_idx in actual_selected:
                continue
            current_cost += costs[selected_pos]
            actual_selected.append(selected_idx)
        print(f"Final cost_utility: {current_cost}", flush=True)
        print(f"Number of selected points: {len(actual_selected)}", flush=True)

        
        print("Selection completed", flush=True)
        print("Selected indices:", [idx for idx in actual_selected], flush=True)
        # print("Corresponding normalized uncertainty-cost ratios:", [uncertainty_cost_ratios[pool_indices.index(idx)] for idx in actual_selected])  
        cost_utility = sum(costs[pool_indices.index(idx)] for idx in actual_selected)
        print(f"cost_utility: {cost_utility}", flush=True)
    else:
        print(f"Iteration {i}", flush=True)
        actual_selected = []
        current_cost = 0
        while current_cost < cost_utility:
            selected_pos = random.choices(range(len(pool_indices)), weights=probabilities, k=1)[0]
            selected_idx = pool_indices[selected_pos]
            if selected_idx in actual_selected:
                continue
            current_cost += costs[selected_pos]
            actual_selected.append(selected_idx)
        print(f"Final cost_utility: {current_cost}", flush=True)
        print(f"Number of selected points: {len(actual_selected)}", flush=True)
        cost_utility = current_cost
        
    print(f"Final cost_utility: {cost_utility}", flush=True)
    print(f"Number of selected points: {len(actual_selected)}", flush=True)
    return num_to_select, actual_selected, cost_utility



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='default')  # experiment name
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--dataset', type=str, default='sizeless')
    parser.add_argument('--num_cls', type=int, default=1)
    parser.add_argument('--num_shots', type=int, default=2)  # number of shots in few-shot learning
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--sizeless', action='store_true')  # to reuse Sizeless architecture (default is false)
    parser.add_argument('--rnn', action='store_true')  # to replace the SNAIL architecture with an RNN (default is false)
    parser.add_argument('--bert', action='store_true')  # to replace the SNAIL architecture with a BERT (default is false)
    parser.add_argument('--bertend2end', action='store_true')  # to replace the SNAIL architecture with an end-to-end BERT (default is false)
    parser.add_argument('--skip_embedding', action='store_true')  # no embedding (fall back to fully-connected NN, default is false)
    parser.add_argument('--cuda', action='store_true')  # to run with CUDA (default is false)
    parser.add_argument('--alpha', type=float, default=1)
    options = parser.parse_args()
    options.exp = 'experiments/exp-' + options.exp
    if not os.path.exists(options.exp):
        os.makedirs(options.exp)

    device = torch.device("cuda:0" if options.cuda else "cpu")
    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # Train Flash Model with alpha=0
    options.alpha = 1
    print(f'Initializing the {options.dataset} dataset with alpha={options.alpha}...')
    tr_dataloader, val_dataloader, test_dataloader, _ = init_dataset(options)
    print('Dataset initialized!')

    model = init_model(options, device)
    optim = torch.optim.Adam(params=model.parameters(), lr=options.lr, weight_decay=options.decay)

    # Train on the training dataset
    print("Training on training dataset...")
    best_state, best_mape, train_loss_per_epoch, train_mape_per_epoch = original_train(
        opt=options,
        tr_dataloader=tr_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        optim=optim
    )

   
    if best_state is None:
        raise ValueError("best_state is None. Ensure the model training process correctly tracks the best model state.")

    # Split the test dataloader into test_training and test_test (70:30 split)
    test_training_data, test_test_data = train_test_split(test_dataloader.dataset, test_size=0.3)
    test_training_dataloader = torch.utils.data.DataLoader(test_training_data, batch_size=options.batch_size, shuffle=True)
    test_test_dataloader = torch.utils.data.DataLoader(test_test_data, batch_size=options.batch_size, shuffle=False)
    
    
    
    initial_indices = np.random.choice(len(test_training_dataloader.dataset), size=int(0.1 * len(test_training_dataloader.dataset)), replace=False)
    active_indices = list(initial_indices)
    print(len(active_indices))
    pool_indices = list(set(range(len(test_training_dataloader.dataset))) - set(active_indices))
    
    n_sim = int(len(active_indices) * 0.1)  # 10% of active indices
    print("n_sim:", n_sim)
    simulation_indices = random.sample(active_indices, n_sim)
    remaining_indices = list(set(active_indices) - set(simulation_indices))
   
    # initial_cost = sum(calculate_total_cost(df.iloc[idx]) for idx in initial_indices)
    # initial_costs = [test_training_dataloader.dataset[idx][1] for idx in initial_indices]
    # Loop through initial indices and print the index and its corresponding cost
    # initial_costs = []
    # for idx in initial_indices:
    #     fake_cost = test_training_dataloader.dataset[idx][0]  # [1] accesses the cost label
    #     initial_costs.append(fake_cost[9]+fake_cost[20]+fake_cost[31])

    #     print(f"Index: {idx}, Cost_9: {fake_cost[9]},Cost_20: {fake_cost[20]},Cost_31: {fake_cost[31]}")
    # print("Initial cost:", sum(initial_costs))
    
    
    total_costs = []
    for idx in range(len(test_training_dataloader.dataset)):
        fake_cost = test_training_dataloader.dataset[idx][0]  # Access the cost component
        total_cost = fake_cost[-1]  # Get the last column value
        total_costs.append(total_cost)

        # Print cost for debugging  
        print(f"Index: {idx}, Total Cost: {total_cost}")

    # Print the total cost for the dataset
    print("Total cost for all samples:", sum(total_costs))

# # Loop through the entire dataset using indices from the DataLoader
#     for idx in range(len(test_training_dataloader.dataset)):
#         fake_cost = test_training_dataloader.dataset[idx][0]  # Access the cost component
#         cost_9 = fake_cost[9]
#         cost_20 = fake_cost[20]
#         cost_31 = fake_cost[31]
#         total_cost = cost_9 + cost_20 + cost_31
        
#         total_costs.append(total_cost)

#         # Print individual costs for debugging
#         print(f"Total Cost:Index: {idx}, Cost_9: {cost_9}, Cost_20: {cost_20}, Cost_31: {cost_31}, Total Cost: {total_cost}")

#     # Print the total cost for the dataset
#     print("Total cost for all samples:", sum(total_costs))

    
    cost_only = 0
    start_time = time.time()
    cost_utility = 0

    print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    for i in range(1000):
        print(f"Training iteration {i+1}")
        with open(os.path.join(options.exp, f'active_indices{i}.txt'), 'w') as f:
            for index in active_indices:
                f.write(f"{index}\n")
               
               
        subset_sampler = SubsetRandomSampler(active_indices)
        subset_dataloader = DataLoader(test_training_dataloader.dataset, batch_size=options.batch_size, sampler=subset_sampler)

        if i > 0:
            model.load_state_dict(torch.load(os.path.join(options.exp, f'last_model_{i-1}.pth')))

        res = train(simulation_indices,i, options, subset_dataloader, model,test_training_dataloader,pool_indices, optim,test_test_dataloader, val_dataloader)
        check,best_state, best_mape, train_loss, train_mape, val_loss, val_mape, train_loss_per_epoch, train_mape_per_epoch = res
        if len(pool_indices) <= 0 :
            break
        
        if i < 1000:
            AL_start_time = time.time()
            print(f"AL started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(AL_start_time))}")
            # print(f"Before AL: active indices: {len(active_indices)}, pool indices: {len(pool_indices)}")

            number,new_indices,cost_utility = active_learning_iteration_no_clustering(cost_utility,i,model, test_training_dataloader, pool_indices, options.batch_size, options)
            print(f"After AL: number to select: {number}, new indices: {len(new_indices)}")
            active_indices.extend(new_indices)
            # print(f"After extending: active indices: {len(active_indices)}")
            pool_indices = list(set(pool_indices) - set(new_indices))
            print(f"After updating pool: pool indices: {len(pool_indices)}")
            n_sim = int(len(active_indices) * 0.1)  # 10% of active indices
            simulation_indices = random.sample(active_indices, n_sim)
            remaining_indices = list(set(active_indices) - set(simulation_indices))
            
            
            AL_end_time = time.time()
            print(f"AL Training ended at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(AL_end_time))}")
            check,best_state, best_mape, train_loss, train_mape, val_loss, val_mape, train_loss_per_epoch, train_mape_per_epoch = res
           
           
            Test_start_time = time.time()
            print(f"test started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(Test_start_time))}")
            print('Testing with last model..')
            test(opt=options, test_dataloader=test_test_dataloader, model=model)


            model.load_state_dict(best_state)
            print('Testing with best model..')
            test(opt=options, test_dataloader=test_test_dataloader, model=model)
            Test_end_time = time.time()
            print(f"test ended at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(Test_end_time))}")
   
    end_time = time.time()
    print(f"Training ended at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
   

    # train_loss, train_mape, val_loss, val_mape, train_loss_per_epoch and train_mape_per_epoch can be used for visualization of the training process
    check,best_state, best_mape, train_loss, train_mape, val_loss, val_mape, train_loss_per_epoch, train_mape_per_epoch = res
    print('Testing with last model..')
    test(opt=options,
         test_dataloader=test_test_dataloader,
         model=model)

    model.load_state_dict(best_state)
    print('Testing with best model..')
    test(opt=options,
         test_dataloader=test_test_dataloader,
         model=model)
    
    
    

    # # Train the model on test_training dataset and evaluate on test_test dataset every 5 epochs
    # print("Training the model on test_training dataset for 50 epochs...")
    # model, test_accuracies = train_on_test_dataloader(opt=options,
    #                                                   test_training_dataloader=test_training_dataloader,
    #                                                   test_test_dataloader=test_test_dataloader,
    #                                                   model=model,
    #                                                   optim=optim)


if __name__ == '__main__':
    main()
