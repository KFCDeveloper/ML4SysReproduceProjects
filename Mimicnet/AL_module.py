import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset, Subset
import random
import time
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
from train_lstm_al_tcp import *
from itertools import chain
from scipy.stats import t


# cost_per_ms = {
#     'y_128': 0.0000000021,
#     'y_256': 3.7750000000000004e-09,
#     'y_512': 0.0000000083,
#     'y_1024': 0.0000000167,
#     'y_2048': 0.0000000333,
#     'y_3008': 4.9116666666666666e-08,
# }
df = pd.read_csv('/mydata/MimicNet/cost_dataset/out_time_in_sys.txt', header=None, names=['time_cost'])


# def calculate_total_cost(row, cost_per_ms):
#     total_cost = 0
#     for col, cost in cost_per_ms.items():
#         if col in row:
#             duration_ms = row[col]  # Duration is already in milliseconds
#             total_cost += duration_ms * cost
#     return total_cost

# def calculatel_cost(df, pool_indices, cost_per_ms):
#     cost = sum([calculate_total_cost(df.iloc[idx], cost_per_ms) for idx in pool_indices])
#     return cost

def get_uncertainty(true_l,prediction_l,true_d,prediction_d):
    variances = (0.5*(true_l - prediction_l)+0.5*(true_d-prediction_d)) ** 2
    return variances.cpu().tolist()

def get_time_cost(iter_training_cost,epoch_time_end,epoch_time_start):
    epoch_duration = epoch_time_end - epoch_time_start 
    iter_training_cost += epoch_duration
    return iter_training_cost

def uncertainty_selection(model, X_input, y_d_input, y_l_input, enable_validation, index, batch_size, window_size):
    uncertainties = []
    index = 0
    with torch.no_grad():
        while window_size + index <= X_input.shape[0]:
            cur_step = int(index / batch_size)
            data = train_both_data_generator(X_input, y_d_input, y_l_input, window_size, index, batch_size)
            if data is None:
                break
            X_sample, y_d_sample, y_l_sample = data
            X_sample = torch.from_numpy(X_sample).to(device, dtype=torch.double)
            y_d_sample = torch.from_numpy(y_d_sample).to(device, dtype=torch.double)
            y_l_sample = torch.from_numpy(y_l_sample).to(device, dtype=torch.double)

            # if enable_validation:
            #     val_size = int(X_sample.shape[0] * 0.1)
            #     X_test = X_sample[-val_size:, :, :]
            #     Y_d_test = y_d_sample[-val_size:, :]
            #     Y_l_test = y_l_sample[-val_size:, :]

            #     X_sample = X_sample[:-val_size, :, :]
            #     y_d_sample = y_d_sample[:-val_size, :]
            #     y_l_sample = y_l_sample[:-val_size, :]

            # Initialize hidden state with the correct batch size
            model.init_hidden(X_sample.shape[0])

            pred_drop, pred_latency = model(X_sample)
            pred_latency = pred_latency.view(-1, 1)
            target_latency = y_l_sample.view(-1, 1)
            
            pred_drop = pred_drop.view(-1, 1)
            target_drop = y_d_sample.view(-1, 1)
            
            # variances = (0.5*(pred_latency - target_latency)+0.5*(pred_drop - target_drop)) ** 2
            pred_latency_detached = pred_latency.detach()
            pred_drop_detached = pred_drop.detach()

            confidence_level = 0.95
            t_value = t.ppf((1 + confidence_level) / 2, len(pred_latency) - 1)
            
            # interval_latency = t_value  * torch.sqrt(
            #     1 + (1 / len(pred_latency_detached)) + ((pred_latency_detached - pred_latency_detached.mean())**2 / torch.sum((pred_latency_detached - pred_latency_detached.mean())**2))
            # )
            # interval_drop = t_value  * torch.sqrt(
            #     1 + (1 / len(pred_drop_detached)) + ((pred_drop_detached - pred_drop_detached.mean())**2 / torch.sum((pred_drop_detached - pred_drop_detached.mean())**2))
            # )
            intervals_latency = []
            for i in range(pred_latency_detached.shape[0]):  # Loop through each sample in the batch
                # Prediction interval for latency
                interval_latency = t_value * torch.sqrt(
                    1 + (1 / len(pred_latency_detached)) + 
                    ((pred_latency_detached[i] - pred_latency_detached.mean(dim=0))**2).sum() / ((pred_latency_detached - pred_latency_detached.mean(dim=0))**2).sum()
                )
                intervals_latency.append(interval_latency.cpu().numpy())
           
            uncertainties.append(intervals_latency)
            index += batch_size
    return uncertainties
##different for different model
# def uncertainty(model, dataloader, device, opt):
def uncertainty(model, X_input, y_d_input, y_l_input, enable_validation, index, batch_size, window_size):
    uncertainties = []
    index = 0
    with torch.no_grad():
        while window_size + index <= X_input.shape[0]:
            cur_step = int(index / batch_size)
            data = train_both_data_generator(X_input, y_d_input, y_l_input, window_size, index, batch_size)
            if data is None:
                break
            X_sample, y_d_sample, y_l_sample = data
            X_sample = torch.from_numpy(X_sample).to(device, dtype=torch.double)
            y_d_sample = torch.from_numpy(y_d_sample).to(device, dtype=torch.double)
            y_l_sample = torch.from_numpy(y_l_sample).to(device, dtype=torch.double)

            # if enable_validation:
            #     val_size = int(X_sample.shape[0] * 0.1)
            #     X_test = X_sample[-val_size:, :, :]
            #     Y_d_test = y_d_sample[-val_size:, :]
            #     Y_l_test = y_l_sample[-val_size:, :]

            #     X_sample = X_sample[:-val_size, :, :]
            #     y_d_sample = y_d_sample[:-val_size, :]
            #     y_l_sample = y_l_sample[:-val_size, :]

            # Initialize hidden state with the correct batch size
            model.init_hidden(X_sample.shape[0])

            pred_drop, pred_latency = model(X_sample)
            pred_latency = pred_latency.view(-1, 1)
            target_latency = y_l_sample.view(-1, 1)
            
            pred_drop = pred_drop.view(-1, 1)
            target_drop = y_d_sample.view(-1, 1)
            
            variances = (0.5*(pred_latency - target_latency)+0.5*(pred_drop - target_drop)) ** 2
            uncertainties.append(variances.cpu().numpy())
            index += batch_size
    return uncertainties


def AL_intrain(sampled_uncertainties, budget, sampled_costs):
    # sampled_uncertainties_flat = list(chain.from_iterable(sampled_uncertainties))
    sampled_uncertainties_flat = [u.item() for u in chain.from_iterable(sampled_uncertainties)]
    print(f"length uncertaintie {len(sampled_uncertainties_flat)}")
    print(f"length cost {len(sampled_costs)}")
    selected_uncertainties = []
    total_cost = 0

    # Calculate utility (uncertainty/cost) for each sample
    utilities = [u / c if c != 0 else 0 for u, c in zip(sampled_uncertainties_flat, sampled_costs)]
    print(f"utilities {len(utilities)}")
    # Sort utilities and their corresponding uncertainties and costs by utility value
    sorted_indices = np.argsort(utilities)[::-1]  # Sort in descending order of utility
    sorted_utilities = np.array(utilities)[sorted_indices]
    sorted_uncertainties = np.array(sampled_uncertainties_flat)[sorted_indices]
    sorted_costs = np.array(sampled_costs)[sorted_indices]
    
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
    print('number of data select in sample {}'.format(i))
    # Return the sum of the selected uncertainties
    return sum(selected_uncertainties),total_cost,i

# def uncertainties_simulation(tr_dataloader,dataset_size,sample_size,iter_training_cost,pool_indices,model,opt,device):
def uncertainties_simulation(X_select, y_d_select, y_l_select,dataset_size,sample_size,iter_training_cost,pool_indices,model,enable_validation,index,batch_size,window_size,lengthcurrent,random_indices_simulation):
    # sample_indice = random.sample(pool_indices, sample_size)
    # subset = Subset(tr_dataloader, sample_indice)
    # sample_dataloader = DataLoader(subset, batch_size=tr_dataloader.batch_size, shuffle=False)
    X_sample = X_select[random_indices_simulation]
    y_d_sample = y_d_select[random_indices_simulation]
    y_l_sample = y_l_select[random_indices_simulation]
    
    sample_scale = dataset_size / sample_size
    budget = iter_training_cost * sample_size / lengthcurrent
    
    ### modified based on model
    sampled_costs = []
    for l in random_indices_simulation:
        row = df.iloc[l]
        # cost = calculate_total_cost(row)
        # sampled_costs.append(cost)
        sampled_costs.append(row['time_cost'] * 122 * 0.1) 
        
    sampled_uncertainties = []
  
    sampled_uncertainties = uncertainty(model,X_sample,y_d_sample,y_l_sample,enable_validation,index,batch_size,window_size)
    data_uncertainty,data_cost,new_labeled = AL_intrain(sampled_uncertainties,budget,sampled_costs)
    return data_uncertainty,data_cost,new_labeled,sample_scale
    
def calculate_and_compare_metrics(uncertainties, last_uncertainty, current_uncertainty, dataset_size, new_labeled, iter_training_cost, data_uncertainty, data_cost, sample_scale, prev_smoothed_delta_cost_ratio, prev_smoothed_diff_sum_cost_ratio, smooth_delta_cost_ratios, smooth_diff_sum_cost_ratios, window_size, epoch):
    """
    Calculate various metrics and perform moving average comparison.

    Parameters:
    - uncertainties (list): List of uncertainty values for the epoch.
    - last_uncertainty (float): Uncertainty from the previous iteration.
    - current_uncertainty (float): Current uncertainty for this iteration.
    - dataset_size (int): Size of the dataset.
    - new_labeled (int): Number of newly labeled samples.
    - iter_training_cost (float): The training cost for the iteration.
    - data_uncertainty (float): Data uncertainty for simulation.
    - data_cost (float): Data cost for simulation.
    - sample_scale (float): Scale for sampling.
    - prev_smoothed_delta_cost_ratio (float or None): Previous smoothed delta/cost ratio.
    - prev_smoothed_diff_sum_cost_ratio (float or None): Previous smoothed diff/sum cost ratio.
    - smooth_delta_cost_ratios (list): List to store delta/cost ratio for smoothing.
    - smooth_diff_sum_cost_ratios (list): List to store diff/sum cost ratio for smoothing.
    - window_size (int): Window size for moving average.
    - epoch (int): Current epoch.

    Returns:
    - bool: Decision based on the comparison between smoothed metrics.
    - float: Updated smoothed delta/cost ratio.
    - float: Updated smoothed diff/sum cost ratio.
    """

    mean_uncertainty = np.mean(uncertainties)
    delta = last_uncertainty - current_uncertainty
    # diff = delta + (data_uncertainty - mean_uncertainty) * sample_scale
    diff = delta + (data_uncertainty * sample_scale) 
    size_increase = (dataset_size + new_labeled) / dataset_size
    sum_cost = iter_training_cost * size_increase + data_cost * sample_scale
    delta_cost_ratio = delta / iter_training_cost
    diff_sum_cost_ratio = diff / sum_cost
    
    print(f"sample_scale: {sample_scale}")
    print(f"data_cost: {data_cost}")
    print(f"Mean uncertainty: {mean_uncertainty}")
    print(f"last_uncertainty: {last_uncertainty}")
    print(f"current_uncertainty: {current_uncertainty}")
    print(f"Delta (last_uncertainty - current_uncertainty): {delta}")
    print(f"Diff: {diff}")
    print(f"Size increase: {size_increase}")
    print(f"Sum cost: {sum_cost}")
    print(f"Delta cost ratio: {delta_cost_ratio}")
    print(f"Diff sum cost ratio: {diff_sum_cost_ratio}")
    

    if epoch != 0:
        if prev_smoothed_delta_cost_ratio is None:
            smoothed_delta_cost_ratio = delta_cost_ratio
        else:
            smoothed_delta_cost_ratio = 0.9 * prev_smoothed_delta_cost_ratio + (1 - 0.9) * delta_cost_ratio
        
        if prev_smoothed_diff_sum_cost_ratio is None:
            smoothed_diff_sum_cost_ratio = diff_sum_cost_ratio
        else:
            smoothed_diff_sum_cost_ratio = 0.9 * prev_smoothed_diff_sum_cost_ratio + (1 - 0.9) * diff_sum_cost_ratio

        def calculate_mean(values):
            return sum(values) / len(values)

        
        smooth_delta_cost_ratios.append(smoothed_delta_cost_ratio)
        smooth_diff_sum_cost_ratios.append(smoothed_diff_sum_cost_ratio)
        
        if len(smooth_delta_cost_ratios) >= window_size:
            # Calculate moving averages
            # avg_delta_cost_ratio = np.mean(smooth_delta_cost_ratios[-window_size:])
            # avg_diff_sum_cost_ratio = np.mean(smooth_diff_sum_cost_ratios[-window_size:])
            avg_delta_cost_ratio = calculate_mean(smooth_delta_cost_ratios[-window_size:])
            avg_diff_sum_cost_ratio = calculate_mean(smooth_diff_sum_cost_ratios[-window_size:])
            print("delta_cost_ratios[-window_size:]:", smooth_delta_cost_ratios[-window_size:])
            print("diff_sum_cost_ratios[-window_size:]:", smooth_diff_sum_cost_ratios[-window_size:])
            print(f'Moving Avg (delta/cost): {avg_delta_cost_ratio}, Moving Avg (diff/sum_cost): {avg_diff_sum_cost_ratio}')

            # Check condition using moving averages
            if avg_delta_cost_ratio < avg_diff_sum_cost_ratio:
                return False, smoothed_delta_cost_ratio, smoothed_diff_sum_cost_ratio
            else:
                return True, smoothed_delta_cost_ratio, smoothed_diff_sum_cost_ratio
        else:
            print(f'Collecting data for moving average. Current window: {len(smooth_delta_cost_ratios)}/{window_size}')
            return True, smoothed_delta_cost_ratio, smoothed_diff_sum_cost_ratio
    else:
        return True, None, None
    
    
def calculate_and_compare_metrics_sample_window(uncertainties, last_uncertainty, current_uncertainty, dataset_size, new_labeled, iter_training_cost, data_uncertainty, data_cost, sample_scale, window_diff, window_delta, window_size, epoch):
    """
    Calculate various metrics and perform moving average comparison.

    Parameters:
    - uncertainties (list): List of uncertainty values for the epoch.
    - last_uncertainty (float): Uncertainty from the previous iteration.
    - current_uncertainty (float): Current uncertainty for this iteration.
    - dataset_size (int): Size of the dataset.
    - new_labeled (int): Number of newly labeled samples.
    - iter_training_cost (float): The training cost for the iteration.
    - data_uncertainty (float): Data uncertainty for simulation.
    - data_cost (float): Data cost for simulation.
    - sample_scale (float): Scale for sampling.
    - prev_smoothed_delta_cost_ratio (float or None): Previous smoothed delta/cost ratio.
    - prev_smoothed_diff_sum_cost_ratio (float or None): Previous smoothed diff/sum cost ratio.
    - smooth_delta_cost_ratios (list): List to store delta/cost ratio for smoothing.
    - smooth_diff_sum_cost_ratios (list): List to store diff/sum cost ratio for smoothing.
    - window_size (int): Window size for moving average.
    - epoch (int): Current epoch.

    Returns:
    - bool: Decision based on the comparison between smoothed metrics.
    - float: Updated smoothed delta/cost ratio.
    - float: Updated smoothed diff/sum cost ratio.
    """
    
    
    if window_diff is None:
        window_diff = []
    if window_delta is None:
        window_delta = []
    
    mean_uncertainty = np.mean(uncertainties)
    delta = last_uncertainty - current_uncertainty
    diff = delta + (data_uncertainty * sample_scale - mean_uncertainty)
    size_increase = (dataset_size + new_labeled) / dataset_size
    sum_cost = iter_training_cost * size_increase + data_cost * sample_scale
    delta_cost_ratio = delta / iter_training_cost
    diff_sum_cost_ratio = diff / sum_cost
    
    print(f"sample_scale: {sample_scale}")
    print(f"data_cost: {data_cost}")
    print(f"Mean uncertainty: {mean_uncertainty}")
    print(f"last_uncertainty: {last_uncertainty}")
    print(f"current_uncertainty: {current_uncertainty}")
    print(f"Delta (last_uncertainty - current_uncertainty): {delta}")
    print(f"Diff: {diff}")
    print(f"Size increase: {size_increase}")
    print(f"Sum cost: {sum_cost}")
    print(f"Delta cost ratio: {delta_cost_ratio}")
    print(f"Diff sum cost ratio: {diff_sum_cost_ratio}")
    def calculate_mean(values):
            return sum(values) / len(values)
    if epoch != 0:        
        window_diff.append(diff_sum_cost_ratio)
        window_delta.append(delta_cost_ratio)
        
        if len(window_delta) >= window_size:
            # Calculate moving averages
            # avg_delta_cost_ratio = np.mean(smooth_delta_cost_ratios[-window_size:])
            # avg_diff_sum_cost_ratio = np.mean(smooth_diff_sum_cost_ratios[-window_size:])
            avg_delta_cost_ratio = calculate_mean(window_delta[-window_size:])
            avg_diff_sum_cost_ratio = calculate_mean(window_diff[-window_size:])
            print("delta_cost_ratios[-window_size:]:", window_delta[-window_size:])
            print("diff_sum_cost_ratios[-window_size:]:", window_diff[-window_size:])
            print(f'Moving Avg (delta/cost): {avg_delta_cost_ratio}, Moving Avg (diff/sum_cost): {avg_diff_sum_cost_ratio}')

            # Check condition using moving averages
            if avg_delta_cost_ratio < avg_diff_sum_cost_ratio:
                return False, window_delta, window_diff
            else:
                return True, window_delta, window_diff
        else:
            print(f'Collecting data for moving average. Current window: {len(window_delta)}/{window_size}')
            return True, window_delta, window_diff
    else:
        return True, None, None


def AL_Select(X,initialize,i):
        initial_indices = np.random.choice(len(X), size=int(0.1 * len(X)), replace=False)
        active_indices = list(initial_indices)
        print(len(active_indices))
        pool_indices = list(set(range(len(X))) - set(active_indices))
        
        with open(os.path.join(f'active_indices{i}.txt'), 'w') as f:
            for index in active_indices:
                f.write(f"{index}\n")
    
        return active_indices,pool_indices,initialize

def active_learning_iteration(cost_utility, i, model, X, y_d, y_l, pool_indices,chunk_indices,pool_trunk_indices):
    print(f"Starting active learning iteration {i}", flush=True)
    # pool_loader = DataLoader(Subset(dataloader.dataset, pool_indices), batch_size=batch_size, shuffle=False)
    
    trunk_uncertainty_cost_sums = []
    chunk_rep_with_indices = []
    chunk_cost = []
    for trunk_idx in pool_trunk_indices:
        X_pool_chunk, y_d_pool_chunk, y_l_pool_chunk = [], [], []
        start_idx, end_idx = chunk_indices[trunk_idx]
        X_pool_chunk .append(X[start_idx:end_idx])
        y_d_pool_chunk .append(y_d[start_idx:end_idx])
        y_l_pool_chunk .append(y_l[start_idx:end_idx])
    
    # X_pool = X[pool_indices]
    # y_d_pool = y_d[pool_indices]
    # y_l_pool = y_l[pool_indices]
        print(f"calculate the uncertainty for trunk: {trunk_idx}", flush=True)
        window_size = 12
        index = 0
        batch_size =128
        enable_validation = False
        
        
        X_pool_chunk = np.concatenate(X_pool_chunk, axis=0)  # or use torch.cat if dealing with tensors
        y_d_pool_chunk = np.concatenate(y_d_pool_chunk, axis=0)
        y_l_pool_chunk = np.concatenate(y_l_pool_chunk, axis=0)
        
        uncertainty_ten = uncertainty_selection(model, X_pool_chunk,y_d_pool_chunk,y_l_pool_chunk,enable_validation,index,batch_size,window_size)
        

        # for tensor in uncertainty_ten:
        #     # Convert the tensor to a NumPy array and flatten it
        #     tensor_np = tensor.flatten()  # Assuming `tensor` is already a NumPy array
        #     uncertainty_values_np.append(tensor_np)

        # # Concatenate all flattened arrays into a single array
        # uncertainty_values = np.concatenate(uncertainty_values_np)
        uncertainty_values = []

        # Flatten each batch of intervals and extend the single list
        for tensor in uncertainty_ten:
            # Extend the list with flattened values from each batch
            uncertainty_values.extend(tensor)
        # print(f"uncertainty_values: {len(uncertainty_values)}", flush=True)
        
        # print("Calculating costs...", flush=True)
        # costs = []
        # for idx in pool_indices:
        #     try:
        #         cost = df.iloc[idx]['time_cost']
        #         costs.append(cost)
        #     except IndexError:
        #         print(f"Warning: Index {idx} is out of bounds. Skipping...")
        # valid_indices = [idx for idx in pool_indices if idx < len(df)]

        # # Retrieve the 'time_cost' column in one go, avoiding iloc in each loop
        # costs = df.loc[valid_indices, 'time_cost'].tolist()
        valid_indices = [idx for idx in range(start_idx, end_idx) if idx < len(df)]
        costs = [cost * 122*0.1 for cost in df.loc[valid_indices, 'time_cost']]
        
        # print(f"pool_indices: {len(pool_indices)}", flush=True)
        # print(f"Costs: {len(costs)}", flush=True)
        # print(f"uncertainty_values: {len(uncertainty_values)}", flush=True)
        # print("Sample of Uncertainty values:", flush=True)
        # print("First 10:", uncertainty_values[:10], flush=True)
        if len(uncertainty_values) != len(costs):
            costs  = costs[:len(uncertainty_values)]
            print(f"costs: {len(costs)}", flush=True)
            print(f"uncertainty_values: {len(uncertainty_values)}", flush=True)
            print("Mismatch between number of uncertainty values and costs")
        
        # print(f"Costs: {len(costs)}", flush=True)
        # print(f"uncertainty_values: {len(uncertainty_values)}", flush=True)
        # print("Calculating uncertainty-cost ratios...", flush=True)
        uncertainty_cost_ratios = [u / c for u, c in zip(uncertainty_values, costs)]
        trunk_uncertainty_cost_sums.append((trunk_idx, sum(uncertainty_cost_ratios)))
        chunk_rep_with_indices.append((trunk_idx,np.mean(X_pool_chunk, axis=0)))
        chunk_cost.append((trunk_idx,sum(costs)))
  
    num_to_select = int(0.1 * len(chunk_indices))
    print(f"Number to select: {num_to_select}", flush=True)
    number_cluster = int(np.sqrt(2*num_to_select))
    # number_cluster = 30
    print(f"number_cluster: {number_cluster}", flush=True)
    print(f"trunk_uncertainty_cost_sums: {trunk_uncertainty_cost_sums}", flush=True)
    print(f"chunk_cost: {chunk_cost}", flush=True)
    if i == 0:
        print("First iteration (i == 0)", flush=True)
        print("Sorting indices...", flush=True)
        # sorted_indices = np.argsort(uncertainty_cost_ratios)[::-1][:2*num_to_select]
        print("Selecting data for clustering...", flush=True)
        # selected_data = [dataloader.dataset[pool_indices[idx]][0] for idx in sorted_indices]
        # selected_data = [X_pool[idx] for idx in sorted_indices]
        
        # print(f"selected_data size: {len(selected_data)}, shape: {np.array(selected_data).shape}", flush=True)
        # if np.any(np.isnan(selected_data)) or np.any(np.isinf(selected_data)): print("Data contains NaN or Inf values!", flush=True)

        trunk_indices, chunk_representations = zip(*chunk_rep_with_indices)
        print("Starting K-means clustering...", flush=True)
        # kmeans = KMeans(n_clusters=number_cluster, max_iter=300, random_state=0).fit(selected_data)
        
        kmeans = MiniBatchKMeans(n_clusters=number_cluster, 
                         batch_size=16,  # Adjust this based on your memory capacity
                         max_iter=200,     # Maximum number of iterations
                         random_state=0,   # For reproducibility
                         n_init=10,    # Number of times algorithm will run with different centroids
                         init='k-means++') # Initialization method

        # Fit the model to your data
        kmeans.fit(chunk_representations)
        
        print("K-means clustering completed", flush=True)
        print(f"Cluster labels: {kmeans.labels_}", flush=True)
        print("Calculating distances...", flush=True)
        distances = cdist(chunk_representations, kmeans.cluster_centers_)
        print("Sorting cluster indices...", flush=True)
        # sorted_cluster_indices = []
        # for cluster_id in range(number_cluster):
        #     cluster_points = np.where(kmeans.labels_ == cluster_id)[0]
        #     # print(f"Cluster {cluster_id}: {len(cluster_points)} points", flush=True)
        #     if len(cluster_points) > 0:
        #         sorted_cluster_points = sorted(cluster_points, key=lambda x: distances[x, cluster_id])
        #         sorted_cluster_indices.append([sorted_indices[idx] for idx in sorted_cluster_points])
        sorted_cluster_indices = []
        for cluster_id in range(number_cluster):
            # Get the indices of the points belonging to the current cluster
            cluster_points = np.where(kmeans.labels_ == cluster_id)[0]
            
            if len(cluster_points) > 0:
                # Sort the chunks in each cluster by their original high-to-low utility order
                sorted_cluster_points = sorted(cluster_points, key=lambda x: trunk_uncertainty_cost_sums[x][1], reverse=True)
                sorted_cluster_indices.append(sorted_cluster_points)
        print(f"sorted_cluster_indices: {sorted_cluster_indices}", flush=True)
        print("Selecting data points...", flush=True)
        actual_selected_chunks = []
        iteration_count = 0
        while True:
            for cluster in sorted_cluster_indices:
                if len(actual_selected_chunks) >= num_to_select:
                    break
                if len(cluster) > 0:
                    idx = cluster.pop(0)
                    actual_selected_chunks.append(chunk_cost[idx][0])
                    cost_utility += chunk_cost[idx][1]  # Add the cost utility of the selected trunk
                    print(f"iteration: {iteration_count}, idx: {idx}, cost: {chunk_cost[idx][1]}, cost_utility: {cost_utility},selected_idx:{chunk_cost[idx][0]}", flush=True)
    
                iteration_count += 1
            if len(actual_selected_chunks) >= num_to_select:
                break
        
        print("Selection completed", flush=True)
        print(len(actual_selected_chunks))
        # print("Selected indices:", [idx for idx in actual_selected], flush=True)
        # print("Corresponding normalized uncertainty-cost ratios:", [uncertainty_cost_ratios[pool_indices.index(idx)] for idx in actual_selected])  
        print(f"cost_utility: {cost_utility}", flush=True)
        
        # cost_utility = sum(costs[pool_indices.index(idx)] for idx in actual_selected)
    else:
        print(f"Iteration {i}", flush=True)
        print("Sorting indices...", flush=True)
        # sorted_indices = np.argsort(uncertainty_cost_ratios)[::-1][:2*num_to_select]
        trunk_indices, chunk_representations = zip(*chunk_rep_with_indices)
        print("Starting K-means clustering...", flush=True)
        # kmeans = KMeans(n_clusters=number_cluster, max_iter=300, random_state=0).fit(selected_data)
        
        kmeans = MiniBatchKMeans(n_clusters=number_cluster, 
                         batch_size=16,  # Adjust this based on your memory capacity
                         max_iter=200,     # Maximum number of iterations
                         random_state=0,   # For reproducibility
                         n_init=10,    # Number of times algorithm will run with different centroids
                         init='k-means++') # Initialization method

        # Fit the model to your data
        kmeans.fit(chunk_representations)
        
        print("K-means clustering completed", flush=True)
        print(f"Cluster labels: {kmeans.labels_}", flush=True)
        print("Calculating distances...", flush=True)
        distances = cdist(chunk_representations, kmeans.cluster_centers_)
        print("Sorting cluster indices...", flush=True)
        # sorted_cluster_indices = []
        # for cluster_id in range(number_cluster):
        #     cluster_points = np.where(kmeans.labels_ == cluster_id)[0]
        #     # print(f"Cluster {cluster_id}: {len(cluster_points)} points", flush=True)
        #     if len(cluster_points) > 0:
        #         sorted_cluster_points = sorted(cluster_points, key=lambda x: distances[x, cluster_id])
        #         sorted_cluster_indices.append([sorted_indices[idx] for idx in sorted_cluster_points])
        sorted_cluster_indices = []
        for cluster_id in range(number_cluster):
            # Get the indices of the points belonging to the current cluster
            cluster_points = np.where(kmeans.labels_ == cluster_id)[0]
            
            if len(cluster_points) > 0:
                # Sort the chunks in each cluster by their original high-to-low utility order
                sorted_cluster_points = sorted(cluster_points, key=lambda x: trunk_uncertainty_cost_sums[x][1], reverse=True)
                sorted_cluster_indices.append(sorted_cluster_points)
        print(f"sorted_cluster_indices: {sorted_cluster_indices}", flush=True)
        print("Selecting data points...", flush=True)
        cumulative_utility = 0
        actual_selected_chunks = []
        iteration = 0
        while True:
            all_clusters_empty = True
            for cluster in sorted_cluster_indices:
                if len(cluster) > 0:
                    all_clusters_empty = False
                    idx = cluster.pop(0)
                    if cumulative_utility + chunk_cost[idx][1] > cost_utility:
                        continue
                    cumulative_utility += chunk_cost[idx][1]
                    iteration += 1
                    print(f"iteration: {iteration}, idx: {idx}, cost: {chunk_cost[idx][1]}, cumulative_utility: {cumulative_utility}", flush=True)
                    actual_selected_chunks.append(chunk_cost[idx][0])
            if all_clusters_empty:  # Break out of the loop if all clusters are empty
                break
        # while True:
        #     for cluster in sorted_cluster_indices:
        #         while cluster:  # As long as the cluster is not empty
        #             idx = cluster.pop(0)  # Get the first element in the cluster
        #             if cumulative_utility + chunk_cost[idx][1] > cost_utility:
        #                 break  # Break out of the inner loop if the cost exceeds utility
        #             cumulative_utility += chunk_cost[idx][1]  # Add cost to cumulative utility
        #             iteration += 1
        #             print(f"iteration: {iteration}, idx: {idx}, cost: {chunk_cost[idx][1]}, cumulative_utility: {cumulative_utility}", flush=True)
        #             actual_selected_chunks.append(chunk_cost[idx][0])  # Add chunk to the selected list
        #     else:
        #         # Continue outer loop only if no break occurred in the inner loop (i.e., all clusters processed without exceeding cost_utility)
        #         continue
        #     # Exit the loop if cost utility is exceeded
        #     break
        print("Selection completed", flush=True)
    
    print(f"Final cost_utility: {cost_utility}", flush=True)
    print(f"Number of selected points: {len(actual_selected_chunks)}", flush=True)
    return num_to_select, actual_selected_chunks, cost_utility



# def train(opt, tr_dataloader, model, optim, device, val_dataloader=None):
    
#     i = 0
#     batch_size = 16
#     ##AL start initialize dataset
#     initialize = True
#     active_indices,pool_indices,initialize = AL_Select(tr_dataloader,initialize,i)
#     start_time = time.time()
#     cost_utility = 0
    
#     subset_sampler = SubsetRandomSampler(active_indices)
#     subset_dataloader = DataLoader(tr_dataloader.dataset, batch_size=tr_dataloader.batch_size, sampler=subset_sampler)
#     #smooth
#     prev_smoothed_delta_cost_ratio = None
#     prev_smoothed_diff_sum_cost_ratio = None
#     smooth_delta_cost_ratios = []  # List to store (delta / iter_training_cost) values
#     smooth_diff_sum_cost_ratios = []
#     window_size = 1
#     #initialzie value
#     iter_training_cost = 0
#     current_uncertainty = 0
#     last_uncertainty = 100
#     #simulation choice
#     sample_size = 100  # Fixed sample size of 100
#     dataset_size = len(tr_dataloader.dataset)
#     apply_check = False
#     simulation_epoch = 0
    
    
#     ##original train
#     if val_dataloader is None:
#         best_state = None
#     train_loss = []
#     train_mape = []
#     val_loss = []
#     val_mape = []
#     best_mape = 100

#     train_loss_per_epoch = []
#     train_mape_per_epoch = []

#     best_model_path = os.path.join(opt.exp, 'best_model.pth')
#     last_model_path = os.path.join(opt.exp, 'last_model.pth')

#     loss_fn = nn.MSELoss()

#     for epoch in tqdm(range(opt.epochs)):
#         simulation_epoch+=1
#         print('=== Epoch: {} ==='.format(epoch))
        
#         #initialize everything if apply AL
#         if apply_check == True:
#             apply_check = False
#             simulation_epoch = 0
#             train_loss = []
#             train_mape = []
#             val_loss = []
#             val_mape = []
#             best_mape = 100
#             train_loss_per_epoch = []
#             train_mape_per_epoch = []
#             prev_smoothed_delta_cost_ratio = None
#             prev_smoothed_diff_sum_cost_ratio = None
#             smooth_delta_cost_ratios = []  # List to store (delta / iter_training_cost) values
#             smooth_diff_sum_cost_ratios = []
#             window_size = 1
#             #initialzie value
#             iter_training_cost = 0
#             current_uncertainty = 0
#             last_uncertainty = 100
            
        
#         #time to calculation
#         epoch_time_start = time.time()
#         #save each epoch uncertainties
#         uncertainties = []
        
#         tr_iter = iter(subset_dataloader)
#         model.train()
#         model = model.to(device)
#         for batch in tr_iter:
#             optim.zero_grad()
#             x, y = batch
#             x, y, last_targets = batch_for_few_shot(opt, x, y, device)
#             model_output = model(x, y)
#             last_model = model_output[:, -1, :].squeeze(0)
            
#             ##get uncertainties
#             true_values = y.view(-1)
#             predictions = last_model.view(-1)
#             uncertainties.extend(get_uncertainty(true_values,predictions))
            
#             last_model = last_model.view((1, -1))
#             last_targets = last_targets.view((1, -1))
#             loss = loss_fn(last_model, last_targets)
#             loss.backward()
#             optim.step()
#             train_loss.append(loss.item())
#             train_mape.append(get_mape(last_model, last_targets))
            
#         avg_loss = np.mean(train_loss[-opt.iterations:])
#         avg_mape = np.mean(train_mape[-opt.iterations:])
#         print('Avg Train Loss: {}, Avg Train MAPE: {}'.format(avg_loss, avg_mape))
#         train_loss_per_epoch.append(avg_loss)
#         train_mape_per_epoch.append(avg_mape)

#         ##uncertainty and cost calculation
#         current_uncertainty = sum(uncertainties)
#         epoch_time_end = time.time()  
#         iter_training_cost = get_time_cost(iter_training_cost,epoch_time_end,epoch_time_start)
#         print('iter_training_cost {}'.format(iter_training_cost))
        
        
        
#         if val_dataloader is None:
#             continue
#         Val_start_time = time.time()
#         print(f"Validation started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(Val_start_time))}")
#         val_iter = iter(val_dataloader)
#         model.eval()
#         for batch in val_iter:
#             x, y = batch
#             x, y, last_targets = batch_for_few_shot(opt, x, y, device)
#             model_output = model(x, y)
#             last_model = model_output[:, -1, :].squeeze(0)
#             last_model = last_model.view((1, -1))
#             last_targets = last_targets.view((1, -1))
#             loss = loss_fn(last_model, last_targets)
#             val_loss.append(loss.item())
#             val_mape.append(get_mape(last_model, last_targets))
#         avg_loss = np.mean(val_loss[-opt.iterations:])
#         avg_mape = np.mean(val_mape[-opt.iterations:])
#         postfix = ' (Best)' if avg_mape <= best_mape else ' (Best: {})'.format(best_mape)
#         print('Avg Val Loss: {}, Avg Val MAPE: {}{}'.format(avg_loss, avg_mape, postfix))
#         if avg_mape <= best_mape:
#             torch.save(model.state_dict(), best_model_path)
#             best_mape = avg_mape
#             best_state = model.state_dict()
        
        
#         data_uncertainty,data_cost,new_labeled,sample_scale = uncertainties_simulation(tr_dataloader,dataset_size,sample_size,iter_training_cost,pool_indices,model,opt,device)
#         ##metrics calculation
#         result, prev_smoothed_delta_cost_ratio, prev_smoothed_diff_sum_cost_ratio = calculate_and_compare_metrics(
#             uncertainties,
#             last_uncertainty,
#             current_uncertainty,
#             dataset_size,
#             new_labeled,
#             iter_training_cost,
#             data_uncertainty,
#             data_cost,
#             sample_scale,
#             prev_smoothed_delta_cost_ratio,
#             prev_smoothed_diff_sum_cost_ratio,
#             smooth_delta_cost_ratios,
#             smooth_diff_sum_cost_ratios,
#             window_size,
#             simulation_epoch
#         )
#         if result == True:
#             i+=1
#             apply_check = True
#             if i>=6:
#                 break
#             number,new_indices,cost_utility = active_learning_iteration(cost_utility,i,model, tr_dataloader, pool_indices, batch_size, device, opt)
#             active_indices.extend(new_indices)
#             pool_indices = list(set(pool_indices) - set(new_indices))
#             subset_sampler = SubsetRandomSampler(active_indices)
#             subset_dataloader = DataLoader(tr_dataloader.dataset, batch_size=batch_size, sampler=subset_sampler)
#         else:
#             last_uncertainty = current_uncertainty
        
        
#         for name in ['train_loss', 'train_mape', 'val_loss', 'val_mape', 'train_loss_per_epoch', 'train_mape_per_epoch']:
#             save_list_to_file(os.path.join(opt.exp, name + '.txt'), locals()[name])

#     torch.save(model.state_dict(), last_model_path)

#     return best_state, best_mape, train_loss, train_mape, val_loss, val_mape, train_loss_per_epoch, train_mape_per_epoch