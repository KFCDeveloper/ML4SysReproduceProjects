from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset, Subset
import random
import time
import os
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.stats import t



# cost_per_ms = {
#     'y_128': 0.0000000021,
#     'y_256': 3.7750000000000004e-09,
#     'y_512': 0.0000000083,
#     'y_1024': 0.0000000167,
#     'y_2048': 0.0000000333,
#     'y_3008': 4.9116666666666666e-08,
# }
df = pd.read_csv(r"/mydata/flash/data-firm/a_cost_summary.csv")


def calculate_total_cost(row, cost_per_ms):
    total_cost = 0
    for col, cost in cost_per_ms.items():
        if col in row:
            duration_ms = row[col]  # Duration is already in milliseconds
            total_cost += duration_ms * cost
    return total_cost

def calculatel_cost(df, pool_indices, cost_per_ms):
    cost = sum([calculate_total_cost(df.iloc[idx], cost_per_ms) for idx in pool_indices])
    return cost

def get_uncertainty(true,prediction):
    variances = (true - prediction) ** 2
    return variances.cpu().tolist()

def get_time_cost(iter_training_cost,epoch_time_end,epoch_time_start):
    epoch_duration = epoch_time_end - epoch_time_start 
    epoch_duration_hours = epoch_duration
    iter_training_cost += epoch_duration_hours 
    return iter_training_cost

##different for different model


# def uncertainty_select(model, dataloader, device, env_base):
#     model.eval()
#     uncertainties = []
#     all_inputs = []
#     with torch.no_grad():
#        with tqdm(dataloader) as tepoch:
#             for inputs, targets in tepoch:
#                 inputs = inputs.to(device)
#                 targets = targets.to(device)
#                 all_inputs.append(inputs)
#                 ###get uncertainty during the forward
#                 # uncertainties.extend(get_uncertainty(targets,yhat))
                
#                 # loss, loss_val, uncertainties_list = loss_fn_maxflow_maxconc_target(yhat, targets, env_base)
#                 # print("inputs shape:", inputs.shape)
#     confidence_level = 0.95
#     t_value = t.ppf((1 + confidence_level) / 2, len(all_inputs) - 1)
#     intervals = []
#     for i in range(all_inputs.shape[0]):  # Loop through each sample in the batch
#                     # Prediction interval for latency
#         interval = t_value * torch.sqrt(
#         1 + (1 / len(all_inputs)) + 
#         ((all_inputs[i] - all_inputs.mean(dim=0))**2).sum() / ((all_inputs - all_inputs.mean(dim=0))**2).sum()
#         )
#         intervals.append(interval.cpu().numpy())

#         uncertainties.extend(intervals)
    
#     return uncertainties
def uncertainty_select(model, dataloader, device, env_base):
    model.eval()
    uncertainties = []
    all_inputs = []
    with torch.no_grad():
        with tqdm(dataloader) as tepoch:
            for inputs, targets in tepoch:
                inputs = inputs.to(device)
                targets = targets.to(device)
                all_inputs.append(inputs)
                ### Uncomment and implement these lines if needed
                # uncertainties.extend(get_uncertainty(targets, yhat))
                # loss, loss_val, uncertainties_list = loss_fn_maxflow_maxconc_target(yhat, targets, env_base)

    # Combine all inputs into a single tensor
    all_inputs = torch.cat(all_inputs, dim=0)  # Concatenate along the batch dimension
    
    confidence_level = 0.95
    t_value = t.ppf((1 + confidence_level) / 2, len(all_inputs) - 1)
    intervals = []
    for i in range(all_inputs.shape[0]):  # Loop through each sample in the batch
        # Prediction interval for latency
        interval = t_value * torch.sqrt(
            1 + (1 / len(all_inputs)) +
            ((all_inputs[i] - all_inputs.mean(dim=0))**2).sum() / 
            ((all_inputs - all_inputs.mean(dim=0))**2).sum()
        )
        intervals.append(interval.cpu().numpy())

    uncertainties.extend(intervals)
    
    return uncertainties




##different for different model
def uncertainty(model, dataloader, device,env_base,paths_to_edges_base,commodities_to_paths):
    model.eval()
    uncertainties = []
    with torch.no_grad():
       with tqdm(dataloader) as tepoch:
            for inputs, targets in tepoch:
                inputs = inputs.to(device)
                targets = targets.to(device)
                yhat = model(inputs)
                ###get uncertainty during the forward
                loss, loss_val, uncertainties_list = loss_fn_maxflow_maxconc_target(yhat, targets, env_base,paths_to_edges_base,device,commodities_to_paths)
                ###get uncertainty during the forward
                uncertainties.extend(uncertainties_list)
    
    return uncertainties

def AL_intrain(sampled_uncertainties, budget, sampled_costs):
    selected_uncertainties = []
    total_cost = 0

    # Calculate utility (uncertainty/cost) for each sample
    utilities = [u / c if c != 0 else 0 for u, c in zip(sampled_uncertainties, sampled_costs)]

    # Sort utilities and their corresponding uncertainties and costs by utility value
    sorted_indices = np.argsort(utilities)[::-1]  # Sort in descending order of utility
    sorted_utilities = np.array(utilities)[sorted_indices]
    sorted_uncertainties = np.array(sampled_uncertainties)[sorted_indices]
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



def uncertainties_simulation(tr_dataloader,dataset_size,sample_size,iter_training_cost,simulation_indices,model,device,env_base,paths_to_edges_base,commodities_to_paths):
    print(f"Simulation indices: {simulation_indices}")
    subset = Subset(tr_dataloader.dataset, simulation_indices)
    sample_dataloader = DataLoader(subset, batch_size=tr_dataloader.batch_size, shuffle=False)
    print(f"Number of batches in sample_dataloader: {len(sample_dataloader)}")

    sample_scale = dataset_size / sample_size
    budget = iter_training_cost / sample_scale
    
    ### modified based on model
    sampled_costs = []
    for l in simulation_indices:
        row = df.iloc[l]

        sampled_costs.append(row['time_cost'])
        
    sampled_uncertainties = []
  
    sampled_uncertainties = uncertainty(model,sample_dataloader,device,env_base,paths_to_edges_base,commodities_to_paths)
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
    diff = delta + (data_uncertainty) * sample_scale
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
        
        smooth_delta_cost_ratios.append(smoothed_delta_cost_ratio)
        smooth_diff_sum_cost_ratios.append(smoothed_diff_sum_cost_ratio)
        
        if len(smooth_delta_cost_ratios) >= window_size:
            # Calculate moving averages
            avg_delta_cost_ratio = np.mean(smooth_delta_cost_ratios[-window_size:])
            avg_diff_sum_cost_ratio = np.mean(smooth_diff_sum_cost_ratios[-window_size:])
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

# def uncertainty_select(model, dataloader, device):
#     model.eval()
#     uncertainties = []
#     with torch.no_grad():
#        with tqdm(dataloader) as tepoch:
#             for inputs, targets in tepoch:
#                 inputs = inputs.to(device)
#                 targets = targets.to(device)
#                 yhat = model(inputs)
#                 uncertainties.extend(get_uncertainty(targets,yhat))
#     return uncertainties
def uncertainty_select_RL(prediction_value):
    uncertainties = []

    
    
    confidence_level = 0.95
    t_value = t.ppf((1 + confidence_level) / 2, len(prediction_value) - 1)
    intervals = []
    for i in range(prediction_value.shape[0]):  # Loop through each sample in the batch
        # Prediction interval for latency
        interval = t_value * torch.sqrt(
            1 + (1 / len(prediction_value)) +
            ((prediction_value[i] - prediction_value.mean(dim=0))**2).sum() / 
            ((prediction_value - prediction_value.mean(dim=0))**2).sum()
        )
        intervals.append(interval.cpu().numpy())

    uncertainties.extend(intervals)
    
    return uncertainties

def AL_Select(total_train_dataset,initialize,i):
    if initialize == True:
        initial_indices = np.random.choice(len(total_train_dataset.dataset), size=int(0.1 * len(total_train_dataset.dataset)), replace=False)
        active_indices = list(initial_indices)
        print(len(active_indices))
        pool_indices = list(set(range(len(total_train_dataset.dataset))) - set(active_indices))
        
        n_sim = int(len(active_indices) * 0.1)  # 10% of active indices
        simulation_indices = random.sample(active_indices, n_sim)
        remaining_indices = list(set(active_indices) - set(simulation_indices))
        initialize = False
        
        costs = []
        for l in active_indices:
            row = df.iloc[l]

            costs.append(row['time_cost'])
        print(f"initailize sum costs: {sum(costs)}")
        
    
    with open(os.path.join(f'active_indices{i}.txt'), 'w') as f:
            for index in active_indices:
                f.write(f"{index}\n")
    
    return active_indices,pool_indices,simulation_indices,remaining_indices,initialize

# def active_learning_iteration(cost_utility, i, model, dataloader, pool_indices, batch_size, device, env_base):
#     print(f"Starting active learning iteration {i}", flush=True)
#     pool_loader = DataLoader(Subset(dataloader.dataset, pool_indices), batch_size=batch_size, shuffle=False)
#     print("Calculating uncertainty values...", flush=True)
#     uncertainty_values = uncertainty_select(model, pool_loader, device, env_base)
   
#     print("Calculating costs...", flush=True)
#     # costs = [calculate_total_cost(df.iloc[idx], cost_per_ms) for idx in pool_indices]
    
#     costs = []
#     for l in pool_indices:
#         row = df.iloc[l]

#         costs.append(row['time_cost'])
    
#     print(f"pool_indices: {len(pool_indices)}", flush=True)
#     print(f"Costs: {len(costs)}", flush=True)
#     print(f"uncertainty_values: {len(uncertainty_values)}", flush=True)
#     print("Sample of Uncertainty values:", flush=True)
#     print("First 10:", uncertainty_values[:10], flush=True)
#     assert len(uncertainty_values) == len(costs), "Mismatch between number of uncertainty values and costs"
    
#     print("Calculating uncertainty-cost ratios...", flush=True)
#     uncertainty_cost_ratios = [u / c for u, c in zip(uncertainty_values, costs)]
    
#     num_to_select = int(0.1 * len(dataloader.dataset))
#     print(f"Number to select: {num_to_select}", flush=True)
    
#     uncertainty_weights = uncertainty_cost_ratios / sum(uncertainty_cost_ratios)
    
    
#     if i == 0:
#         print("First iteration (i == 0)", flush=True)
#         print("Sorting indices...", flush=True)
#         while len(actual_selected) < num_to_select:
        
#         print("Selection completed", flush=True)
#         print("Selected indices:", [idx for idx in actual_selected], flush=True)
#         print("Corresponding normalized uncertainty-cost ratios:", [uncertainty_cost_ratios[pool_indices.index(idx)] for idx in actual_selected])  
#         cost_utility = sum(costs[pool_indices.index(idx)] for idx in actual_selected)
#     else:
#         print(f"Iteration {i}", flush=True)
#         print("Sorting indices...", flush=True)
#         sorted_indices = np.argsort(uncertainty_cost_ratios)[::-1][:2*num_to_select]
#         print("Selecting data for clustering...", flush=True)
#         selected_data = [dataloader.dataset[pool_indices[idx]][0] for idx in sorted_indices]
#         print("Starting K-means clustering...", flush=True)
#         kmeans = KMeans(n_clusters=number_cluster, max_iter=300, random_state=0).fit(selected_data)
#         print("K-means clustering completed", flush=True)
#         print("Calculating distances...", flush=True)
#         distances = cdist(selected_data, kmeans.cluster_centers_)
        
#         print("Sorting cluster indices...", flush=True)
#         sorted_cluster_indices = []
#         for cluster_id in range(number_cluster):
#             cluster_points = np.where(kmeans.labels_ == cluster_id)[0]
#             if len(cluster_points) > 0:
#                 sorted_cluster_points = sorted(cluster_points, key=lambda x: distances[x, cluster_id])
#                 sorted_cluster_indices.append([sorted_indices[idx] for idx in sorted_cluster_points])
        
#         print("Selecting data points...", flush=True)
#         cumulative_utility = 0
#         actual_selected = []
#         iteration = 0
#         while True:
#             for cluster in sorted_cluster_indices:
#                 if len(cluster) > 0:
#                     idx = cluster.pop(0)
#                     if cumulative_utility + costs[idx] > cost_utility:
#                         break
#                     cumulative_utility += costs[idx]
#                     iteration += 1
#                     print(f"iteration: {iteration}, idx: {idx}, cost: {costs[idx]}, cumulative_utility: {cumulative_utility}", flush=True)
#                     actual_selected.append(pool_indices[idx])
#             else:
#                 continue
#             break
#         print("Selection completed", flush=True)
    
#     print(f"Final cost_utility: {cost_utility}", flush=True)
#     print(f"Number of selected points: {len(actual_selected)}", flush=True)
#     return num_to_select, actual_selected, cost_utility

# def active_learning_iteration_RL(training_table,cost_utility, i, prediction_value, unrevealed_table,env):
#     print(f"Starting active learning iteration {i}", flush=True)
#     # pool_loader = DataLoader(Subset(dataloader.dataset, pool_indices), batch_size=batch_size, shuffle=False)
#     print("Calculating uncertainty values...", flush=True)
#     uncertainty_values = uncertainty_select_RL(prediction_value)
    
#     ##get key for unrevealed table
#     unique_filenames = {env.get_function_for_key(key) for key in unrevealed_table.keys()}
#     print("Calculating costs...", flush=True)
#     # costs = [calculate_total_cost(df.iloc[idx], cost_per_ms) for idx in pool_indices]
    
#     # costs = []
#     # for l in pool_indices:
#     #     row = df.iloc[l]

#     #     costs.append(row['time_cost'])
#     costs = []
#     for filename in unique_filenames:
#         result = df.loc[df['filename'] == filename, 'total_cost']
#         if not result.empty:
#             costs.append(result.values[0])
#             print(f"costs: {costs}")
    
    
#     print(f"pool_indices: {len(pool_indices)}", flush=True)
#     print(f"Costs: {len(costs)}", flush=True)
#     print(f"uncertainty_values: {len(uncertainty_values)}", flush=True)
#     print("Sample of Uncertainty values:", flush=True)
#     print("First 10:", uncertainty_values[:10], flush=True)
#     assert len(uncertainty_values) == len(costs), "Mismatch between number of uncertainty values and costs"
    
#     print("Calculating uncertainty-cost ratios...", flush=True)
#     uncertainty_cost_ratios = [u / c for u, c in zip(uncertainty_values, costs)]
    
#     num_to_select = int(0.1 * len(training_table))
#     print(f"Number to select: {num_to_select}", flush=True)
#     number_cluster = int(np.sqrt(2*num_to_select))
#     print(f"number_cluster: {number_cluster}", flush=True)
#     if i == 0:
#         print("First iteration (i == 0)", flush=True)
#         print("Sorting indices...", flush=True)
#         sorted_indices = np.argsort(uncertainty_cost_ratios)[::-1][:2*num_to_select]
#         print("Selecting data for clustering...", flush=True)
#         # selected_data = [dataloader.dataset[pool_indices[idx]][0] for idx in sorted_indices]
#         selected_data = [unrevealed_table[key] for key in sorted_indices]
#         print(f"selected_data: {len(selected_data)}", flush=True)
#         print("Starting K-means clustering...", flush=True)
#         kmeans = KMeans(n_clusters=number_cluster, max_iter=300, random_state=0).fit(selected_data)
#         print("K-means clustering completed", flush=True)
#         print(f"Cluster labels: {kmeans.labels_}", flush=True)
#         print("Calculating distances...", flush=True)
#         distances = cdist(selected_data, kmeans.cluster_centers_)
#         print("Sorting cluster indices...", flush=True)
#         sorted_cluster_indices = []
#         for cluster_id in range(number_cluster):
#             cluster_points = np.where(kmeans.labels_ == cluster_id)[0]
#             print(f"Cluster {cluster_id}: {len(cluster_points)} points", flush=True)
#             if len(cluster_points) > 0:
#                 sorted_cluster_points = sorted(cluster_points, key=lambda x: distances[x, cluster_id])
#                 sorted_cluster_indices.append([sorted_indices[idx] for idx in sorted_cluster_points])
        
#         print("Selecting data points...", flush=True)
#         actual_selected = []
#         iteration_count = 0
#         while True:
#             for cluster in sorted_cluster_indices:
#                 if len(actual_selected) >= num_to_select:
#                     break
#                 if len(cluster) > 0:
#                     idx = cluster.pop(0)
#                     actual_selected.append(idx)
#             iteration_count += 1
#             print(f"Iteration count: {iteration_count}, selected so far: {len(actual_selected)}", flush=True)
#             if len(actual_selected) >= num_to_select:
#                 break
        
#         print("Selection completed", flush=True)
#         print("Selected indices:", actual_selected, flush=True)
#         print("Corresponding normalized uncertainty-cost ratios:", [uncertainty_cost_ratios[idx] for idx in actual_selected], flush=True)
#         cost_utility = sum(costs[idx] for idx in actual_selected)
#     else:
#         print(f"Iteration {i}", flush=True)
#         print("Sorting indices...", flush=True)
#         sorted_indices = np.argsort(uncertainty_cost_ratios)[::-1][:2*num_to_select]
#         print("Selecting data for clustering...", flush=True)
#         # selected_data = [dataloader.dataset[pool_indices[idx]][0] for idx in sorted_indices]
#         selected_data = [unrevealed_table[key] for key in sorted_indices]
#         print("Starting K-means clustering...", flush=True)
#         kmeans = KMeans(n_clusters=number_cluster, max_iter=300, random_state=0).fit(selected_data)
#         print("K-means clustering completed", flush=True)
#         print("Calculating distances...", flush=True)
#         distances = cdist(selected_data, kmeans.cluster_centers_)
        
#         print("Sorting cluster indices...", flush=True)
#         sorted_cluster_indices = []
#         for cluster_id in range(number_cluster):
#             cluster_points = np.where(kmeans.labels_ == cluster_id)[0]
#             if len(cluster_points) > 0:
#                 sorted_cluster_points = sorted(cluster_points, key=lambda x: distances[x, cluster_id])
#                 sorted_cluster_indices.append([sorted_indices[idx] for idx in sorted_cluster_points])
        
#         print("Selecting data points...", flush=True)
#         cumulative_utility = 0
#         actual_selected = []
#         iteration = 0
#         while True:
#             for cluster in sorted_cluster_indices:
#                 if len(cluster) > 0:
#                     idx = cluster.pop(0)
#                     if cumulative_utility + costs[idx] > cost_utility:
#                         break
#                     cumulative_utility += costs[idx]
#                     iteration += 1
#                     print(f"iteration: {iteration}, idx: {idx}, cost: {costs[idx]}, cumulative_utility: {cumulative_utility}", flush=True)
#                     # actual_selected.append(pool_indices[idx])
#                     actual_selected.append(list(unrevealed_table.keys())[idx])
#             else:
#                 continue
#             break
#         print("Selection completed", flush=True)
    
#     print(f"Final cost_utility: {cost_utility}", flush=True)
#     print(f"Number of selected points: {len(actual_selected)}", flush=True)
#     return num_to_select,

def active_learning_iteration_RL(training_table,cumulative_utility, i, prediction_value, unrevealed_table,env):
    print(f"Starting active learning iteration {i}", flush=True)
    # pool_loader = DataLoader(Subset(dataloader.dataset, pool_indices), batch_size=batch_size, shuffle=False)
    print("Calculating uncertainty values...", flush=True)
    uncertainty_values = uncertainty_select_RL(prediction_value)
    
    ##get key for unrevealed table
    unique_filenames = {env.get_function_for_key(key) for key in unrevealed_table.keys()}
    print("Calculating costs...", flush=True)
    # costs = [calculate_total_cost(df.iloc[idx], cost_per_ms) for idx in pool_indices]
    
    # costs = []
    # for l in pool_indices:
    #     row = df.iloc[l]

    #     costs.append(row['time_cost'])
    costs = []
    for filename in unique_filenames:
        result = df.loc[df['filename'] == filename, 'total_cost']
        if not result.empty:
            costs.append(result.values[0])
            print(f"costs: {costs}")
    
    
    print(f"pool_indices: {len(pool_indices)}", flush=True)
    print(f"Costs: {len(costs)}", flush=True)
    print(f"uncertainty_values: {len(uncertainty_values)}", flush=True)
    print("Sample of Uncertainty values:", flush=True)
    print("First 10:", uncertainty_values[:10], flush=True)
    assert len(uncertainty_values) == len(costs), "Mismatch between number of uncertainty values and costs"
    
    print("Calculating uncertainty-cost ratios...", flush=True)
    uncertainty_cost_ratios = [u / c for u, c in zip(uncertainty_values, costs)]
    
    num_to_select = int(0.1 * len(training_table))
    print(f"Number to select: {num_to_select}", flush=True)



    probabilities = [uncertainty_cost_ratios[i] / sum(uncertainty_cost_ratios) for i in range(len(uncertainty_cost_ratios))]

    if i == 0:
        print("First iteration (i == 0)", flush=True)
        print("Sorting indices...", flush=True)
        
        key_mapping = list(unrevealed_table.keys())
        
        # Select data directly without sorting
        print("Selecting data...", flush=True)
        selected_data = [unrevealed_table[key] for key in key_mapping]
        
        # Perform probabilistic selection
        actual_selected = []
        cost_utility = 0
        while len(actual_selected) < num_to_select:
            idx = np.random.choice(range(len(selected_data)), p=probabilities)
            selected_index = key_mapping[idx]
            if selected_index not in actual_selected:
                actual_selected.append(selected_index)
                cost_utility += costs[idx]
        
        
        print(f"Final cost_utility: {cost_utility}", flush=True)
        print(f"Number of selected points: {len(actual_selected)}", flush=True)
        print("Selected indices:", actual_selected, flush=True)
        print("Corresponding normalized uncertainty-cost ratios:", [uncertainty_cost_ratios[idx] for idx in actual_selected], flush=True)
    else:
        print(f"Iteration {i}", flush=True)
        print("Sorting indices...", flush=True)
        
        
        key_mapping = list(unrevealed_table.keys())
        
        # Select data directly without sorting
        print("Selecting data...", flush=True)
        selected_data = [unrevealed_table[key] for key in key_mapping]
        
        # Perform probabilistic selection
        actual_selected = []
        cost_utility = 0
        while cost_utility < cumulative_utility:
            idx = np.random.choice(range(len(selected_data)), p=probabilities)
            selected_index = key_mapping[idx]
            if selected_index not in actual_selected:
                actual_selected.append(selected_index)
                cost_utility += costs[idx]
        

        print("Selection completed", flush=True)
    
    print(f"Final cost_utility: {cost_utility}", flush=True)
    print(f"Number of selected points: {len(actual_selected)}", flush=True)
    return num_to_select, actual_selected, cost_utility


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

#     return best_state, best_mape, train_loss, train_mape, 