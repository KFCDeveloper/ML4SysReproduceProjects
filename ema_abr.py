
#TODO: FOR EACH TOD: WE NNT TO CHECK IT ONCE AGAIN!!
import os
import sys
import numpy as np
import torch
import pickle

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pprint import pprint
from munch import Munch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import cfg
from baseline_special.utils.utils import load_traces
from baseline_special.utils.constants import BITRATE_LEVELS
from plm_special.trainer import Trainer
from plm_special.evaluate import evaluate_on_env
from plm_special.test import test_on_env
from plm_special.data.dataset import ExperienceDataset
from plm_special.models.rl_policy import OfflineRLPolicy
from plm_special.models.state_encoder import EncoderNetwork
from plm_special.models.low_rank import peft_model
from plm_special.utils.utils import set_random_seed
from plm_special.utils.plm_utils import load_plm
from plm_special.utils.console_logger import ConsoleLogger

from AL_module import *
from plm_special.trainer import all

import json

output_json_path = '/projects/bcrn/yliang7/research/NetLLM/adaptive_bitrate_streaming/data/total_costs.json'
def load_costs(file_path):
    with open(file_path, 'r') as json_file:
        costs = json.load(json_file)
    return costs

costs = load_costs(output_json_path)


PLM_LAYER_SIZES = {
    'gpt2': {
        'base': 24,
        'small': 12,
        'large': 36,
        'xl': 48
    },
    'llama': {
        'base': 32,
    },
    't5-lm': { 
        'base': 12,
        'small': 6,
        'large': 24,
        'xl': 24
    }
}


def save_model(args, model, save_dir):
    if args.rank > 0:
        # save lora weights
        model.plm.save_pretrained(save_dir)
        # save other modules except plm
        torch.save(model.modules_except_plm.state_dict(), os.path.join(save_dir, 'modules_except_plm.bin'))
    else:
        # lora is disabled, save whole model
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.bin'))


def load_model(args, model, model_dir):
    if args.rank > 0:
        # load lora weights
        model.plm.load_adapter(model_dir, adapter_name='default')
        # load other modules except plm
        model.modules_except_plm.load_state_dict(torch.load(os.path.join(model_dir, 'modules_except_plm.bin')))
    else:
        # lora is disabled, load whole model
        model.load_state_dict(torch.load(os.path.join(model_dir, 'model.bin')))
    return model


def adapt(args, model, exp_dataset, exp_dataset_info, eval_env_settings, checkpoint_dir, best_model_dir, eval_process_reward_fn, env_settings, results_dir):

    #! This can be regarded as test_train dl
    training_dl = DataLoader(exp_dataset, args.bs, shuffle=True, pin_memory=True)


    #! We start AL here:

    ##initialize 10% dataset as initialize dataset
    i = 0
    ##AL start initialize dataset
    initialize = True

    #! I am not sure whether exp_dataset is the correct dataset to use??
    #TODO: Make sure which dataset and Dataloader we should use here
    active_indices,pool_indices,simulation_indices,remaining_indices,initialize,selection_cost = AL_Select(training_dl,initialize,i)
    start_time = time.time()
    cost_utility = 0

    print(f"selection_cost: {selection_cost}")
    subset_sampler = SubsetRandomSampler(remaining_indices)
    subset_dataloader = DataLoader(training_dl.dataset, batch_size=args.batch_size, sampler=subset_sampler)
    #smooth
    prev_smoothed_delta_cost_ratio = None
    prev_smoothed_diff_sum_cost_ratio = None
    smooth_delta_cost_ratios = []  # List to store (delta / iter_training_cost) values
    smooth_diff_sum_cost_ratios = []
    window_size = 1
    #initialzie value
    iter_training_cost = 0
    current_uncertainty = 0
    last_uncertainty = 100
    #simulation choice
    sample_size = len(simulation_indices)  # Fixed sample size of 100
    dataset_size = len(exp_dataset.dataset)
    apply_check = False
    simulation_epoch = 0
    
    number = 0

    #! Enable resume, so we can skip the training dataset built and the traning process
    if args.resume:
        rl_policy = load_model(args, rl_policy, args.resume_path)
        print('Resume weights for training from:', args.resume_path)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / args.warmup_steps, 1)
    )
    loss_fn = CrossEntropyLoss()

    trainer = Trainer(args, model=model, optimizer=optimizer, exp_dataset=exp_dataset, loss_fn=loss_fn, device=args.device, lr_scheduler=lr_scheduler, 
                      grad_accum_steps=args.grad_accum_steps, dataloader=training_dl)

    target_return = exp_dataset_info.max_return * args.target_return_scale
    best_eval_return = 0.

    total_train_losses = []

    print('Start training on the fine-tuning dataset...')


    additional_epochs = 1000

    for epoch in range(additional_epochs):

        if apply_check == True:
            apply_check = False
            simulation_epoch = 0
            train_loss = []
            train_mape = []
            val_loss = []
            val_mape = []
            best_mape = 100
            train_loss_per_epoch = []
            train_mape_per_epoch = []
            prev_smoothed_delta_cost_ratio = None
            prev_smoothed_diff_sum_cost_ratio = None
            smooth_delta_cost_ratios = []  # List to store (delta / iter_training_cost) values
            smooth_diff_sum_cost_ratios = []
            window_size = 1
            #initialzie value
            iter_training_cost = 0
            current_uncertainty = 0
            last_uncertainty = 100
        #time to calculation
        epoch_time_start = time.time()
        #save each epoch uncertainties
        uncertainties = []

        #! Update the trainer
        trainer.dataloader = subset_dataloader
        train_logs, train_losses = trainer.train_epoch()

        #! From AL
        uncertainties.extend(train_losses)

        total_train_losses.extend(train_losses)
        print('='* 20, 'Training Iteration #{epoch}', '=' * 20)
        print('>' * 10, 'Training Information:')
        pprint(train_logs)

        if epoch % args.save_checkpoint_per_epoch == 0:  # save checkpoint
            checkpoint_dir_epoch = os.path.join(checkpoint_dir, "_additional_", str(epoch))
            if not os.path.exists(checkpoint_dir_epoch):
                os.makedirs(checkpoint_dir_epoch)
            save_model(args, model, checkpoint_dir_epoch)
            print('Checkpoint saved at:', checkpoint_dir_epoch)

        #! We do not use the default evaluation here
        # if epoch % args.eval_per_epoch == 0:
        #     eval_logs = evaluate_on_env(args, env_settings=eval_env_settings, model=model, target_return=target_return, max_ep_num=args.trace_num,
        #                                 process_reward_fn=eval_process_reward_fn)
        #     episodes_return = eval_logs['episodes_return']
        #     if best_eval_return < episodes_return:
        #         best_eval_return = episodes_return
        #         save_model(args, model, best_model_dir)
        #         print('Best model saved at:', best_model_dir)

        #     eval_logs['best_return'] = best_eval_return
        #     print('>' * 10, 'Evaluation Information')
        #     pprint(eval_logs)

        current_uncertainty = sum(uncertainties)
        print("uncertainties shape:", len(uncertainties))
        epoch_time_end = time.time()  
        iter_training_cost = get_time_cost(iter_training_cost,epoch_time_end,epoch_time_start)
        print('iter_training_cost {}'.format(iter_training_cost))

        ##simulation of active learning
        #TOD:Fix this function uncertainties_simulation()
        data_uncertainty = uncertainties_simulation(args, training_dl,dataset_size,sample_size,iter_training_cost,simulation_indices,model,len(active_indices), len(pool_indices), trainer)

        ##metrics to stop al
        #! In this part, the para to pass in are all related to AL itself
        result, prev_smoothed_delta_cost_ratio, prev_smoothed_diff_sum_cost_ratio = calculate_and_compare_metrics(
            uncertainties,
            last_uncertainty,
            current_uncertainty,
            dataset_size,
            iter_training_cost,
            data_uncertainty,
            prev_smoothed_delta_cost_ratio,
            prev_smoothed_diff_sum_cost_ratio,
            smooth_delta_cost_ratios,
            smooth_diff_sum_cost_ratios,
            window_size,
            simulation_epoch,
            len(active_indices),
            len(pool_indices),
            selection_cost
        )
        simulation_epoch+=1
        if result == False:
            apply_check = True
            if len(pool_indices) <= 0:
                break
        
            #TOD: active_learning_iteration is what we should modify!
            number,new_indices,cost_utility = active_learning_iteration(cost_utility,i,model, training_dl, pool_indices, args.bs)
            i+=1
            active_indices.extend(new_indices)
            pool_indices = list(set(pool_indices) - set(new_indices))
            selection_cost = cost_utility
            simulation_size = min(max(50,int(0.1 * len(active_indices))),1000)
            simulation_indices = random.sample(new_indices, simulation_size)
            remaining_indices = list(set(active_indices) - set(simulation_indices))
            subset_sampler = SubsetRandomSampler(remaining_indices)
            subset_dataloader = DataLoader(exp_dataset.dataset, batch_size=args.bs, sampler=subset_sampler)
        else:
            last_uncertainty = current_uncertainty

        # Every 2 epochs, test the model on test_test
        if (epoch + 1) % 1 == 0:
            test_model_on_test_test(args, model, exp_dataset_info, env_settings, results_dir, eval_process_reward_fn, global_epoch = epoch)



    # save training losses
    train_losses_path = os.path.join(checkpoint_dir, 'train_losses.txt')
    np.savetxt(train_losses_path, total_train_losses, fmt='%.6f', delimiter='\n')


def test_model_on_test_test(args, model, exp_dataset_info, env_settings, results_dir, test_process_reward_fn, global_epoch = 0):

    #! add global epoch to the results_dir
    results_dir = os.path.join(results_dir, f'global_epoch_{global_epoch}')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    #! The model is loaded so we do not need to load it from the model_dir

    # model_dir = args.model_dir if args.model_dir is not None else ValueError('Please specify the model weight dir for testing.')

    # assert os.path.exists(model_dir), f'Model weight dir {model_dir} does not exist.'

    # model = load_model(args, model, model_dir)
    # print('Load model from:', model_dir)
    target_return = exp_dataset_info.max_return * args.target_return_scale
    results = test_on_env(args, model, results_dir, env_settings, target_return, args.trace_num, test_process_reward_fn, seed=args.seed)
    print(results)
    print('Test time:', results['time'], '\nMean reward:', results['mean_reward'])
    print('Results saved at:', results_dir)



def uncertainty_select(dataloader):

    def calculate_intervals(data, confidence_level=0.95):
        t_value = t.ppf((1 + confidence_level) / 2, len(data) - 1)
        intervals = []
        for i in range(data.shape[0]):  # Loop through each sample in the batch
            interval = t_value * torch.sqrt(
                1 + (1 / len(data)) + 
                ((data[i] - data.mean(dim=0))**2).sum() / ((data - data.mean(dim=0))**2).sum()
            )
            intervals.append(interval.cpu().numpy())
        return intervals

    uncertainties = []
    with tqdm(dataloader) as tepoch:
        for batch in tepoch:
            batch_states, batch_actions, batch_returns, batch_timesteps = batch

            # Flatten the batch for processing
            states = torch.cat(batch_states, dim=0)
            actions = torch.cat(batch_actions, dim=0)
            returns = torch.cat(batch_returns, dim=0)
            timesteps = torch.cat(batch_timesteps, dim=0)

            # Calculate intervals for each packed value
            state_intervals = calculate_intervals(states)
            action_intervals = calculate_intervals(actions)
            return_intervals = calculate_intervals(returns)
            timestep_intervals = calculate_intervals(timesteps)

            # Calculate the mean of the intervals
            mean_intervals = [
                (state_interval + action_interval + return_interval + timestep_interval) / 4
                for state_interval, action_interval, return_interval, timestep_interval in zip(
                    state_intervals, action_intervals, return_intervals, timestep_intervals
                )
            ]

            uncertainties.extend(mean_intervals)
    
    return uncertainties


# TOD: We need to change this function to fit our AL
#! Unfinished
def active_learning_iteration(cost_utility, i, model, dataloader, pool_indices, batch_size):
    print(f"Starting active learning iteration {i}", flush=True)
    
    pool_loader = DataLoader(Subset(dataloader.dataset, pool_indices), batch_size=batch_size, shuffle=False)
    
    print("Calculating uncertainty values...", flush=True)
    uncertainty_values = uncertainty_select(pool_loader)
   
    print("Calculating costs...", flush=True)
    # Create a dictionary mapping each pool index to its corresponding cost
    costs = {l : costs[l] for l in pool_indices}
    
    print(f"pool_indices: {len(pool_indices)}", flush=True)
    print(f"Costs: {len(costs)}", flush=True)
    print(f"uncertainty_values: {len(uncertainty_values)}", flush=True)
    print("Sample of Uncertainty values:", flush=True)
    print("First 10:", uncertainty_values[:10], flush=True)
    
    assert len(uncertainty_values) == len(costs), "Mismatch between number of uncertainty values and costs"
    
    print("Calculating uncertainty-cost ratios...", flush=True)
    # Compute uncertainty-cost ratios using the dictionary
    uncertainty_cost_ratios = [u / costs[l] for u, l in zip(uncertainty_values, pool_indices)]
    
    num_to_select = int(0.1 * len(dataloader.dataset))
    uncertainty_weights = [1/u for u in uncertainty_cost_ratios]
    
    selected_indices = []
    current_cost = 0
    
    if i == 0:
        print("First iteration (i == 0)", flush=True)
        print("Selecting indices based on number to select...", flush=True)
        while len(selected_indices) < num_to_select:
            selected_data = random.choices(pool_indices, weights=uncertainty_weights, k=1)[0]
            if selected_data not in selected_indices:  # Avoid duplicates
                selected_indices.append(selected_data)
                selected_cost = costs[selected_data]
                current_cost += selected_cost
                # print(f"selected_data: {selected_data}, selected_cost: {selected_cost}, cost_utility: {current_cost}", flush=True)
    else:
        print(f"Iteration {i}", flush=True)
        print("Selecting indices based on cost utility...", flush=True)
        while current_cost < cost_utility:
            selected_data = random.choices(pool_indices, weights=uncertainty_weights, k=1)[0]
            if selected_data not in selected_indices:  # Avoid duplicates
                selected_indices.append(selected_data)
                selected_cost = costs[selected_data]
                current_cost += selected_cost
                # print(f"selected_data: {selected_data}, selected_cost: {selected_cost}, cost_utility: {current_cost}", flush=True)
    
    cost_utility = current_cost
    print(f"Final cost_utility: {cost_utility}", flush=True)
    print(f"Number of selected points: {len(selected_indices)}", flush=True)
    
    return num_to_select, selected_indices, cost_utility


def uncertainties_simulation(args, tr_dataloader,dataset_size,sample_size,iter_training_cost,simulation_indices,model,AL_select,AL_leftover, trainer):
    top_k = AL_select / AL_leftover

    
    sample_dataloader = DataLoader(
        tr_dataloader.dataset,  # Use the dataset attribute
        batch_size=args.bs,
        sampler=SubsetRandomSampler(simulation_indices)  # Use a sampler instead of Subset
    )
    
    sample_scale = dataset_size / sample_size
    budget = iter_training_cost / sample_scale
    
    ### modified based on model
    #TOD: Finish the cost calculation here
    sampled_costs = []
    for l in simulation_indices:
        sampled_costs.append(costs[l])
        
    sampled_uncertainties = []  
  
    sampled_uncertainties = uncertainty(model,sample_dataloader, trainer)
    data_uncertainty = AL_intrain(sampled_uncertainties,budget,sampled_costs,top_k)
    return data_uncertainty



def uncertainty(model, dataloader, trainer):

    def process_batch(batch, device='cpu'):
        """
        Process batch of data.
        """
        states, actions, returns, timesteps = batch

        states = torch.cat(states, dim=0).unsqueeze(0).float().to(device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=device).reshape(1, -1)
        labels = actions.long()
        actions = ((actions + 1) / BITRATE_LEVELS).unsqueeze(2)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=device).reshape(1, -1, 1)
        timesteps = torch.as_tensor(timesteps, dtype=torch.int32, device=device).unsqueeze(0)

        return states, actions, returns, timesteps, labels

    
    def train_step(batch, model):
        states, actions, returns, timesteps, labels = process_batch(batch, device=args.device)
        #! This is the forward func()
        actions_pred = model(states, actions, returns, timesteps)
        actions_pred = actions_pred.permute(0, 2, 1)
        loss = CrossEntropyLoss(actions_pred, labels)
        return loss

    trainer.dataloader = dataloader
    model.eval()
    uncertainties = []
    with torch.no_grad():
        dataset_size = len(dataloader)
        for step, batch in enumerate(dataloader):
            train_loss = train_step(batch, model)
            uncertainties.append(train_loss.item())
    
    return uncertainties

def run(args):
    assert args.plm_type in cfg.plm_types
    assert args.plm_size in cfg.plm_sizes
    assert args.exp_pool_path is not None, 'please specify a experience pool path for training'
    assert args.trace in cfg.trace_dirs.keys()
    assert args.video in cfg.video_size_dirs.keys()

    # 1. set seed
    set_random_seed(args.seed)

    # 2. create environment setting
    trace_dir = cfg.trace_dirs[args.trace]
    video_size_dir = cfg.video_size_dirs[args.video]

    #! Load the TCA-Transformed data
    if args.use_tca:
        #! This seems to be not useful since we will use the exp pool for training as dataloader
        tca_train_dir = cfg.trace_dirs['target-train-scale-0.7-transformed']
        tca_test_dir = cfg.trace_dirs['target-test-scale-0.7-transformed']

        all_cooked_time_train_tca, all_cooked_bw_train_tca, all_file_names_train_tca, all_mahimahi_ptrs_train_tca = load_traces(tca_train_dir)

        all_cooked_time_test_tca, all_cooked_bw_test_tca, all_file_names_test_tca, all_mahimahi_ptrs_test_tca = load_traces(tca_test_dir)

        #! This is the env settings for test_train
        env_settings_tca_train = {
            'all_cooked_time': all_cooked_time_train_tca,
            'all_cooked_bw': all_cooked_bw_train_tca,
            'all_file_names': all_file_names_train_tca,
            'all_mahimahi_ptrs': all_mahimahi_ptrs_train_tca,
            'video_size_dir': video_size_dir,
            'fixed': args.fixed_order,
            'trace_num': len(all_file_names_train_tca),
        }

        #! This is the env settings for test_test
        env_settings_tca_test = {
            'all_cooked_time': all_cooked_time_test_tca,
            'all_cooked_bw': all_cooked_bw_test_tca,
            'all_file_names': all_file_names_test_tca,
            'all_mahimahi_ptrs': all_mahimahi_ptrs_test_tca,
            'video_size_dir': video_size_dir,
            'fixed': args.fixed_order,
            'trace_num': len(all_file_names_test_tca),
        }


    # 3. create training dataset, fetch info

    #! Fetch the experience pool created previously
    exp_pool = pickle.load(open(args.exp_pool_path, 'rb'))
    exp_dataset = ExperienceDataset(exp_pool, gamma=args.gamma, scale=args.scale, max_length=args.w, sample_step=args.sample_step)
    exp_dataset_info = Munch(exp_dataset.exp_dataset_info)
    print('Experience dataset info:')
    pprint(exp_dataset_info)
    
    # 4. create model
    
    # 4.1 load plm
    # args.device_out and args.device_mid are used for model parallelism (currently only support llama) 
    # For data/modules near the input side, we use args.device.
    # For data/modules near the output side, we use args.device_out.
    # For data/modules lying in the middle, we use args.device_mid (it can be None). 
    # If args.device == args.device_out == args.device_mid (if not None), everything will be the same as using only one device.
    plm, *_ = load_plm(args.plm_type, os.path.join(cfg.plm_dir, args.plm_type, args.plm_size), 
                       device_input_side=args.device, device_output_side=args.device_out, device_middle_side=args.device_mid)

    if args.plm_type != 'llama':
        plm = plm.to(args.device)
    
    if args.rank != -1:
        plm = peft_model(plm, args.plm_type, rank=args.rank)

    # 4.2 create state encoder
    assert args.state_feature_dim is not None, 'please specify state feature dim to create state encoder'
    state_encoder = EncoderNetwork(embed_dim=args.state_feature_dim)
    state_encoder = state_encoder.to(args.device)

    # 4.3 create rl policy
    plm_embed_size = cfg.plm_embed_sizes[args.plm_type][args.plm_size]
    max_ep_len = exp_dataset_info.max_timestep + 1
    rl_policy = OfflineRLPolicy(state_feature_dim=args.state_feature_dim, bitrate_levels=BITRATE_LEVELS, state_encoder=state_encoder, plm=plm, plm_embed_size=plm_embed_size, 
                                           max_length=args.w, max_ep_len=max_ep_len, device=args.device, device_out=args.device_out, which_layer=args.which_layer)


    # 5. handling directory and path

    # extract training experience pool information
    train_exp_pool_info = args.exp_pool_path.split('/')[-4:-1]
    train_exp_pool_info = '_'.join(train_exp_pool_info)
    models_dir = os.path.join(cfg.plm_ft_dir, f'{args.plm_type}_{args.plm_size}', train_exp_pool_info + f'_ss_{args.sample_step}', f'rank_{args.rank}_w_{args.w}_gamma_{args.gamma}_sfd_{args.state_feature_dim}'\
                              f'_lr_{args.lr}_wd_{args.weight_decay}_warm_{args.warmup_steps}_epochs_{args.num_epochs}_seed_{args.seed}_task_{args.task_name}')
    
    results_dir = os.path.join(cfg.results_dir, f'{args.trace}_{args.video}', f'trace_num_{args.trace_num}_fixed_{args.fixed_order}', f'{args.plm_type}_{args.plm_size}',
                               f'early_stop_{args.which_layer}_rank_{args.rank}_w_{args.w}_gamma_{args.gamma}_tgt_scale_{args.target_return_scale}_seed_{args.seed}')
    checkpoint_dir = os.path.join(models_dir, f'early_stop_{args.which_layer}_checkpoint')
    best_model_dir = os.path.join(models_dir, f'early_stop_{args.which_layer}_best_model')


    # 6. start training/testing
    def process_reward(reward, 
                       max_reward=exp_dataset_info.max_reward, 
                       min_reward=exp_dataset_info.min_reward, 
                       scale=args.scale):
        reward = min(max_reward, max(min_reward, reward))  # bound reward
        return (reward - min_reward) / (max_reward - min_reward) / scale
    
    torch.backends.cudnn.benchmark = True

    if args.adapt:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)
        console_log = open(os.path.join(models_dir, f'early_stop_{args.which_layer}_console.log'), 'w')
        sys.stdout = ConsoleLogger(sys.__stdout__, console_log)
        
        adapt(args, rl_policy, exp_dataset, exp_dataset_info, env_settings_tca_test, checkpoint_dir, best_model_dir, process_reward, env_settings_tca_train, results_dir)

        # test_model_on_test_test(args, rl_policy, exp_dataset_info, env_settings_tca_test, results_dir, process_reward, global_epoch = 0)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    # training dataset settings
    parser.add_argument('--exp-pool-path', help='the path storing the experience pool file for training', default='artifacts/exp_pools/exp_pool.pkl')
    parser.add_argument('--sample-step', type=int, help='the steps for sampling experiences')
    # environment settings
    parser.add_argument('--trace', help='name of traces (e.g., fcc-test)', type=str, default='fcc-test')
    parser.add_argument('--trace-num', help='number of traces. if set to -1, use all traces in the trace dir.', type=int, default=100)
    parser.add_argument('--video', help='name of video (e.g., video1)', type=str, default='video1')
    parser.add_argument('--fixed-order', action='store_true', help='iterate over test traces in a fixed sequential order.')
    # plm settings
    parser.add_argument('--plm-type', type=str, default='gpt2')
    parser.add_argument('--plm-size', type=str, default='base')
    parser.add_argument('--rank', type=int, help='rank of low-rank matrices. if set to -1, low-rank matrices will not be enabled', default=-1)
    # state encoder settings
    parser.add_argument('--state-feature-dim', type=int, help='feature dim of the state encoder', default=256)
    # rl policy related settings
    parser.add_argument('--w', type=int, help='context window for learning return distribution', default=10)
    parser.add_argument('--gamma', type=float, help='discounted factor of reward', default=1.)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-steps', type=int, default=2000)
    parser.add_argument('--num-epochs', type=int, default=80)
    parser.add_argument('--eval-per-epoch', type=int, help='evaluation per epoch', default=1)
    parser.add_argument('--save-checkpoint-per-epoch', type=int, help='saving checkpoint per iteration')
    parser.add_argument('--target-return-scale', type=float, help='target return, which specifies the expected performance for the model to achieve', default=1.)
    parser.add_argument('--which-layer', type=int, help='for early stopping (not used in our experiments): specify which layer to stop (layer index starts from 0)', default=-1)
    # other settings
    parser.add_argument('--adapt', action="store_true", help='adapt model')
    parser.add_argument('--test', action="store_true", help='test model')
    parser.add_argument('--grad-accum-steps', dest='grad_accum_steps', type=int, default=32)
    parser.add_argument('--seed', help='random seed', type=int, default=100003)
    parser.add_argument('--scale', help='scale reward/return', type=int, default=1000)
    parser.add_argument('--model-dir', help='model weight dir for testing')
    parser.add_argument('--device', action='store', dest='device', help='device (cuda or cpu) to run experiment')
    parser.add_argument('--device-out', action='store', dest='device_out', help='device (cuda or cpu) to place the split of model near the output')
    parser.add_argument('--device-mid', action='store', dest='device_mid', help='device (cuda or cpu) to place the split of model between the input and output')
    parser.add_argument('--task_name', type=str, default=None, help='task name')
    parser.add_argument('--resume', action='store_true', dest='resume', help='(Optional) Resume model weights from checkpoint for training.')
    parser.add_argument('--resume-path', action="store", dest='resume_path', help='using for resume')
    parser.add_argument('--use-tca', action='store_true', help='whether to use TCA or not.')
    
    args = parser.parse_args()


    if args.device_out is None:  
        args.device_out = args.device
    
    if args.save_checkpoint_per_epoch is None:
        args.save_checkpoint_per_epoch = args.eval_per_epoch
    assert args.save_checkpoint_per_epoch <= args.num_epochs

    print('Arguments:')
    pprint(args)

    run(args)
