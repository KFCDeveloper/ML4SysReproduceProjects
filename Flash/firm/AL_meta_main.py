import argparse
import os
import random
from tqdm import tqdm
import numpy as np
import torch

from AL_serverless_env import SimEnvironment
from ppo import PPO
from AL_meta_ppo import MetaPPO
from util import META_TRAIN_ITERATIONS, FLAG_BERT_TINY, RESULTS_PATH

# How to train
# python meta_main.py --operation=train --model_dir=./model --data_path=../data-firm/writefile_writefile_imageresize_output.csv
# How to test
# python meta_main.py --operation=test --checkpoint_path=./model/writefile_writefile_imageresize/ppo-ep200.pth.tar --data_path=../data-firm/writefile_writefile_imageresize_output.csv
def main(operation, data_path, checkpoint_path, model_dir, device, verbose=False, disable_meta=False, use_bert=False):
    """
    This is the main function for RL training and inference on an individual application.
    """
    print('Data Path:', data_path)
    print('Model Path:', model_dir)
    print('Checkpoint Path:', checkpoint_path)
    print('Operation:', operation)
    print('Verbosity:', verbose, '| Disable Meta-learning?', disable_meta)
    if operation == 'train':
        if not model_dir:
            print('Please specify the model directory! e.g., --model_dir=./model')
            exit()
        train(data_path, model_dir, device, verbose=verbose, use_bert=use_bert)
    elif operation == 'test':
        if not checkpoint_path:
            print('Please specify checkpoint path! e.g., --checkpoint_path=./model/app_name/checkpoint.pth.tar')
            exit()
        test(data_path, checkpoint_path, device, adaptation=False, verbose=verbose, use_bert=use_bert)
    elif operation == 'adaptation':
        if not checkpoint_path:
            print('Please specify checkpoint path! e.g., --checkpoint_path=./model/app_name/checkpoint.pth.tar')
            exit()
        test(data_path, checkpoint_path, device, adaptation=True, verbose=verbose, use_bert=use_bert)
    else:
        print('Unknown operation!')
        exit()


# E.g., --operation=train --data_path=../data-firm --model_dir=./model --pool=./training_pool.txt
# E.g., --operation=test --data_path=../data-firm --checkpoint_path=./model --pool=./adaptation_pool.txt
def mass_main(operation, data_dir, checkpoint_path, model_dir, application_pool, device, verbose=False, use_bert=False):
    """
    This is the main function for mass RL training and inference on a pool of applications.
    """
    print('Data Dir:', data_dir)
    print('Model Dir:', model_dir)
    print('Checkpoint Path:', checkpoint_path)
    print('Operation:', operation)
    print('Verbosity:', verbose)
    if not model_dir:
        print('Please specify the model directory! e.g., --model_dir=./model')
        exit()
    # initialize the model folder path for the particular function
    folder_path = model_dir + "/pretraining-metarl"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    results_path = RESULTS_PATH  # "./results"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    if use_bert:
        mode_type = 'berttiny' if FLAG_BERT_TINY else 'bert'
    else:
        mode_type = 'rnn'

    # create and initialize the environment for rl training
    if operation == 'train':

        f = open(application_pool, 'r')
        print('Applications from', application_pool)
        lines = f.readlines()
        task_rewards = []
        for meta_train_iter in tqdm(range(META_TRAIN_ITERATIONS)):
            line = random.choice(lines)
            csv_file = line.strip()
            
            # init the RL environment with all csv_files from the application_pool
            env = SimEnvironment(data_dir + '/' + csv_file)
            function_name = env.get_function_name()
            print('Environment initialized for function', function_name)
            initial_state = env.reset(function_name)

            # initialize the agent and start training
            if meta_train_iter == 0:
                agent = MetaPPO(env, function_name, folder_path, 'meta-train', device, verbose=verbose, bert_embedding=use_bert)
            else:
                agent.env = env
                agent.function_name = function_name
                
            task_reward = agent.train(mode='meta-train')
            task_rewards.append(task_reward)
            if meta_train_iter > 0 and meta_train_iter % 100 == 0:
                agent.save_checkpoint(meta_train_iter)
        agent.save_checkpoint(META_TRAIN_ITERATIONS)

        with open(results_path + '/train_rewards_' + mode_type + '.txt', 'w') as f:
            for reward in task_rewards:
                f.write(str(reward) + '\n')
            f.write(str(np.mean(task_rewards)) + '\n')
        print('Average training reward: ', np.mean(task_rewards))
    elif operation == 'test':
        if not checkpoint_path:
            print('Please specify checkpoint path! e.g., --checkpoint_path=./model/app_name/checkpoint.pth.tar')
            exit()
        task_rewards = []
        f = open(application_pool, 'r')
        print('Applications from', application_pool)
        lines = f.readlines()
        for line in tqdm(lines):
            csv_file = line.strip()
            print('Testing for', csv_file)
            task_reward = test(data_dir + '/' + csv_file, checkpoint_path, device, adaptation=False, verbose=verbose, use_bert=use_bert)
            task_rewards.append(task_reward)
        with open(results_path + '/test_rewards_' + mode_type + '.txt', 'w') as f:
            for reward in task_rewards:
                f.write(str(reward) + '\n')
            f.write(str(np.mean(task_rewards)) + '\n')
        print('Average test reward: ', np.mean(task_rewards))
    elif operation == 'adaptation':
        if not checkpoint_path:
            print('Please specify checkpoint path! e.g., --checkpoint_path=./model/app_name/checkpoint.pth.tar')
            exit()
        task_rewards = []
        f = open(application_pool, 'r')
        print('Applications from', application_pool)
        lines = f.readlines()
        for line in tqdm(lines):
            csv_file = line.strip()
            print('Adaptation for', csv_file)
            task_reward = test(data_dir + '/' + csv_file, checkpoint_path, device, adaptation=True, verbose=verbose, use_bert=use_bert)
            task_rewards.append(task_reward)
        with open(results_path + '/adapt_rewards_' + mode_type + '.txt', 'w') as f:
            for reward in task_rewards:
                f.write(str(reward) + '\n')
            f.write(str(np.mean(task_rewards)) + '\n')
        print('Average adaptation reward: ', np.mean(task_rewards))


# training
# e.g., data_path=../data-firm/writefile_writefile_imageresize_output.csv
# e.g., model_dir=./model
def train(data_path, model_path, device, verbose=False, use_bert=False):
    # create and initialize the environment for rl training
    print("data_path", data_path)
    env = SimEnvironment(data_path)
    function_name = env.get_function_name()
    initial_state = env.reset(function_name)
    if verbose:
        print('Environment initialized for function', function_name)
        print('Initial state:', initial_state)

    # initialize the model folder path for the particular function
    folder_path = model_path + "/" + str(function_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # initialize the agent and start training
    agent = MetaPPO(env, function_name, folder_path, 'train', device, verbose=verbose, bert_embedding=use_bert)
    
    for epoch in range(1000):
        if env.unrevealed_table == {}:
            break
        agent.train()
        env.reveal_new_data()


# testing
# e.g., data_path=../data-firm/writefile_writefile_imageresize_output.csv
# e.g., checkpoint_path=./model/writefile_writefile_imageresize/ppo-ep200.pth.tar
def test(data_path, checkpoint_path, device, adaptation=True, verbose=False, use_bert=False):
    env = SimEnvironment(data_path)
    function_name = env.get_function_name()
    initial_state = env.reset(function_name)
    mode = 'adapt' if adaptation else 'test'
    if verbose:
        print('Environment initialized for function', function_name)
        print('Initial state:', initial_state)
    if adaptation:
        folder_path = os.path.dirname(checkpoint_path) + '/' + function_name + '_adaptation'
    else:
        folder_path = os.path.dirname(checkpoint_path) + '/' + function_name + '_eval'
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)

    # init an RL agent
    agent = MetaPPO(env, function_name, folder_path, mode, device, verbose=verbose, bert_embedding=use_bert)
    checkpoint_file = checkpoint_path
    agent.load_checkpoint(checkpoint_file)

    if not adaptation:
        agent.disable_update()

    task_reward = agent.train(mode=mode)
    return task_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Meta-learning PPO Agent Training in Autoscaling Tasks.')
    parser.add_argument('--operation', choices=['train', 'test', 'adaptation'], required=True, help="Mode in either train or test.")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data file or directory.')
    parser.add_argument('--pool', type=str, required=False, help="Training or testing on a pool of applications, e.g., --pool ./training_pool.txt")
    parser.add_argument('--checkpoint_path', type=str, required=False, help='Path to the checkpoint file.')
    parser.add_argument('--model_dir', type=str, required=False, help='Path to the model directory to store to.')
    parser.add_argument('-v', '--verbose', help="Increase output verbosity", action="store_true")
    parser.add_argument('-n', '--no_meta_learning', help="Disable meta-learning.", action="store_true")
    parser.add_argument('--bert', help="Use BERT for embedding.", action="store_true")
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    if args.pool:
        print('Training/testing on a pool of applications:', args.pool)
        mass_main(args.operation, args.data_path, args.checkpoint_path, args.model_dir, args.pool, device, use_bert=args.bert)
    else:
        main(args.operation, args.data_path, args.checkpoint_path, args.model_dir, device, verbose=args.verbose, disable_meta=args.no_meta_learning, use_bert=args.bert)
