import numpy as np


"""
State and Action Space
"""
NUM_STATES = 13  # TODO: To be confirmed
NUM_ACTIONS = 7
VERTICAL_SCALING_STEP = 128
HORIZONTAL_SCALING_STEP = 1

"""
Meta-learning
"""
BUFFER_UPDATE_MODE = 'best'
BUFFER_SIZE = 32

NUM_FEATURES_PER_SHOT = 13  # 11 (obs) + 1 (action) + 1 (reward)
RNN_HIDDEN_SIZE = 256  # equal to the max steps of each episode
RNN_NUM_LAYERS = 2
EMBEDDING_DIM = 32

MAX_TIMESTEPS_PER_EPISODE = 10  # equal to EPISODE_LENGTH
MAX_NUM_TIMESTEPS = 100000000

"""
Resource Scaling Constraints
"""
MIN_INSTANCES = 1
MAX_INSTANCES = 20    # determined by the cluster capacity
MIN_CPU_LIMIT = 128   # millicore
MAX_CPU_LIMIT = 2048  # millicore
MIN_MEMORY_LIMIT = 256   # MiB
MAX_MEMORY_LIMIT = 3072  # MiB

LOWER_BOUND_UTIL = 0.7
UPPER_BOUND_UTIL = 0.9

"""
Logging and Checkpointing
"""
CHECKPOINT_DIR = './checkpoints/'
LOG_DIR = './logs/'
PLOT_FIG = True
SAVE_FIG = True
SAVE_TO_FILE = True
DATA_PATH = 'data.csv'

"""
Hyperparameters
"""
TOTAL_ITERATIONS = 500
EPISODES_PER_ITERATION = 5
EPISODE_LENGTH = 10

DISCOUNT = 0.99
HIDDEN_SIZE = 64
LR = 3e-4  # 5e-3 5e-6
SGD_EPOCHS = 5
MINI_BATCH_SIZE = 5
CLIP = 0.2
ENTROPY_COEFFICIENT = 0.01  # 0.001
CRITIC_LOSS_DISCOUNT = 0.05  # 0.03

FLAG_CONTINUOUS_ACTION = False

MAX_SAME_ITERATIONS = 2 * SGD_EPOCHS
MAX_NUM_REWARDS_TO_CHECK = 10

# thresholds for retraining monitoring
REWARD_AVG_THRESHOLD = 100
REWARD_STD_THRESHOLD = 10

ILLEGAL_PENALTY = -1


# return the current states (dictionary) in the form of a vector
def convert_state_dict_to_list(state):
    # TODO: normalize each variable in the state vector
    return list(state.values())

# convert the state (dictionary) to a string
def state_to_string(state):
    state_string = 'State: \n' +\
            'Avg CPU utilization: {:.3f}\n'.format(state['cpu_util']) +\
            'Avg memory utilization: {:.3f}\n'.format(state['memory_util']) +\
            'Avg disk I/O usage: {:.3f}\n'.format(state['disk_io_usage']) +\
            'Avg file discovery rate: {:.3f}\n'.format(state['file_discovery_rate']) +\
            'Avg rate: {:.3f}\n'.format(state['rate']) +\
            'Avg processing rate: {:.3f}\n'.format(state['processing_rate']) +\
            'Avg ingestion rate: {:.3f}\n'.format(state['ingestion_rate']) +\
            'Avg latency: {:.3f}\n'.format(state['rate']) +\
            'Num of replicas: {:d}\n'.format(state['num_replicas']) +\
            'CPU limit: {:d}\n'.format(state['cpu_limit']) +\
            'Memory limit: {:d}'.format(state['memory_limit'])
    return state_string

# print (state, action, reward) for the current step
def print_step_info(step, state_dict, action_dict, reward):
    state = state_to_string(state_dict)
    action = 'Action: N/A'
    if action_dict['vertical_cpu'] > 0:
        action = 'Action: Scaling-up by ' + str(VERTICAL_SCALING_STEP) + ' CPU limit'
    elif action_dict['vertical_cpu'] < 0:
        action = 'Action: Scaling-down by ' + str(VERTICAL_SCALING_STEP) + ' CPU limit'
    elif action_dict['vertical_memory'] > 0:
        action = 'Action: Scaling-up by ' + str(VERTICAL_SCALING_STEP) + ' memory limit'
    elif action_dict['vertical_memory'] < 0:
        action = 'Action: Scaling-down by ' + str(VERTICAL_SCALING_STEP) + ' memory limit'
    elif action_dict['horizontal'] > 0:
        action = 'Action: Scaling-out by ' + str(HORIZONTAL_SCALING_STEP) + ' replicas'
    elif action_dict['horizontal'] < 0:
        action = 'Action: Scaling-in by ' + str(HORIZONTAL_SCALING_STEP) + ' replicas'
    print('Step #' + str(step))
    print(state)
    print(action, '| Reward:', reward)

# define reward functions
# CHANGEME: [Define customized SLO-driven reward function here]
# calculate the reward based on the current state (after the execution of the current action)
# + utilization [0, 1]
# + data processing rate [0, 1]
# - penalty
# v1: R = alpha * RU + (1-alpha) * DP - penalty
def convert_state_action_to_reward(state, action, last_action, last_state, app_name='my-app'):
    alpha = 0.3
    resource_util_score = state['cpu_util'] + state['memory_util'] / 2.0
    if state['ingestion_rate'] == 0:
        data_processing_rate = 1
    else:
        data_processing_rate = state['processing_rate'] / state['ingestion_rate']

    # reward function definition
    reward = alpha * resource_util_score + (1-alpha) * data_processing_rate

    # give penalty to frequent dangling actions: e.g., scale in and out
    if last_action['horizontal'] * action['horizontal'] < 0:
        reward += -ILLEGAL_PENALTY
    elif last_action['vertical_cpu'] * action['vertical_cpu'] < 0:
        reward += -ILLEGAL_PENALTY
    elif last_action['vertical_memory'] * action['vertical_memory'] < 0:
        reward += -ILLEGAL_PENALTY

    lag_increased = True if state['rate'] < last_state['rate'] else False

    # give penalty to any lag increase
    if lag_increased:
        reward += -ILLEGAL_PENALTY

    # if latency-SLO is defined, add SLO preservation ratio as reward as well
    # reward += (1 - alpha) * SLO_Latency / state['latency']

    return reward

# v2: R = RU * DP - penalty
def convert_state_action_to_reward_v2(state, action, last_action, last_state, app_name='my-app'):
    resource_util_score = state['cpu_util'] + state['memory_util'] / 2.0
    if state['ingestion_rate'] == 0:
        data_processing_rate = 1
    else:
        data_processing_rate = state['processing_rate'] / state['ingestion_rate']

    # reward function definition
    reward = resource_util_score * data_processing_rate

    # give penalty to frequent dangling actions: e.g., scale in and out
    if last_action['horizontal'] * action['horizontal'] < 0:
        reward += -ILLEGAL_PENALTY
    elif last_action['vertical_cpu'] * action['vertical_cpu'] < 0:
        reward += -ILLEGAL_PENALTY
    elif last_action['vertical_memory'] * action['vertical_memory'] < 0:
        reward += -ILLEGAL_PENALTY

    lag_increased = True if state['rate'] < last_state['rate'] else False

    # give penalty to any lag increase
    if lag_increased:
        reward += -ILLEGAL_PENALTY

    return reward

# count the number of parameters in a model
def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
