import os
import psutil
import torch
from torch import nn
import matplotlib.pyplot as plt
import heapq

from util import *
from meta_learning.rnn import RNNEmbedding

PLOT_FIG = False
SAVE_FIG = True
SAVE_TO_FILE = True

CHECKPOINT_DIR = './checkpoints/'

TOTAL_ITERATIONS = 500
EPISODES_PER_ITERATION = 5
EPISODE_LENGTH = 10

# hyperparameters
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
REWARD_AVG_THRESHOLD = 120
REWARD_STD_THRESHOLD = 10

MAX_TIMESTEPS_PER_EPISODE = 10  # equal to EPISODE_LENGTH
BUFFER_SIZE = 32
BUFFER_UPDATE_MODE = 'best'

class MetaPPO:
    def __init__(self, env, function_name, verbose=False):
        self.env = env
        self.function_name = function_name

        self.state_size = NUM_STATES
        self.action_size = NUM_ACTIONS  # NUM_TOTAL_ACTIONS

        self.env_dim = {
            'state': self.state_size,
            'action': self.action_size,
            'reward': 1
        }

        # self.actor = ActorNetwork(self.state_size, HIDDEN_SIZE, self.action_size)
        # self.critic = CriticNetwork(self.state_size, HIDDEN_SIZE, 1)
        self.actor = MetaActorNetwork(self.state_size, HIDDEN_SIZE, self.action_size, self.env_dim, self, verbose=verbose)
        self.critic = MetaCriticNetwork(self.state_size, HIDDEN_SIZE, 1, self.env_dim, self, verbose=verbose)

        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=LR)

        self.cov = torch.diag(torch.ones(self.action_size, ) * 0.5)

        self.skip_update = False

        self.num_same_parameter_actor = 0
        self.num_same_parameter_critic = 0
        self.parameter_actor = None
        self.parameter_critic = None

        # episode buffer for embedding generation
        self.episode_buffer = None

        self.config = {
            'mode': BUFFER_UPDATE_MODE,
            'buffer_size': BUFFER_SIZE
        }

        self.clear_episode_buffer()

        # store recent episode rewards
        self.recent_rewards = []
        
    def train():
        for iteration in range(TOTAL_ITERATIONS):
            print(1)


# 结合embedding一起做 allocation 策略推理
class MetaNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, env_dim, agent, verbose=True):
        super(MetaNetwork, self).__init__()

        self.env_dim = env_dim
        self.agent = agent

        # rnn embedding
        self.rnn = RNNEmbedding(1, BUFFER_SIZE, 'wa', num_channels=MAX_TIMESTEPS_PER_EPISODE, verbose=verbose)

        # 输入是 embedding_len + traffic_matrix + MLU? 还是什么忘记了
        self.fc1 = nn.Linear(input_size + self.rnn.embedding_dim, hidden_size)
        # 如果要加 transformer 要在这里加，然后再往后，要在这里把 embedding 和 traffic matrix 进行综合
        # 这里和dote的一样
    

    def forward(self, input_):
        

        return output