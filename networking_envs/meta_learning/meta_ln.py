import os
import psutil
import torch
from torch import nn
import matplotlib.pyplot as plt
import heapq

from util import *
from ppo.rnn import RNNEmbedding

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

class MetaCriticNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, env_dim, agent, verbose=True):
        super(MetaCriticNetwork, self).__init__()

        self.env_dim = env_dim
        self.agent = agent

        # rnn embedding
        num_features_per_sample = env_dim['state'] + env_dim['action'] + env_dim['reward']
        self.rnn = RNNEmbedding(1, BUFFER_SIZE, 'wa', num_channels=MAX_TIMESTEPS_PER_EPISODE, verbose=verbose)

        self.fc1 = nn.Linear(input_size + self.rnn.embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, input_):
        padded_states, padded_actions, padded_rewards = get_padded_trajectories(self.agent.episode_buffer, state_dim=self.env_dim['state'], action_dim=self.env_dim['action'])

        # encode RL trajectories with the RNN and generate the embedding
        rnn_input = torch.cat((padded_states, padded_actions, padded_rewards), dim=-2)
        num_sequences = rnn_input.shape[0]
        hidden = self.rnn.init_hidden(num_sequences=num_sequences)
        rnn_output, hidden_state = self.rnn.gru(rnn_input, hidden)  # use the last hidden state to generate the embedding
        rnn_output = rnn_output[:, -1, :]
        hidden_state = torch.mean(hidden_state, dim=1, keepdim=True)
        hidden_state = hidden_state.view((1, 1, -1))
        embedding = self.rnn.embedding_layer(hidden_state[0][0])
        embedding = self.rnn.relu(embedding)

        input_ = torch.FloatTensor(input_)
        # concat input + embedding
        embedding = embedding.reshape(1, -1)
        embedding = embedding.repeat(input_.size(0), 1)
        input_ = torch.cat((input_, embedding), dim=1)
        output = self.relu(self.fc1(input_))
        output = self.relu(self.fc2(output))
        output = self.fc3(output)

        return output


class MetaActorNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, env_dim, agent, verbose=True):
        super(MetaActorNetwork, self).__init__()

        self.env_dim = env_dim
        self.agent = agent

        # rnn embedding
        num_features_per_sample = env_dim['state'] + env_dim['action'] + env_dim['reward']
        self.rnn = RNNEmbedding(1, BUFFER_SIZE, 'wa', num_channels=MAX_TIMESTEPS_PER_EPISODE, verbose=verbose)

        self.fc1 = nn.Linear(input_size + self.rnn.embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_):
        padded_states, padded_actions, padded_rewards = get_padded_trajectories(self.agent.episode_buffer, state_dim=self.env_dim['state'], action_dim=self.env_dim['action'])

        # encode RL trajectories with the RNN and generate the embedding
        rnn_input = torch.cat((padded_states, padded_actions, padded_rewards), dim=-2)
        num_sequences = rnn_input.shape[0]
        hidden = self.rnn.init_hidden(num_sequences=num_sequences)
        rnn_output, hidden_state = self.rnn.gru(rnn_input, hidden)  # use the last hidden state to generate the embedding
        rnn_output = rnn_output[:, -1, :]
        hidden_state = torch.mean(hidden_state, dim=1, keepdim=True)
        hidden_state = hidden_state.view((1, 1, -1))
        # embedding = self.rnn.embedding_layer(torch.cat((rnn_output[:, :self.rnn.hidden_size], rnn_output[:, self.rnn.hidden_size:]), dim=-1))
        embedding = self.rnn.embedding_layer(hidden_state[0][0])
        embedding = self.rnn.relu(embedding)
        # print('Generated embedding:', type(embedding), embedding.shape)

        input_ = torch.FloatTensor(input_)
        if len(input_.shape) > 1:
            # concat input + embedding for batched inputs
            embedding = embedding.reshape(1, -1)
            embedding = embedding.repeat(input_.size(0), 1)
            input_ = torch.cat((input_, embedding), dim=1)
        else:
            # concat input + embedding
            input_ = torch.cat((input_, embedding), dim=-1)
        output = self.relu(self.fc1(input_))
        output = self.relu(self.fc2(output))
        output = self.fc3(output)
        if not FLAG_CONTINUOUS_ACTION:
            output = self.softmax(output)

        return output