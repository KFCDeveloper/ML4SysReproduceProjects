import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.autograd import Variable

import numpy as np

LR = 3e-4


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=LR):
        super(PolicyNetwork, self).__init__()

        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear_act = nn.Linear(hidden_size, int(num_actions))
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    # a neural network two hidden layers with the same size
    def forward(self, state):
        x = f.relu(self.linear1(state))
        x = f.relu(self.linear2(x))
        x_act = self.linear_act(x)
        x_act = f.softmax(x_act, dim=1)

        # return the probability list of the actions
        return x_act

    # get the action with the maximum probability
    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probabilities = self.forward(Variable(state))

        highest_prob_action = np.random.choice(int(self.num_actions), p=np.squeeze(probabilities.detach().numpy()))
        log_probabilities = torch.log(probabilities.squeeze(0)[highest_prob_action])

        return highest_prob_action, log_probabilities
