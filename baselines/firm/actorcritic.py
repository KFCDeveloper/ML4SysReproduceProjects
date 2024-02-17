import numpy as np
import torch
import torch.nn as nn

HIDDEN_LAYER1 = 40
HIDDEN_LAYER2 = 40
W_FINAL = 0.003


# utility function for initializing actor and critic
def fan_in_init(size, fan_in=None):
    fan_in = fan_in or size[0]
    w = 1. / np.sqrt(fan_in)
    return torch.Tensor(size).uniform_(-w, w)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.norm0 = nn.BatchNorm1d(self.state_dim)

        self.fc1 = nn.Linear(self.state_dim, HIDDEN_LAYER1)
        self.fc1.weight.data = fan_in_init(self.fc1.weight.data.size())
        self.bn1 = nn.BatchNorm1d(HIDDEN_LAYER1)

        self.fc2 = nn.Linear(HIDDEN_LAYER1, HIDDEN_LAYER2)
        self.fc2.weight.data = fan_in_init(self.fc2.weight.data.size())

        self.bn2 = nn.BatchNorm1d(HIDDEN_LAYER2)

        self.fc3 = nn.Linear(HIDDEN_LAYER2, self.action_dim)
        self.fc3.weight.data.uniform_(-W_FINAL, W_FINAL)

        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, ip):
        ip_norm = self.norm0(ip)
        h1 = self.ReLU(self.fc1(ip_norm))
        h1_norm = self.bn1(h1)
        h2 = self.ReLU(self.fc2(h1_norm))
        h2_norm = self.bn2(h2)
        action = self.Tanh((self.fc3(h2_norm)))
        # action = self.Softmax(self.fc3(h2_norm))
        return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(self.state_dim, HIDDEN_LAYER1)
        self.fc1.weight.data = fan_in_init(self.fc1.weight.data.size())

        self.bn1 = nn.BatchNorm1d(HIDDEN_LAYER1)
        self.fc2 = nn.Linear(HIDDEN_LAYER1 + self.action_dim, HIDDEN_LAYER2)
        self.fc2.weight.data = fan_in_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(HIDDEN_LAYER2, 1)
        self.fc3.weight.data.uniform_(-W_FINAL, W_FINAL)

        self.ReLU = nn.ReLU()

    def forward(self, ip, action):
        h1 = self.ReLU(self.fc1(ip))
        h1_norm = self.bn1(h1)
        h2 = self.ReLU(self.fc2(torch.cat([h1_norm, action], dim=1)))
        q_val = self.fc3(h2)
        return q_val
