import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import *
from util import *


class RNNEmbedding(nn.Module):
    def __init__(self, N, K, task, num_channels=NUM_FEATURES_PER_SHOT, embedding_dim=EMBEDDING_DIM, use_cuda=False, verbose=True):
        # N-way (N classes, N = 1 for non-classification tasks), K-shot (K samples used to generate per embedding)
        super(RNNEmbedding, self).__init__()
        if task == 'wa':
            # num_channels is the dimension of x which is equal to the number of features per shot/sample
            # num_channels = NUM_FEATURES_PER_SHOT
            if verbose:
                print('Task:', task)
                print('Number of features per sample:', num_channels)
        else:
            raise ValueError('Not recognized task!')

        # configs for a 2-layer bi-directional RNN
        self.hidden_size = RNN_HIDDEN_SIZE
        self.num_layers = RNN_NUM_LAYERS
        self.bidirectional = True
        self.directions = 2 if self.bidirectional else 1
        self.gru = nn.GRU(num_channels, self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=self.bidirectional)

        # FC layer for embedding
        # self.embedding_layer = nn.ReLU(nn.Linear(self.hidden_size * self.num_layers * self.directions, embedding_dim))
        self.embedding_layer = nn.Linear(self.hidden_size * self.num_layers * self.directions, embedding_dim)
        self.relu = nn.ReLU()
        self.embedding_dim = embedding_dim

        self.N = N
        self.K = K
        self.use_cuda = use_cuda

    # forward with embedding
    def forward(self, input):
        # input to the embedding layer is the features to the samples
        x = np.array([input[0].numpy()[i * num_channels : (i+1) * num_channels] for i in range(self.K)])
        # creating a tensor from a list of numpy.ndarrays is extremely slow -> convert the list to a single numpy.ndarray first
        x = torch.FloatTensor(x)
        x = x.view((1, self.K, -1))
        hidden = self.init_hidden()

        output, x = self.gru(x, hidden)

        # attach the generated embedding with the input
        x = x.view((1, 1, -1))
        input = input.view((1, 1, -1))
        x = torch.cat((x, input), 2)

        # append a fully connected neural network
        x = self.relu(self.fc1(x.float()))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def init_hidden(self, num_sequences=1):
        return torch.zeros((self.directions * self.num_layers, num_sequences, self.hidden_size))


class SnailEmbedding(nn.Module):
    def __init__(self, N, K, task, use_cuda=False):
        # N-way (N classes, N = 1 for non-classification tasks), K-shot (K samples, default K = 1)
        super(RNNEmbedding, self).__init__()
        if task == 'wa':
            # num_channels is the dimension of x which is equal to the number of features per shot/sample
            num_channels = NUM_FEATURES_PER_SHOT
        else:
            raise ValueError('Not recognized task!')

        num_filters = int(math.ceil(math.log(N * K + 1, 2)))
        self.attention1 = AttentionBlock(num_channels, 64, 32)
        num_channels += 32
        self.tc1 = TCBlock(num_channels, N * K + 1, 128)
        num_channels += num_filters * 128
        self.attention2 = AttentionBlock(num_channels, 256, 128)
        num_channels += 128
        self.tc2 = TCBlock(num_channels, N * K + 1, 128)
        num_channels += num_filters * 128
        self.attention3 = AttentionBlock(num_channels, 512, 256)
        num_channels += 256

        self.fc = nn.Linear(num_channels, N)
        self.fc1 = nn.Linear(num_channels * K + NUM_FEATURES_PER_SHOT * K + NUM_CONFIG_PARAMS, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, N)
        self.relu = nn.ReLU()

        self.N = N
        self.K = K
        self.use_cuda = use_cuda

    # forward with embedding
    def forward(self, input):
        # input to the embedding layer is the features to the samples (excluding the config to predict for)
        x = np.array([input[0].numpy()[i * NUM_FEATURES_PER_SHOT : (i+1) * NUM_FEATURES_PER_SHOT] for i in range(self.K)])
        # # creating a tensor from a list of numpy.ndarrays is extremely slow -> convert the list to a single numpy.ndarray first
        x = torch.FloatTensor(x)
        x = x.view((1, self.K, -1))

        # apply the embedding layer
        x = self.attention1(x.float())
        x = self.tc1(x)
        x = self.attention2(x)
        x = self.tc2(x)
        x = self.attention3(x)
        # add a fully connected layer before passing the embedding
        # x = self.fc(x)

        # attach the generated embedding with the input
        x = x.view((1, 1, -1))
        x = torch.cat((x, torch.tensor(np.array([[input[0].numpy()[-NUM_CONFIG_PARAMS:]]]))), 2)

        # append a fully connected neural network
        x = self.relu(self.fc1(x.float()))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
