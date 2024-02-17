from collections import namedtuple, deque
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

from util import *

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
NUM_EPISODES = 1000
NUM_STEPS = 100

PLOT_FIG = True
SAVE_FIG = True
SAVE_TO_FILE = True

CHECKPOINT_DIR = './checkpoints/'

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNNetwork(nn.Module):
    def __init__(self, num_states, num_actions):
        super(DQNNetwork, self).__init__()

        self.fc1 = nn.Linear(num_states, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_actions)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.FloatTensor(x)
        # print('DEBUG: x_dimension =', len(x), x)
        output = self.relu(self.fc1(x))
        output = self.relu(self.fc2(output))
        output = self.fc3(output)
        return output


class DQN:
    def __init__(self, env, function_name):
        self.env = env
        self.function_name = function_name

        self.state_size = NUM_STATES
        self.action_size = NUM_TOTAL_ACTIONS  # NUM_ACTIONS

        self.policy_net = DQNNetwork(self.state_size, self.action_size)
        self.target_net = DQNNetwork(self.state_size, self.action_size)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)

        self.steps_completed = 0

    """
    Select an action accordingly to an epsilon greedy policy.
    We’ll sometimes use our model for choosing the action, and sometimes we’ll just sample one uniformly.
    The probability of choosing a random action will start at EPS_START and will decay exponentially towards EPS_END.
    EPS_DECAY controls the rate of the decay.
    """
    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_completed / EPS_DECAY)
        self.steps_completed += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was found
                # so we pick action with the larger expected reward.
                # print('DEBUG: state =', state)
                # print('DEBUG: self.policy_net(state) =', self.policy_net(state))
                # print('DEBUG:', self.policy_net(state).argmax().view(1, 1))
                return self.policy_net(state).argmax().view(1, 1)
        else:
            random_action = torch.tensor([[random.randrange(self.action_size)]], dtype=torch.long)
            # print('DEBUG: random_action =', random_action)
            return random_action

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)
        # print('DEBUG: transitions =', transitions)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for detailed explanation).
        # This converts batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        # print('DEBUG: batch =', batch)

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        # print('DEBUG: non_final_mask =', non_final_mask)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        # print('DEBUG: non_final_next_states =', non_final_next_states)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # print('DEBUG: state_batch =', state_batch)
        # print('DEBUG: action_batch =', action_batch)
        # print('DEBUG: reward_batch =', reward_batch)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken. These are the
        # actions which would've been taken for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train(self):
        all_rewards = []
        smoothed_rewards = []

        # for explainability
        all_slo_preservations = []
        smoothed_slo_preservations = []
        all_cpu_utils = []
        smoothed_cpu_utils = []

        for episode in range(NUM_EPISODES):
            state = self.env.reset(self.function_name)[:NUM_STATES]
            rewards = []
            slo_preservations = []
            cpu_utils = []
            down_steps = 95-episode
            if down_steps < 0:
                down_steps = 0
            for step in range(NUM_STEPS - down_steps):
                action = self.select_action(state)
                # print('DEBUG: action =', action, action.item())

                action_to_execute = {'vertical': 0, 'horizontal': 0, 'scale_to': action.item()+1}

                next_state, reward, done = self.env.step(self.function_name, action_to_execute)
                next_state = next_state[:NUM_STATES]
                rewards.append(reward)
                cpu_utils.append(next_state[0])
                slo_preservations.append(next_state[1])
                reward = torch.tensor([[reward]])

                self.memory.push(torch.FloatTensor([state]), action, torch.FloatTensor([next_state]), reward)

                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # verbose
                if episode % 50 == 0:
                    print_step_info(step, state, action_to_execute, rewards[-1])
            # end of one episode
            all_rewards.append(np.sum(rewards))
            smoothed_rewards.append(np.mean(all_rewards[-10:]))
            all_cpu_utils.append(np.mean(cpu_utils))
            smoothed_cpu_utils.append(np.mean(all_cpu_utils[-10:]))
            all_slo_preservations.append(np.mean(slo_preservations))
            smoothed_slo_preservations.append(np.mean(all_slo_preservations[-10:]))

            if episode % 1 == 0:
                print("Episode: {}, total episode reward: {}, smoothed reward: {}".format(
                    episode, np.round(np.sum(rewards), decimals=3), np.round(np.mean(all_rewards[-10:]), decimals=3)))

            if PLOT_FIG:
                if episode % 1000 == 0 and episode != 0:
                    plt.plot(all_rewards, color='darkorange')  # total rewards in an episode
                    plt.plot(smoothed_rewards, color='b')      # (moving avg) rewards
                    plt.xlabel('Episodes')
                    if SAVE_FIG:
                        plt.savefig('ep' + str(episode) + '.png')

            # Update the target network, copying all weights and biases in DQN
            if episode % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        # end of all episodes

        if PLOT_FIG:
            plt.plot(all_rewards, color='steelblue', alpha=0.3)  # total rewards in an iteration or episode
            plt.plot(smoothed_rewards, color='steelblue')      # (moving avg) rewards
            plt.xlabel('Episodes')
            plt.ylabel('Total Reward per Episode')

            plt.tight_layout()
            if not SAVE_FIG:
                plt.show()
            else:
                plt.savefig('final.pdf')

        if SAVE_TO_FILE:
            # write rewards to file
            file = open("dqn_smoothed_rewards.txt", "w")
            for reward in smoothed_rewards:
                file.write(str(reward) + "\n")
            file.close()
            file = open("dqn_episode_rewards.txt", "w")
            for reward in all_rewards:
                file.write(str(reward) + "\n")
            file.close()

            # write cpu_utils and slo_preservations to file
            file = open("dqn_cpu_utils_all.txt", "w")
            for cpu_util in all_cpu_utils:
                file.write(str(cpu_util) + "\n")
            file.close()
            file = open("dqn_cpu_utils_smoothed.txt", "w")
            for cpu_util in smoothed_cpu_utils:
                file.write(str(cpu_util) + "\n")
            file.close()
            file = open("dqn_slo_preservation_all.txt", "w")
            for ratio in all_slo_preservations:
                file.write(str(ratio) + "\n")
            file.close()
            file = open("dqn_slo_preservation_smoothed.txt", "w")
            for ratio in smoothed_slo_preservations:
                file.write(str(ratio) + "\n")
            file.close()

    def load_checkpoint(self, checkpoint_file_path):
        if os.path.isfile(checkpoint_file_path):
            print('Loading checkpoint...')
            checkpoint = torch.load(checkpoint_file_path)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.memory = checkpoint['memory']
            print('Checkpoint successfully loaded!')
        else:
            raise OSError('Checkpoint not found!')

    def save_checkpoint(self, episode_num):
        checkpoint_name = CHECKPOINT_DIR + 'dqn-ep{}.pth.tar'.format(episode_num)
        checkpoint = {
            'episode': episode_num,
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'memory': self.memory
        }

        torch.save(checkpoint, checkpoint_name)
        print('Checkpoint saved to', checkpoint_name, 'successfully!')
