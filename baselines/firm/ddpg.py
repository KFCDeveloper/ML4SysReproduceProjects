import os.path
import matplotlib.pyplot as plt
from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

from ddpg.noise import OrnsteinUhlenbeckActionNoise as OUNoise
from ddpg.replaybuffer import Buffer
from ddpg.actorcritic import Actor, Critic
from util import *

# hyperparameters
NUM_STATES = 6
NUM_ACTIONS = 5
NUM_EPISODES = 9000
NUM_STEPS = 300
ACTOR_LR = 0.0003
CRITIC_LR = 0.003
SIGMA = 0.2
BUFFER_SIZE = 100000
MINIBATCH_SIZE = 64
CHECKPOINT_DIR = './checkpoints/'
DISCOUNT = 0.9
EPSILON = 1.0
EPSILON_DECAY = 1e-6
TAU = 0.001
WARMUP = 70  # should be greater than the minibatch size

PLOT_FIG = False
SAVE_FIG = False
SAVE_TO_FILE = False


# convert a state variable [s1, s2, ..., s_n] to a state tensor
def observation_to_state(state_list):
    return torch.FloatTensor(state_list).view(1, -1)


class DDPG:
    def __init__(self, env, function_name):
        self.env = env
        self.function_name = function_name
        self.state_dim = NUM_STATES
        self.action_dim = NUM_ACTIONS
        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim, self.action_dim)
        self.target_actor = deepcopy(Actor(self.state_dim, self.action_dim))
        self.target_critic = deepcopy(Critic(self.state_dim, self.action_dim))
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        self.critic_loss = nn.MSELoss()
        self.noise = OUNoise(mu=np.zeros(self.action_dim), sigma=SIGMA)
        self.replay_buffer = Buffer(BUFFER_SIZE)
        self.batch_size = MINIBATCH_SIZE
        self.checkpoint_dir = CHECKPOINT_DIR
        self.discount = DISCOUNT
        self.warmup = WARMUP
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.reward_graph = []
        self.smoothed_reward_graph = []
        self.start = 0
        self.end = NUM_EPISODES
        self.tau = TAU

    # calculate target Q-value as reward and bootstrapped Q-value of next state via the target actor and target critic
    # inputs: Batch of next states, rewards and terminal flags of size self.batch_size
    # output: Batch of Q-value targets
    def get_q_target(self, next_state_batch, reward_batch, terminal_batch):
        target_batch = torch.FloatTensor(reward_batch)
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: True if not s else False, terminal_batch)))
        next_state_batch = torch.cat(next_state_batch)
        next_action_batch = self.target_actor(next_state_batch)
        q_next = self.target_critic(next_state_batch, next_action_batch)

        non_final_mask = self.discount * non_final_mask.type(torch.FloatTensor)
        target_batch += non_final_mask * q_next.squeeze().data

        return Variable(target_batch).view(-1, 1)

    # weighted average update of the target network and original network
    # Inputs: target actor(critic) and original actor(critic)
    def update_targets(self, target, original):
        for target_param, original_param in zip(target.parameters(), original.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * original_param.data)

    # get the action that returns the maximum Q-value
    # inputs: Current state of the episode
    # output: the action which maximizes the Q-value of the current state-action pair
    def get_max_action(self, curr_state):
        noise = self.epsilon * Variable(torch.FloatTensor(self.noise()))
        action = self.actor(curr_state)
        action_with_noise = action + noise

        # get the action with max value
        action_list = action_with_noise.tolist()[0]
        max_action = max(action_list)
        max_index = action_list.index(max_action)

        return max_index, action_with_noise

    # training of the original and target actor-critic networks
    def train(self):
        # create the checkpoint directory if not created
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        print('Training started...')

        for episode in range(self.start, self.end):
            print('Episode #' + str(episode) + ':')
            state_list = self.env.reset(self.function_name)
            episode_reward = 0
            for step in range(NUM_STEPS):
                print('Step #' + str(step) + ':')

                # print current state
                print_state(state_list)

                # get max action
                curr_state_tensor = Variable(observation_to_state(state_list))
                self.actor.eval()
                action_idx, action_to_buffer = self.get_max_action(curr_state_tensor)

                action_to_execute = {
                    'vertical_cpu': 0,
                    'vertical_memory': 0,
                    'horizontal': 0,
                }

                if action == 0:
                    # do nothing
                    pass
                elif action == 1:
                    # scaling out
                    action_to_execute['horizontal'] = HORIZONTAL_SCALING_STEP
                elif action == 2:
                    # scaling in
                    action_to_execute['horizontal'] = -HORIZONTAL_SCALING_STEP
                elif action == 3:
                    # scaling up CPU
                    action_to_execute['vertical_cpu'] = VERTICAL_SCALING_STEP
                elif action == 4:
                    # scaling down CPU
                    action_to_execute['vertical_cpu'] = -VERTICAL_SCALING_STEP
                elif action == 5:
                    # scaling up memory
                    action_to_execute['vertical_memory'] = VERTICAL_SCALING_STEP
                elif action == 6:
                    # scaling down memory
                    action_to_execute['vertical_memory'] = -VERTICAL_SCALING_STEP

                next_state_dict, reward = self.env.step(action_to_execute)
                next_state = convert_state_dict_to_list(next_state_dict)

                next_state_tensor = Variable(next_state)
                episode_reward += reward

                # update the replay buffer
                self.replay_buffer.append((curr_state_tensor, action_to_buffer, next_state_tensor, reward, done))

                # training loop
                if len(self.replay_buffer) >= self.warmup:
                    curr_state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = \
                        self.replay_buffer.sample_batch(self.batch_size)
                    curr_state_batch = torch.cat(curr_state_batch)
                    action_batch = torch.cat(action_batch)

                    q_prediction_batch = self.critic(curr_state_batch, action_batch)
                    q_target_batch = self.get_q_target(next_state_batch, reward_batch, terminal_batch)

                    # critic update
                    self.critic_optimizer.zero_grad()
                    critic_loss = self.critic_loss(q_prediction_batch, q_target_batch)
                    print('Critic loss: {}'.format(critic_loss))
                    critic_loss.backward(retain_graph=True)
                    self.critic_optimizer.step()

                    # actor update
                    self.actor_optimizer.zero_grad()
                    actor_loss = -torch.mean(self.critic(curr_state_batch, self.actor(curr_state_batch)))
                    print('Actor loss: {}'.format(actor_loss))
                    actor_loss.backward(retain_graph=True)
                    self.actor_optimizer.step()

                    # update targets
                    self.update_targets(self.target_actor, self.actor)
                    self.update_targets(self.target_critic, self.critic)
                    self.epsilon -= self.epsilon_decay
                # end of current step
            # end of current episode
            print('EP #' + str(episode) + ': total reward =', episode_reward)
            self.reward_graph.append(episode_reward)
            self.smoothed_reward_graph.append(np.mean(self.reward_graph[-10:]))

            # save to checkpoint
            if episode % 20 == 0:
                self.save_checkpoint(episode)

            # plot the reward graph
            if PLOT_FIG:
                if episode % 1000 == 0 and episode != 0:
                    plt.plot(self.reward_graph, color='darkorange')
                    plt.plot(self.smoothed_reward_graph, color='b')
                    plt.xlabel('Episodes')
                    if SAVE_FIG:
                        plt.savefig('ep' + str(episode) + '.png')
        # end of all episodes
        if PLOT_FIG:
            plt.plot(self.reward_graph, color='darkorange')
            plt.plot(self.smoothed_reward_graph, color='b')
            plt.xlabel('Episodes')
            if SAVE_FIG:
                plt.savefig('final.png')

        if SAVE_TO_FILE:
            # write rewards to file
            file = open("ddpg_smoothed_rewards.txt", "w")
            for reward in self.smoothed_reward_graph:
                file.write(str(reward) + "\n")
            file.close()
            file = open("ddpg_episode_rewards.txt", "w")
            for reward in self.reward_graph:
                file.write(str(reward) + "\n")
            file.close()

    # save checkpoints to file
    def save_checkpoint(self, episode_num):
        checkpoint_name = self.checkpoint_dir + 'ddpg-ep{}.pth.tar'.format(episode_num)
        checkpoint = {
            'episode': episode_num,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'replay_buffer': self.replay_buffer,
            'reward_graph': self.reward_graph,
            'epsilon': self.epsilon
        }

        torch.save(checkpoint, checkpoint_name)

    # load checkpoints from file
    def load_checkpoint(self, checkpoint_file_name):
        if os.path.isfile(checkpoint_file_name):
            print('Loading checkpoint...')
            checkpoint = torch.load(checkpoint_file_name)
            self.start = checkpoint['episode'] + 1
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.target_actor.load_state_dict(checkpoint['target_actor'])
            self.target_critic.load_state_dict(checkpoint['target_critic'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            self.replay_buffer = checkpoint['replay_buffer']
            self.reward_graph = checkpoint['reward_graph']
            self.epsilon = checkpoint['epsilon']
            print('Checkpoint successfully loaded')
        else:
            raise OSError('Checkpoint not found!')
