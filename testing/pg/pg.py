import torch
import numpy as np
import matplotlib.pyplot as plt
# import time

from pg.policy_network import PolicyNetwork
from util import *

PLOT_FIG = True
SAVE_FIG = False
SAVE_TO_FILE = False

# hyperparameters
NUM_STATES = NUM_STATES
NUM_ACTIONS = NUM_TOTAL_ACTIONS
HIDDEN_SIZE = 40
NUM_EPS = 4000
NUM_STEPS = 100
GAMMA = 0.9


class PG:
    def __init__(self, env, function_name):
        self.env = env
        self.function_name = function_name

        self.state_dim = NUM_STATES
        self.action_dim = NUM_ACTIONS
        self.hidden_size = HIDDEN_SIZE

        # policy network
        self.policy_network = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_size)

        self.max_num_episodes = NUM_EPS
        self.max_num_steps = NUM_STEPS
        self.gamma = GAMMA

    # update the policy network
    def update_policy(self, rewards, log_probabilities):
        discounted_rewards = []

        for t in range(len(rewards)):
            gt = 0
            pw = 0
            for r in rewards[t:]:
                gt = gt + self.gamma ** pw * r
                pw = pw + 1
            discounted_rewards.append(gt)

        discounted_rewards = torch.tensor(discounted_rewards)
        # normalize discounted rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std(0) + 1e-9)

        policy_gradient = []
        for log_probability, gt in zip(log_probabilities, discounted_rewards):
            policy_gradient.append(-log_probability * gt)
            # policy_gradient.append(1.0 / log_probability * gt)

        self.policy_network.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        # policy_gradient.backward()
        policy_gradient.backward(retain_graph=True)
        self.policy_network.optimizer.step()

    # training
    def train(self):
        all_rewards = []
        smoothed_rewards = []

        print('Training started...')
        for episode in range(self.max_num_episodes):
            # initialization
            state_list = self.env.reset(self.function_name)
            log_probabilities = []
            rewards = []

            for step in range(self.max_num_steps):
                # if episode % 500 == 0:
                #     print('Step #' + str(step) + ':')
                #
                #     # print current state
                #     print_state(state_list)

                # start = time.perf_counter()
                state_vector = np.asarray(state_list, dtype=float)
                action_idx, log_probability = self.policy_network.get_action(state_vector)
                # end = time.perf_counter() - start
                # print('{:.6f}s for the RL network inference'.format(end))

                action = {
                    'vertical': 0,
                    'horizontal': 0,
                    'scale_to': -1
                }
                """
                if action_idx == 0:
                    # do nothing
                    pass
                elif action_idx == 1:
                    # scaling out
                    action['horizontal'] = HORIZONTAL_SCALING_STEP
                elif action_idx == 2:
                    # scaling in
                    action['horizontal'] = -HORIZONTAL_SCALING_STEP
                elif action_idx == 3:
                    # scaling up
                    action['vertical'] = VERTICAL_SCALING_STEP
                elif action_idx == 4:
                    # scaling down
                    action['vertical'] = -VERTICAL_SCALING_STEP
                """
                action['scale_to'] = action_idx

                next_state_list, reward, done = self.env.step(self.function_name, action)

                # print action and reward
                if episode % 500 == 0:
                    # print_action(action)
                    # print('Reward:', reward)
                    print_step_info(step, state_list, action, reward)
                state_list = next_state_list

                log_probabilities.append(log_probability)
                rewards.append(reward)

            self.update_policy(rewards, log_probabilities)
            all_rewards.append(np.sum(rewards))
            smoothed_rewards.append(np.mean(all_rewards[-10:]))
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
            # end of an episode

        if PLOT_FIG:
            plt.plot(all_rewards, color='darkorange')  # total rewards in an iteration or episode
            plt.plot(smoothed_rewards, color='b')      # (moving avg) rewards
            plt.xlabel('Episodes')
            plt.show()
            if SAVE_FIG:
                plt.savefig('final.png')

        if SAVE_TO_FILE:
            # write rewards to file
            file = open("pg_smoothed_rewards.txt", "w")
            for reward in smoothed_rewards:
                file.write(str(reward) + "\n")
            file.close()
            file = open("pg_episode_rewards.txt", "w")
            for reward in all_rewards:
                file.write(str(reward) + "\n")
            file.close()

    def load_checkpoint(self, checkpoint_file_path):
        pass

    def save_checkpoint(self, episode_num):
        pass