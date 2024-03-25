import os
import psutil
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from util import *


class ActorNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_):
        input_ = torch.FloatTensor(input_)
        output = self.relu(self.fc1(input_))
        output = self.relu(self.fc2(output))
        output = self.fc3(output)
        if not FLAG_CONTINUOUS_ACTION:
            output = self.softmax(output)

        return output


class CriticNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CriticNetwork, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, input_):
        input_ = torch.FloatTensor(input_)
        output = self.relu(self.fc1(input_))
        output = self.relu(self.fc2(output))
        output = self.fc3(output)

        return output


def calc_gae(rewards):
    returns = []
    for episode_rewards in reversed(rewards):
        discounted_return = 0.0
        # Caution: Episodes might have different lengths if stopped earlier
        for reward in reversed(episode_rewards):
            discounted_return = reward + discounted_return * DISCOUNT
            returns.insert(0, discounted_return)

    returns = torch.FloatTensor(returns)
    return returns


def visualization(iteration_rewards, smoothed_rewards):
    plt.plot(iteration_rewards, color='steelblue', alpha=0.3)  # total rewards in an iteration or episode
    plt.plot(smoothed_rewards, color='steelblue')  # (moving avg) rewards
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward per Episode')

    plt.tight_layout()
    if not SAVE_FIG:
        plt.show()
    else:
        plt.savefig('final.pdf')


class PPO:
    def __init__(self, env):
        self.env = env

        self.state_size = NUM_STATES
        self.action_size = NUM_ACTIONS

        self.actor = ActorNetwork(self.state_size, HIDDEN_SIZE, self.action_size)
        self.critic = CriticNetwork(self.state_size, HIDDEN_SIZE, 1)

        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=LR)

        self.cov = torch.diag(torch.ones(self.action_size, ) * 0.5)

        self.skip_update = False

        self.num_same_parameter_actor = 0
        self.num_same_parameter_critic = 0
        self.parameter_actor = None
        self.parameter_critic = None

        # store recent episode rewards
        self.recent_rewards = []

    # skip update for the policy and critic network, i.e., policy evaluation/serving stage
    def disable_update(self):
        self.skip_update = True

    # enable update for the policy and critic network, i.e., policy training stage
    def enable_update(self):
        self.skip_update = False

    def calc_action(self, state):
        if FLAG_CONTINUOUS_ACTION:
            mean = self.actor(state)
            dist = torch.distributions.MultivariateNormal(mean, self.cov)
        else:
            action_probs = self.actor(state)
            dist = torch.distributions.Categorical(action_probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().numpy(), log_prob.detach()

    def train(self):
        # create the checkpoint and log directory if not created
        if not os.path.exists(CHECKPOINT_DIR + self.env.app_name + '/'):
            os.makedirs(CHECKPOINT_DIR + self.env.app_name + '/')
        if not os.path.exists(LOG_DIR + self.env.app_name + '/'):
            os.makedirs(LOG_DIR + self.env.app_name + '/')

        # for plots
        iteration_rewards = []
        smoothed_rewards = []

        pid = os.getpid()
        python_process = psutil.Process(pid)
        for iteration in range(TOTAL_ITERATIONS):
            # get resource profiles
            memory_usage = python_process.memory_info()[0] / 2. ** 20  # memory use in MB
            cpu_util = python_process.cpu_percent(interval=None)
            print('RL Agent Memory Usage:', memory_usage, 'MB', '| CPU Util:', cpu_util)

            states = []
            actions = []
            rewards = []
            log_probs = []

            for episode in range(EPISODES_PER_ITERATION):
                state_dict = self.env.reset()
                state = convert_state_dict_to_list(state_dict)
                episode_rewards = []

                for step in range(EPISODE_LENGTH):
                    action, log_prob = self.calc_action(state)

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

                    states.append(state)
                    episode_rewards.append(reward)
                    log_probs.append(log_prob)
                    if FLAG_CONTINUOUS_ACTION:
                        actions.append(action)
                    else:
                        actions.append(action.item())

                    # verbose
                    if episode % 5 == 0 and iteration % 50 == 0:
                        print_step_info(step, state_dict, action_to_execute, reward)

                    state = next_state
                    state_dict = next_state_dict

                # end of one episode
                rewards.append(episode_rewards)
                if len(self.recent_rewards) < MAX_NUM_REWARDS_TO_CHECK:
                    # add the episode reward to the list
                    self.recent_rewards.append(np.sum(episode_rewards))
                else:
                    # remove the oldest item first
                    self.recent_rewards.pop(0)
                    # add the episode reward to the list
                    self.recent_rewards.append(np.sum(episode_rewards))

                    # check during policy-serving stage if retraining is needed
                    if self.skip_update:
                        print('Checking if policy re-training is needed...')
                        avg_reward = np.mean(self.recent_rewards)
                        std_reward = np.std(self.recent_rewards)
                        if avg_reward < REWARD_AVG_THRESHOLD or std_reward >= REWARD_STD_THRESHOLD:
                            # trigger retraining
                            self.skip_update = False
                            print('Re-training is enabled! avg(rewards) =', avg_reward, 'std(rewards) =', std_reward)

                    # check during policy-training stage if training is done
                    if not self.skip_update:
                        avg_reward = np.mean(self.recent_rewards)
                        std_reward = np.std(self.recent_rewards)
                        if avg_reward >= REWARD_AVG_THRESHOLD and std_reward < REWARD_STD_THRESHOLD:
                            # stop retraining
                            self.skip_update = True
                            print('Training is completed! avg(rewards) =', avg_reward, 'std(rewards) =', std_reward)

            # end of one iteration
            iteration_rewards.append(np.mean([np.sum(episode_rewards) for episode_rewards in rewards]))
            smoothed_rewards.append(np.mean(iteration_rewards[-10:]))

            # states = torch.FloatTensor(states)
            states = torch.FloatTensor(np.array(states))
            if FLAG_CONTINUOUS_ACTION:
                actions = torch.FloatTensor(actions)
            else:
                actions = torch.IntTensor(actions)
            log_probs = torch.FloatTensor(log_probs)

            average_rewards = np.mean([np.sum(episode_rewards) for episode_rewards in rewards])
            print('Iteration:', iteration, '- Average rewards across episodes:', np.round(average_rewards, decimals=3),
                  '| Moving average:', np.round(np.mean(iteration_rewards[-10:]), decimals=3))

            if SAVE_TO_FILE:
                all_rewards = [reward for reward_ep in rewards for reward in reward_ep]
                self.save_trajectories(iteration, states, actions, all_rewards)
                print('Trajectory saved!')

            if self.skip_update:
                continue

            returns = calc_gae(rewards)

            values = self.critic(states).squeeze()
            advantage = returns - values.detach()
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            for epoch in range(SGD_EPOCHS):
                batch_size = states.size(0)  # whole batch size = num of steps
                # use mini-batch instead of the whole batch
                for mini_batch in range(batch_size // MINI_BATCH_SIZE):
                    ids = np.random.randint(0, batch_size, MINI_BATCH_SIZE)

                    values = self.critic(states[ids]).squeeze()
                    if FLAG_CONTINUOUS_ACTION:
                        mean = self.actor(states[ids])
                        dist = torch.distributions.MultivariateNormal(mean, self.cov)
                    else:
                        action_probs = self.actor(states[ids])
                        dist = torch.distributions.Categorical(action_probs)

                    log_probs_new = dist.log_prob(actions[ids])
                    entropy = dist.entropy().mean()

                    ratios = (log_probs_new - log_probs[ids]).exp()

                    surrogate1 = ratios * advantage[ids]
                    surrogate2 = torch.clamp(ratios, 1 - CLIP, 1 + CLIP) * advantage[ids]
                    actor_loss = - torch.min(surrogate1, surrogate2).mean()
                    critic_loss = (returns[ids] - values).pow(2).mean()

                    loss = actor_loss + CRITIC_LOSS_DISCOUNT * critic_loss - ENTROPY_COEFFICIENT * entropy
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                if self.parameter_actor is None:
                    self.parameter_actor = []
                    for parameter in self.actor.parameters():
                        self.parameter_actor.append(parameter.clone())
                else:
                    # compare the model parameters
                    is_equal = True
                    for idx, parameter in enumerate(list(self.actor.parameters())):
                        if not torch.equal(parameter, self.parameter_actor[idx]):
                            is_equal = False
                            break
                    if is_equal:
                        self.num_same_parameter_actor += 1
                    else:
                        self.num_same_parameter_actor = 0
                        self.parameter_actor = []
                        for parameter in self.actor.parameters():
                            self.parameter_actor.append(parameter.clone())
                if self.parameter_critic is None:
                    self.parameter_critic = []
                    for parameter in self.critic.parameters():
                        self.parameter_critic.append(parameter.clone())
                else:
                    # compare the model parameters one by one
                    is_equal = True
                    for idx, parameter in enumerate(list(self.critic.parameters())):
                        if not torch.equal(parameter, self.parameter_critic[idx]):
                            is_equal = False
                            # print('DEBUG: critic network parameter is updated')
                            break
                    if is_equal:
                        self.num_same_parameter_critic += 1
                    else:
                        self.num_same_parameter_critic = 0
                        # self.parameter_critic = list(self.critic.parameters())
                        self.parameter_critic = []
                        for parameter in self.critic.parameters():
                            self.parameter_critic.append(parameter.clone())

            if self.num_same_parameter_critic > MAX_SAME_ITERATIONS and\
                    self.num_same_parameter_actor > MAX_SAME_ITERATIONS:
                print('Model parameters are not updating! Turning to policy-serving stage...')
                self.skip_update = True

            # save to checkpoint
            if iteration % 100 == 0 and iteration != 0:
                self.save_checkpoint(iteration)

        # plot
        if PLOT_FIG:
            visualization(iteration_rewards, smoothed_rewards)

        # write rewards to file
        if SAVE_TO_FILE:
            file = open(LOG_DIR + self.env.app_name + '/' + "ppo_smoothed_rewards.txt", "w")
            for reward in smoothed_rewards:
                file.write(str(reward) + "\n")
            file.close()
            file = open(LOG_DIR + self.env.app_name + '/' + "ppo_iteration_rewards.txt", "w")
            for reward in iteration_rewards:
                file.write(str(reward) + "\n")
            file.close()

    # load all model parameters from a saved checkpoint
    def load_checkpoint(self, checkpoint_file_path):
        if os.path.isfile(checkpoint_file_path):
            print('Loading checkpoint...')
            checkpoint = torch.load(checkpoint_file_path)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('Checkpoint successfully loaded!')
        else:
            raise OSError('Checkpoint not found!')

    # save all model parameters to a checkpoint
    def save_checkpoint(self, episode_num):
        checkpoint_name = CHECKPOINT_DIR + self.env.app_name + '/' + 'ppo-ep{}.pth.tar'.format(episode_num)
        checkpoint = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

        torch.save(checkpoint, checkpoint_name)

    # record trajectories
    def save_trajectories(self, episode_num, states, actions, rewards):
        file = open(LOG_DIR + self.env.app_name + '/' + "ppo_trajectories.csv", "a")
        count = 0
        for state in states:
            file.write(str(episode_num) + ',' + ','.join([str(item.item()) for item in state]) + ',' + str(actions[count].item()) + ',' + str(rewards[count]) + "\n")
            count += 1
        file.close()
