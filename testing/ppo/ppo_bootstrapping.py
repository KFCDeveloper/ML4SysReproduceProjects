import os

import psutil
import torch
from torch import nn
import matplotlib.pyplot as plt
from util import *

PLOT_FIG = False
SAVE_FIG = True
SAVE_TO_FILE = True

CHECKPOINT_DIR = './checkpoints/'

TOTAL_ITERATIONS = 1500
EPISODES_PER_ITERATION = 5
EPISODE_LENGTH = 200

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


class PPO:
    def __init__(self, env, function_name):
        self.env = env
        self.function_name = function_name

        self.state_size = NUM_STATES
        self.action_size = NUM_ACTIONS  # NUM_TOTAL_ACTIONS

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

    # skip update for the policy and critic network
    def disable_update(self):
        self.skip_update = True

    # enable update for the policy and critic network
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
        if not os.path.exists(CHECKPOINT_DIR):
            print('Checkpoints not found!')
            exit()
        self.load_checkpoint('./checkpoints/ppo-ep400.pth.tar')

        # for plots
        iteration_rewards = []
        smoothed_rewards = []

        # for explainability
        iteration_slo_preservations = []
        smoothed_slo_preservations = []
        iteration_cpu_utils = []
        smoothed_cpu_utils = []

        pid = os.getpid()
        python_process = psutil.Process(pid)
        for iteration in range(TOTAL_ITERATIONS):
            # get resource profiles
            memory_usage = python_process.memory_info()[0] / 2. ** 20  # memory use in MB
            cpu_util = python_process.cpu_percent(interval=None)/psutil.cpu_count()
            print('Memory use (MB):', memory_usage, 'CPU util (%):', cpu_util)

            states = []
            actions = []
            rewards = []
            log_probs = []
            all_cpu_utils = []
            all_slo_preservations = []

            for episode in range(EPISODES_PER_ITERATION):
                state = self.env.reset(self.function_name)[:NUM_STATES]
                episode_rewards = []

                for step in range(EPISODE_LENGTH):
                    action, log_prob = self.calc_action(state)

                    # print('debug: action =', action)

                    action_to_execute = {
                        'vertical': 0,
                        'horizontal': 0,
                        'scale_to': -1
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
                        # scaling up
                        action_to_execute['vertical'] = VERTICAL_SCALING_STEP
                    elif action == 4:
                        # scaling down
                        action_to_execute['vertical'] = -VERTICAL_SCALING_STEP
                    # action_to_execute['scale_to'] = int(action)

                    next_state, reward, done = self.env.step(self.function_name, action_to_execute)
                    next_state = next_state[:NUM_STATES]

                    states.append(state)
                    episode_rewards.append(reward)
                    log_probs.append(log_prob)
                    if FLAG_CONTINUOUS_ACTION:
                        actions.append(action)
                    else:
                        actions.append(action.item())

                    all_cpu_utils.append(next_state[0])
                    all_slo_preservations.append(next_state[1])

                    # verbose
                    if episode % 5 == 0 and iteration % 50 == 0:
                        print_step_info(step, state, action_to_execute, reward)

                    if done:
                        break
                    state = next_state

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
            iteration_slo_preservations.append(np.mean(all_slo_preservations))
            smoothed_slo_preservations.append(np.mean(iteration_slo_preservations[-10:]))
            iteration_cpu_utils.append(np.mean(all_cpu_utils))
            smoothed_cpu_utils.append(np.mean(iteration_cpu_utils[-10:]))

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

            if self.skip_update:
                continue

            returns = calc_gae(rewards)

            values = self.critic(states).squeeze()
            advantage = returns - values.detach()
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            for epoch in range(SGD_EPOCHS):
                batch_size = states.size(0)  # whole batch of size 4000 (= 20 * 200)
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
                            # print('DEBUG: actor network parameter is updated')
                            # print(parameter, self.parameter_actor[idx])
                            break
                    # list(self.actor.parameters()) == self.parameter_actor
                    if is_equal:
                        self.num_same_parameter_actor += 1
                    else:
                        self.num_same_parameter_actor = 0
                        self.parameter_actor = []
                        for parameter in self.actor.parameters():
                            self.parameter_actor.append(parameter.clone())
                        # self.parameter_actor = list(self.actor.parameters())
                if self.parameter_critic is None:
                    self.parameter_critic = []
                    for parameter in self.critic.parameters():
                        self.parameter_critic.append(parameter.clone())
                    # self.parameter_critic = list(self.critic.parameters())
                else:
                    # compare the model parameters one by one
                    is_equal = True
                    for idx, parameter in enumerate(list(self.critic.parameters())):
                        if not torch.equal(parameter, self.parameter_critic[idx]):
                            is_equal = False
                            break
                    # list(self.critic.parameters()) == self.parameter_critic:
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
                break

            # save to checkpoint
            if iteration % 100 == 0:
                self.save_checkpoint(iteration)

        # plot
        if PLOT_FIG:
            plt.plot(iteration_rewards, color='steelblue', alpha=0.3)  # total rewards in an iteration or episode
            plt.plot(smoothed_rewards, color='steelblue')  # (moving avg) rewards
            plt.xlabel('Episodes')
            plt.ylabel('Total Reward per Episode')

            plt.tight_layout()
            if not SAVE_FIG:
                plt.show()
            else:
                plt.savefig('final.pdf')

        # write rewards to file
        if SAVE_TO_FILE:
            file = open("ppo_smoothed_rewards.txt", "w")
            for reward in smoothed_rewards:
                file.write(str(reward) + "\n")
            file.close()
            file = open("ppo_iteration_rewards.txt", "w")
            for reward in iteration_rewards:
                file.write(str(reward) + "\n")
            file.close()

            # write cpu_utils and slo_preservations to file
            file = open("ppo_cpu_utils_all.txt", "w")
            for cpu_util in iteration_cpu_utils:
                file.write(str(cpu_util) + "\n")
            file.close()
            file = open("ppo_cpu_utils_smoothed.txt", "w")
            for cpu_util in smoothed_cpu_utils:
                file.write(str(cpu_util) + "\n")
            file.close()
            file = open("ppo_slo_preservation_all.txt", "w")
            for ratio in iteration_slo_preservations:
                file.write(str(ratio) + "\n")
            file.close()
            file = open("ppo_slo_preservation_smoothed.txt", "w")
            for ratio in smoothed_slo_preservations:
                file.write(str(ratio) + "\n")
            file.close()

    # policy serving
    def test(self):
        self.skip_update = True
        self.load_checkpoint('./checkpoints/final.pth.tar')
        self.train()

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
        checkpoint_name = CHECKPOINT_DIR + 'ppo-ep{}.pth.tar'.format(episode_num)
        checkpoint = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

        torch.save(checkpoint, checkpoint_name)
