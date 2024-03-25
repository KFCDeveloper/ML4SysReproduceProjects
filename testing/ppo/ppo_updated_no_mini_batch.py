# import time
import torch
from torch import nn
import matplotlib.pyplot as plt
from util import *

PLOT_FIG = True
SAVE_FIG = False
SAVE_TO_FILE = False

TOTAL_ITERATIONS = 2000
EPISODES_PER_ITERATION = 20
EPISODE_LENGTH = 200

# hyperparameters
DISCOUNT = 0.99
HIDDEN_SIZE = 32
LR = 3e-4  # 5e-3 5e-6
SGD_EPOCHS = 5
CLIP = 0.2
ENTROPY_COEFFICIENT = 0.01  # 0.001
CRITIC_LOSS_DISCOUNT = 0.05  # 0.03

FLAG_CONTINUOUS_ACTION = False

MAX_SAME_ITERATIONS = 2 * SGD_EPOCHS


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


class PPOWithoutMiniBatch:
    def __init__(self, env, function_name):
        self.env = env
        self.function_name = function_name

        self.state_size = NUM_STATES
        self.action_size = NUM_ACTIONS  # NUM_TOTAL_ACTIONS

        self.actor = ActorNetwork(self.state_size, HIDDEN_SIZE, self.action_size)
        self.critic = CriticNetwork(self.state_size, HIDDEN_SIZE, 1)

        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=LR)

        self.cov = torch.diag(torch.ones(self.action_size, ) * 0.5)

        self.num_same_parameter_actor = 0
        self.num_same_parameter_critic = 0
        self.parameter_actor = None
        self.parameter_critic = None

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
        # for plots
        iteration_rewards = []
        smoothed_rewards = []

        for iteration in range(TOTAL_ITERATIONS):
            states = []
            actions = []
            rewards = []
            log_probs = []

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

                    # verbose
                    if episode % 5 == 0 and iteration % 50 == 0:
                        print_step_info(step, state, action_to_execute, reward)

                    if done:
                        break
                    state = next_state

                # end of one episode
                rewards.append(episode_rewards)

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

            returns = calc_gae(rewards)

            values = self.critic(states).squeeze()
            advantage = returns - values.detach()
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            for epoch in range(SGD_EPOCHS):
                # use the whole batch (mini-batch not implemented)

                values = self.critic(states).squeeze()
                if FLAG_CONTINUOUS_ACTION:
                    mean = self.actor(states)
                    dist = torch.distributions.MultivariateNormal(mean, self.cov)
                else:
                    action_probs = self.actor(states)
                    dist = torch.distributions.Categorical(action_probs)
                log_probs_new = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                ratios = (log_probs_new - log_probs).exp()

                surrogate1 = ratios * advantage
                surrogate2 = torch.clamp(ratios, 1 - CLIP, 1 + CLIP) * advantage
                actor_loss = - torch.min(surrogate1, surrogate2).mean()
                critic_loss = (returns - values).pow(2).mean()

                loss = actor_loss + CRITIC_LOSS_DISCOUNT * critic_loss - ENTROPY_COEFFICIENT * entropy
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # print('debug:\n', self.actor.parameters())
                # i = 0
                # for param in self.actor.parameters():
                #     if i == 2:
                #         print(type(param), param.size(), param, float(param[31][2]))
                #     i += 1
                # print('debug:\n', self.critic.parameters())
                # i = 0
                # for param in self.critic.parameters():
                #     if i == 2:
                #         print(type(param), param.size(), param, float(param[31][2]))
                #     i += 1

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
                            # print('DEBUG: critic network parameter is updated')
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
                # print('DEBUG: # same critic params =', self.num_same_parameter_critic)
                # print('DEBUG: # same actor params =', self.num_same_parameter_actor)
                # print('DEBUG: critic parameter =', self.parameter_critic)
                # print('DEBUG: actor parameter =', self.parameter_actor)
                # exit()

            if self.num_same_parameter_critic > MAX_SAME_ITERATIONS and\
                    self.num_same_parameter_actor > MAX_SAME_ITERATIONS:
                break

        # plot
        if PLOT_FIG:
            plt.plot(iteration_rewards, color='darkorange')  # total rewards in an iteration
            plt.plot(smoothed_rewards, color='b')  # moving avg rewards
            plt.xlabel('Iteration')
            plt.show()
            if SAVE_FIG:
                plt.savefig('final.png')

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

    def load_checkpoint(self, checkpoint_file_path):
        pass

    def save_checkpoint(self, episode_num):
        pass
