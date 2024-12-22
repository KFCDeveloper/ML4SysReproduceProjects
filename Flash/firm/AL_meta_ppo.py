import os
import psutil
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import gym
import heapq
import time
from env_for_simulation import Environment_simulation
from util import *
from models import MetaActorNetwork, MetaCriticNetwork
from AL_module import *

file_path = '../data-firm/compress_compress_decompress_output.csv'

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


def visualization(iteration_rewards, smoothed_rewards, data_path, testing=False):
    plt.figure()
    plt.plot(iteration_rewards, color='steelblue', alpha=0.3)  # total rewards in an iteration or episode
    plt.plot(smoothed_rewards, color='steelblue')  # (moving avg) rewards
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward per Episode')

    plt.tight_layout()
    if not SAVE_FIG:
        plt.show()
    else:
        if testing:
            plt.savefig(data_path + '/testing-reward-curve.pdf')
        else:
            plt.savefig(data_path + '/training-reward-curve.pdf')


class DictList(dict):
    def __setitem__(self, key, value):
        try:
            # Assumes there is a list on the key
            self[key].append(value)
        except KeyError: # If it fails, because there is no key
            super(DictList, self).__setitem__(key, value)
        except AttributeError: # If it fails because it is not a list
            super(DictList, self).__setitem__(key, [self[key], value])


class MetaPPO:
    def __init__(self, env, function_name, data_path, mode, device, verbose=True, bert_embedding=False):
        self.env = env
        self.function_name = function_name
        self.data_path = data_path
        self.verbose = verbose
        self.bert_embedding = bert_embedding

        self.state_size = NUM_STATES
        self.action_size = NUM_ACTIONS
        self.device = device
        lr = LR
        if mode == 'adapt':
            lr = FINE_TUNE_LR
        elif (mode == 'train' or mode == 'meta-train') and bert_embedding:
            lr = BERT_LR

        self.env_dim = {
            'state': self.state_size,
            'action': self.action_size,
            'reward': 1
        }

        self.actor = MetaActorNetwork(self.state_size, HIDDEN_SIZE, self.action_size, self.env_dim, self, verbose=verbose).to(device)
        self.critic = MetaCriticNetwork(self.state_size, HIDDEN_SIZE, 1, self.env_dim, self, verbose=verbose).to(device)

        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        
        if bert_embedding:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lambda steps: min((steps + 1) / WARMUP_STEPS, 1)
            )

        self.cov = torch.diag(torch.ones(self.action_size, ) * 0.5).to(device)

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
        self.cached_embedding = None
        self.use_cached_embedding = False

        # delete existing files
        if os.path.exists(self.data_path + "/metappo_trajectories.csv"):
            os.remove(self.data_path + "/metappo_trajectories.csv")

    # skip update for the policy and critic network, i.e., policy evaluation/serving stage
    def disable_update(self):
        self.skip_update = True

    # enable update for the policy and critic network, i.e., policy training stage
    def enable_update(self):
        self.skip_update = False

    # set the environment to interact with
    def set_env(self, env):
        self.env = env

    # get the action based on the state
    def calc_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        if FLAG_CONTINUOUS_ACTION:
            mean = self.actor(state)
            dist = torch.distributions.MultivariateNormal(mean, self.cov)
        else:
            action_probs = self.actor(state)
            dist = torch.distributions.Categorical(action_probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.cpu().detach().numpy(), log_prob.detach()

    # clear episode buffer
    def clear_episode_buffer(self):
        self.episode_buffer = DictList()  # allowing duplicate keys
        self.episode_buffer_rewards = []
        self.episode_buffer_index = 0
        heapq.heapify(self.episode_buffer_rewards)

    # update episode buffer
    def update_episode_buffer(self, states_ep, actions_ep, rewards_ep, steps_ep):
        assert len(states_ep) == steps_ep
        assert len(actions_ep) == steps_ep
        assert len(rewards_ep) == steps_ep

        if self.config['mode'] == 'best':
            # episode buffer contains the episodes with the highest rewards
            reward = round(np.sum(rewards_ep), 5)
            if len(self.episode_buffer_rewards) >= self.config['buffer_size']:
                # check if the new episode has a higher reward than the least episode reward in the buffer
                if reward >= self.episode_buffer_rewards[0] and (not reward in self.episode_buffer_rewards):
                    # remove the episode with the least reward
                    removed = heapq.heappop(self.episode_buffer_rewards)
                    if self.verbose:
                        print('Removing trajectory with reward:', removed)
                        print('After removing, trajectory buffer rewards:', self.episode_buffer_rewards)
                    # del self.episode_buffer[removed]
                    if isinstance(self.episode_buffer[removed], list):
                        del self.episode_buffer[removed][0]
                    else:
                        del self.episode_buffer[removed]
                else:
                    # ignore the new episode since its reward is smaller than any episode in the buffer
                    return

            # add the new episode and its reward to the buffer
            heapq.heappush(self.episode_buffer_rewards, reward)
            self.episode_buffer[reward] = {
                'states': states_ep,
                'actions': actions_ep,
                'rewards': rewards_ep
            }
            # cached embedding is not valid
            self.use_cached_embedding = False
        elif self.config['mode'] == 'latest':
            # episode buffer contains the lastest episodes
            if len(self.episode_buffer) >= self.config['buffer_size']:
                # remove the oldest episode
                oldest = self.episode_buffer_rewards[self.episode_buffer_index]
                del self.episode_buffer[oldest]
                self.episode_buffer[self.episode_buffer_index] = {
                    'states': states_ep,
                    'actions': actions_ep,
                    'rewards': rewards_ep
                }
                self.episode_buffer_index = (self.episode_buffer_index + 1) % self.config['buffer_size']
            else:
                # add the new episode to the buffer
                heapq.heappush(self.episode_buffer_rewards, self.episode_buffer_index)
                self.episode_buffer[self.episode_buffer_index] = {
                    'states': states_ep,
                    'actions': actions_ep,
                    'rewards': rewards_ep
                }
                self.episode_buffer_index = (self.episode_buffer_index + 1) % self.config['buffer_size']
            # cached embedding is not valid
            self.use_cached_embedding = False
        else:
            raise NotImplementedError

    # get episode buffer
    def get_episode_buffer(self):
        return [self.episode_buffer[r] for r in reversed(self.episode_buffer_rewards)]

    # model training
    def train(self, callback=None, mode='train'):
        # for RL learning curve plots
        iteration_rewards = []
        smoothed_rewards = []
        
        predction_value = []
         ##AL_METRICS
        cost_utility = 0
        #smooth
        prev_smoothed_delta_cost_ratio = None
        prev_smoothed_diff_sum_cost_ratio = None
        smooth_delta_cost_ratios = []  # List to store (delta / iter_training_cost) values
        smooth_diff_sum_cost_ratios = []
        window_size_al = 1
        #initialzie value
        iter_training_cost = 0
        current_uncertainty = 0
        last_uncertainty = 100
        #simulation choice
        # sample_size = 1152  # Fixed sample size of 100
        # dataset_size = len(X_train_data)
        apply_check = False
        simulation_epoch = 0
        iteration = 0
        window_diff = []
        window_delta = []
        
        pid = os.getpid()
        python_process = psutil.Process(pid)
        
        return_uncertainty = []

        if not self.skip_update:
            for iteration in range(TOTAL_ITERATIONS*10):
                
                if apply_check == True:
                    apply_check = False
                    simulation_epoch = 0
                    best_mape = 100
                    
                    #momentum
                    prev_smoothed_delta_cost_ratio = None
                    prev_smoothed_diff_sum_cost_ratio = None
                    smooth_delta_cost_ratios = []  # List to store (delta / iter_training_cost) values
                    smooth_diff_sum_cost_ratios = []
                    
                    #sample window
                    window_diff = []
                    window_delta = []
                    
                    
                    window_size = 12
                    #initialzie value
                    iter_training_cost = 0
                    current_uncertainty = 0
                    last_uncertainty = 100
                ##
                epoch_time_start = time.time()
                ##
                
                
                
                if self.verbose:
                    print('\n********** Iteration %i ************' % iteration)
                steps_per_iteration = 0

                # get resource consumption profiles
                memory_usage = python_process.memory_info()[0] / 2. ** 20  # memory use in MB
                cpu_util = python_process.cpu_percent(interval=None)/psutil.cpu_count()
                if self.verbose:
                    print('RL Agent Memory Usage:', memory_usage, 'MB', '| CPU Util:', str(cpu_util)+'%')

                states = []
                actions = []
                rewards = []
                log_probs = []
                for episode in range(EPISODES_PER_ITERATION):
                    state = self.env.reset(self.function_name)[:NUM_STATES]
                    episode_rewards = []

                    # roll out a trajectory (or so called episode)
                    done = False
                    steps_per_ep = 0
                    trajectory_action = []
                    trajectory_state = []
                    for step in range(EPISODE_LENGTH):
                        action, log_prob = self.calc_action(state)

                        action_to_execute = {
                            'vertical': 0,
                            'horizontal': 0
                        }
                        action = int(action)
                        action_to_execute['vertical'] = action

                        next_state, reward, done = self.env.step(self.function_name, action_to_execute)
                        next_state = next_state[:NUM_STATES]

                        states.append(state)
                        episode_rewards.append(reward)
                        log_probs.append(log_prob)
                        if FLAG_CONTINUOUS_ACTION:
                            actions.append(action)
                        else:
                            actions.append(action)

                        trajectory_state.append(state)
                        trajectory_action.append(action)

                        steps_per_ep += 1
                        steps_per_iteration += 1

                        # verbose
                        if self.verbose and episode % 5 == 0 and iteration % 50 == 0:
                            print_step_info(step, state, action_to_execute, reward)
                        
                        if done:
                            break

                        state = next_state

                        # evaluate the current policy at each checkpoint
                        if callback != None:
                            callback.on_step()

                    # end of one episode
                    if self.verbose:
                        print('End of an episode:', steps_per_ep, 'steps executed with', round(np.sum(episode_rewards), 5))
                    rewards.append(episode_rewards)

                    # update the episode buffer
                    self.update_episode_buffer(trajectory_state, trajectory_action, episode_rewards, steps_per_ep)

                # end of one iteration
                iteration_rewards.append(np.mean([np.sum(episode_rewards) for episode_rewards in rewards]))
                smoothed_rewards.append(np.mean(iteration_rewards[-10:]))

                # states = torch.FloatTensor(states)
                # states = states.reshape((-1,) + self.observation_space.shape)
                states = torch.FloatTensor(np.array(states)).to(self.device)
                if FLAG_CONTINUOUS_ACTION:
                    actions = torch.FloatTensor(actions).to(self.device)
                else:
                    actions = torch.IntTensor(actions).to(self.device)
                log_probs = torch.FloatTensor(log_probs).to(self.device)

                average_rewards = np.mean([np.sum(episode_rewards) for episode_rewards in rewards])
                print('Iteration:', iteration, '('+str(steps_per_iteration)+' steps) - Average rewards across episodes:', np.round(average_rewards, decimals=3),
                    '| Moving average:', np.round(np.mean(iteration_rewards[-10:]), decimals=3))

                # check if model update is skipped
                if self.skip_update:
                    continue

                if SAVE_TO_FILE:
                    # save the trajectories in this iteration to local
                    all_rewards = [reward for reward_ep in rewards for reward in reward_ep]
                    self.save_trajectories(iteration, states, actions, all_rewards)
                    if self.verbose:
                        print('Trajectory saved!')

                # update RL policy
                t_start = time.time()
                returns = calc_gae(rewards).to(self.device)

                values = self.critic(states).squeeze()
                ##prediction value
                predction_value = values.detach()
                
                advantage = returns - values.detach()
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                uncertainties_multiple_epochs = []
                for epoch in range(SGD_EPOCHS):
                    batch_size = states.size(0)  # whole batch size = num of steps
                    # use mini-batch instead of the whole batch
                    uncertainties = []
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
                        
                        ##uncertainties
                        uncertainties.append(critic_loss.item())
                        
                        if hasattr(self, 'scheduler'):
                            self.scheduler.step()
                    uncertainties_multiple_epochs.append(uncertainties)
                    # check if the model parameters are still being updated or converged
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
                                break
                        if is_equal:
                            self.num_same_parameter_critic += 1
                        else:
                            self.num_same_parameter_critic = 0
                            # self.parameter_critic = list(self.critic.parameters())
                            self.parameter_critic = []
                            for parameter in self.critic.parameters():
                                self.parameter_critic.append(parameter.clone())
                
                ##training finished
                uncertainties = np.mean(uncertainties_multiple_epochs, axis=0)
                return_uncertainty = uncertainties
                epoch_time_end = time.time() 
                print('end_time {}'.format(epoch_time_end)) 
                print('start_time {}'.format(epoch_time_start)) 
                iter_training_cost = get_time_cost(iter_training_cost,epoch_time_end,epoch_time_start)
                print('iter_training_cost {}'.format(iter_training_cost))  
                print('length of current_uncertainty {}'.format(len(uncertainties)))
                
                if iteration > 1:
                    percentile_95 = np.percentile(uncertainties, 95)
                    capped_uncertainties = [min(u, percentile_95) for u in uncertainties]
                    uncertainties = capped_uncertainties
                    current_uncertainty = sum(uncertainties)
                    ###TO DO:
                    #1. simulation uncertainty
                    
                    env_simulation = Environment_simulation(file_path,self.env.simulation_table)
                    function_name = env_simulation.get_function_name()
                    initial_state = env_simulation.reset(function_name)
                    mode = 'test'
                    base_dir = os.path.dirname(self.data_path)
                    sim_folder_path = os.path.join(base_dir, f'{function_name}_eval')
                    if not os.path.exists(sim_folder_path):
                        os.makedirs(sim_folder_path)
                    
                    # folder_path = os.path.dirname(checkpoint_path) + '/' + function_name + '_eval'
                    # # if not os.path.exists(folder_path):
                    # #     os.makedirs(folder_path)

                    # init an RL agent
                    agent_simulation = MetaPPO(env_simulation, function_name, sim_folder_path, mode, self.device, verbose=False, bert_embedding=False)
                    
                    agent_simulation.load_checkpoint(self.saved_checkpoint_path)

                    agent_simulation.disable_update()

                    useless,uncertainty_simulation,useless_2 = agent_simulation.train(mode=mode)
                    
                    data_uncertainty,data_cost,new_labeled,sample_scale = uncertainties_simulation(uncertainty_simulation,sample_size,iter_training_cost)
                    
                    result, prev_smoothed_delta_cost_ratio, prev_smoothed_diff_sum_cost_ratio = calculate_and_compare_metrics(
                        uncertainties,
                        last_uncertainty,
                        current_uncertainty,
                        dataset_size,
                        new_labeled,
                        iter_training_cost,
                        data_uncertainty,
                        data_cost,
                        sample_scale,
                        prev_smoothed_delta_cost_ratio,
                        prev_smoothed_diff_sum_cost_ratio,
                        smooth_delta_cost_ratios,
                        smooth_diff_sum_cost_ratios,
                        window_size_al,
                        simulation_epoch
                    )
                    simulation_epoch+=1
                    if result == False:
                        if self.env.unrevealed_table == {}:
                            break
                        apply_check = True
                        self.env.reveal_new_data()
                                    
                        
                    else:
                        last_uncertainty = current_uncertainty
                
                
                if self.verbose:
                    print("Time used for model update: {:.2f}s".format(time.time() - t_start))
                if self.num_same_parameter_critic > MAX_SAME_ITERATIONS and\
                        self.num_same_parameter_actor > MAX_SAME_ITERATIONS:
                    print('Model parameters are not updating! Turning to policy-serving stage...')
                    self.skip_update = True
                
                # save to checkpoint
                # Not saving to checkpoint when mode=='meta-train'
                if mode == 'train' and iteration==1:
                    self.save_checkpoint(iteration)
                if mode == 'train' and iteration > 0 and iteration % 100 == 0:
                    self.save_checkpoint(iteration)
        else:
            for iteration in range(TOTAL_TEST_ITERATIONS):
                if self.verbose:
                    print('\n********** Iteration %i ************' % iteration)
                steps_per_iteration = 0

                # get resource consumption profiles
                memory_usage = python_process.memory_info()[0] / 2. ** 20  # memory use in MB
                cpu_util = python_process.cpu_percent(interval=None)/psutil.cpu_count()
                if self.verbose:
                    print('RL Agent Memory Usage:', memory_usage, 'MB', '| CPU Util:', str(cpu_util)+'%')

                states = []
                actions = []
                rewards = []
                log_probs = []
                for episode in range(EPISODES_PER_ITERATION):
                    state = self.env.reset(self.function_name)[:NUM_STATES]
                    episode_rewards = []

                    # roll out a trajectory (or so called episode)
                    done = False
                    steps_per_ep = 0
                    trajectory_action = []
                    trajectory_state = []
                    for step in range(EPISODE_LENGTH):
                        action, log_prob = self.calc_action(state)

                        action_to_execute = {
                            'vertical': 0,
                            'horizontal': 0
                        }
                        action = int(action)
                        action_to_execute['vertical'] = action

                        next_state, reward, done = self.env.step(self.function_name, action_to_execute)
                        next_state = next_state[:NUM_STATES]

                        states.append(state)
                        episode_rewards.append(reward)
                        log_probs.append(log_prob)
                        if FLAG_CONTINUOUS_ACTION:
                            actions.append(action)
                        else:
                            actions.append(action)

                        trajectory_state.append(state)
                        trajectory_action.append(action)

                        steps_per_ep += 1
                        steps_per_iteration += 1

                        # verbose
                        if self.verbose and episode % 5 == 0 and iteration % 50 == 0:
                            print_step_info(step, state, action_to_execute, reward)
                        
                        if done:
                            break

                        state = next_state

                        # evaluate the current policy at each checkpoint
                        if callback != None:
                            callback.on_step()

                    # end of one episode
                    if self.verbose:
                        print('End of an episode:', steps_per_ep, 'steps executed with', round(np.sum(episode_rewards), 5))
                    rewards.append(episode_rewards)

                    # update the episode buffer
                    self.update_episode_buffer(trajectory_state, trajectory_action, episode_rewards, steps_per_ep)

                # end of one iteration
                iteration_rewards.append(np.mean([np.sum(episode_rewards) for episode_rewards in rewards]))
                smoothed_rewards.append(np.mean(iteration_rewards[-10:]))

                # states = torch.FloatTensor(states)
                # states = states.reshape((-1,) + self.observation_space.shape)
                states = torch.FloatTensor(np.array(states)).to(self.device)
                if FLAG_CONTINUOUS_ACTION:
                    actions = torch.FloatTensor(actions).to(self.device)
                else:
                    actions = torch.IntTensor(actions).to(self.device)
                log_probs = torch.FloatTensor(log_probs).to(self.device)

                average_rewards = np.mean([np.sum(episode_rewards) for episode_rewards in rewards])
                print('Iteration:', iteration, '('+str(steps_per_iteration)+' steps) - Average rewards across episodes:', np.round(average_rewards, decimals=3),
                    '| Moving average:', np.round(np.mean(iteration_rewards[-10:]), decimals=3))

                # check if model update is skipped
                if self.skip_update:
                    continue

                if SAVE_TO_FILE:
                    # save the trajectories in this iteration to local
                    all_rewards = [reward for reward_ep in rewards for reward in reward_ep]
                    self.save_trajectories(iteration, states, actions, all_rewards)
                    if self.verbose:
                        print('Trajectory saved!')

                # update RL policy
                t_start = time.time()
                returns = calc_gae(rewards).to(self.device)

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
                        
                        if hasattr(self, 'scheduler'):
                            self.scheduler.step()

                    # check if the model parameters are still being updated or converged
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
                                break
                        if is_equal:
                            self.num_same_parameter_critic += 1
                        else:
                            self.num_same_parameter_critic = 0
                            # self.parameter_critic = list(self.critic.parameters())
                            self.parameter_critic = []
                            for parameter in self.critic.parameters():
                                self.parameter_critic.append(parameter.clone())
                
                if self.verbose:
                    print("Time used for model update: {:.2f}s".format(time.time() - t_start))
                if self.num_same_parameter_critic > MAX_SAME_ITERATIONS and\
                        self.num_same_parameter_actor > MAX_SAME_ITERATIONS:
                    print('Model parameters are not updating! Turning to policy-serving stage...')
                    self.skip_update = True
                
                # save to checkpoint
                # Not saving to checkpoint when mode=='meta-train'
                if mode == 'train' and iteration > 0 and iteration % 100 == 0:
                    self.save_checkpoint(iteration)
            
            
            
            
            
        # Save the final checkpoint if in training mode. Note that in adaptation mode, we do not 
        # want to rewrite the pretrained model using the fine-tuned one.
        # Not saving to checkpoint when mode=='meta-train'.
        if mode == 'train':
            self.save_checkpoint(iteration+1)

        # plot
        if PLOT_FIG:
            visualization(iteration_rewards, smoothed_rewards, self.data_path, testing=self.skip_update)

        # write iteration rewards to file
        if SAVE_TO_FILE:
            postfix = ""
            if self.skip_update:
                postfix = "_testing"
            file = open(self.data_path + "/metappo_smoothed_rewards" + postfix + ".txt", "w")
            for reward in smoothed_rewards:
                file.write(str(reward) + "\n")
            file.close()
            file = open(self.data_path + "/metappo_iteration_rewards" + postfix + ".txt", "w")
            for reward in iteration_rewards:
                file.write(str(reward) + "\n")
            file.close()
        return np.mean(iteration_rewards),return_uncertainty,predction_value


    # load all model parameters from a saved checkpoint
    def load_checkpoint(self, checkpoint_file_path):
        if os.path.isfile(checkpoint_file_path):
            if self.verbose:
                print('Loading checkpoint...')
            checkpoint = torch.load(checkpoint_file_path)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.verbose:
                print('Checkpoint successfully loaded!')
        else:
            raise OSError('Checkpoint not found!')

    # save all model parameters to a checkpoint
    def save_checkpoint(self, episode_num):
        if self.bert_embedding:
            model_type = 'berttiny' if FLAG_BERT_TINY else 'bert'
        else:
            model_type = 'rnn'
        checkpoint_name = self.data_path + '/metappo-' + model_type + '-ep{}.pth.tar'.format(episode_num)
        checkpoint = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

        torch.save(checkpoint, checkpoint_name)
        self.saved_checkpoint_path = checkpoint_name

    def save_model_to_path(self, model_path):
        checkpoint = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

        torch.save(checkpoint, model_path)
        
    # record trajectories
    def save_trajectories(self, episode_num, states, actions, rewards):
        file = open(self.data_path + "/metappo_trajectories.csv", "a")
        count = 0
        for state in states:
            file.write(str(episode_num) + ',' + ','.join([str(item.item()) for item in state]) + ',' + str(actions[count].item()) + ',' + str(rewards[count]) + "\n")
            count += 1
        file.close()

    # predict the action given the observation
    def predict(self, observation):
        # observation = np.array(observation)
        # observation = observation.reshape((-1,) + self.env.observation_space.shape)
        action, _ = self.calc_action(observation)

        # clip the action to avoid out of bound error
        if isinstance(self.env.action_space, gym.spaces.Box):
            clipped_action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

        return clipped_action


if __name__ == '__main__':
    test = DictList()
    test[0] = 10
    test[1] = 20
    test[1] = 40
    test[1] = 50
    test[2] = 10
    print(test)
    del test[1][0]
    print(test)
    del test[1]
    print(test)

