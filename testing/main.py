import os
import random
import psutil

from rl_env import RLEnvironment
from ddpg import ddpg
from pg import pg
from ppo.ppo_updated import PPO  # with mini-batch implemented
from ppo.meta_ppo import MetaPPO  # with meta-learning
from ppo.ppo_updated_no_mini_batch import PPOWithoutMiniBatch  # without mini-batch implemented
from dqn import dqn


def test_env(env, function_name):
    env.reset(function_name)
    env.platform.print_function_info()

    env.platform.print_registered_functions()

    # horizontal scaling tests
    test_action = {
        'vertical': 0,
        'horizontal': 2
    }
    states, _, _ = env.step(function_name, test_action, env.initial_memory_limit)
    print('States:', states)
    env.platform.print_function_info()

    # horizontal scaling tests
    test_action = {
        'vertical': 0,
        'horizontal': 1
    }
    states, _, _ = env.step(function_name, test_action, env.initial_memory_limit)
    print('States:', states)
    env.platform.print_function_info()

    # vertical scaling tests
    test_action = {
        'vertical': 128,
        'horizontal': 0
    }
    states, _, _ = env.step(function_name, test_action, env.initial_memory_limit)
    print('States:', states)
    env.platform.print_function_info()

    # vertical scaling tests
    test_action = {
        'vertical': -128,
        'horizontal': 0
    }
    states, _, _ = env.step(function_name, test_action, env.initial_memory_limit)
    print('States:', states)
    env.platform.print_function_info()

def main():
    function_name = 'example_app'
    env = RLEnvironment(function_name)
    env.load_data()

    # init an RL agent
    agent_type = 'PPO_Meta'  # 'PPO', 'PPO_Meta'
    agent = None
    if agent_type == 'PPO':
        agent = PPO(env, function_name)
    elif agent_type == 'PPO_Meta':
        agent = MetaPPO(env, function_name)
    elif agent_type == 'PPO_NoMiniBatch':
        agent = PPOWithoutMiniBatch(env, function_name)
    if agent_type == 'DDPG':
        agent = ddpg.DDPG(env, function_name)
    elif agent_type == 'PG':
        agent = pg.PG(env, function_name)
    elif agent_type == 'DQN':
        agent = dqn.DQN(env, function_name)

    # init from saved checkpoints (e.g., offline bootstrapping or incremental retraining)
    use_checkpoint = False
    checkpoint_file = './checkpoints/final.pth.tar'
    if use_checkpoint:
        agent.load_checkpoint(checkpoint_file)

    # start training
    agent.train()


if __name__ == "__main__":
    main()
