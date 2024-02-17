import os
import argparse

from rl_env import RLEnvironment
from ppo import PPO
from meta_ppo import MetaPPO


def main():
    """
    This is the main function for RL training and inference.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--app_name', type=str, default='hamster')  # name of the app to control
    parser.add_argument('--app_namespace', type=str, default='default')  # namespace of the app
    parser.add_argument('--mpa_name', type=str, default='hamster-mpa')  # name of the mpa object
    parser.add_argument('--mpa_namespace', type=str, default='default')  # namespace of the mpa
    parser.add_argument('--inference', action='store_true')  # True for skipping RL training, default False
    parser.add_argument('--use_checkpoint', action='store_true')  # True for loading from a model checkpoint, default False
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/ppo-ep1000.pth.tar')  # path to the checkpoint file
    parser.add_argument('--bootstrapping', action='store_true')  # True for bootstrapping RL training (offline), default False
    parser.add_argument('--meta', action='store_true')  # True for meta-learning, default False
    parser.set_defaults(inference=False)
    parser.set_defaults(use_checkpoint=False)
    options = parser.parse_args()

    # create and initialize the environment for rl training
    print('Initializing environment for app', options.app_name, '('+options.app_namespace+')')
    env = RLEnvironment(app_name=options.app_name, app_namespace=options.app_namespace, mpa_name=options.mpa_name, mpa_namespace=options.mpa_namespace)
    print('Initial state:')
    env.print_info()

    # init an RL agent
    if options.meta:
        agent = MetaPPO(env)
        print('RL agent initialized (with meta-learning)!')
    else:
        agent = PPO(env)
        print('RL agent initialized!')

    # init from saved model checkpoints
    if options.use_checkpoint:
        print('Loading model checkpoint from', options.checkpoint)
        if os.path.exists(options.checkpoint):
            agent.load_checkpoint(options.checkpoint)
        else:
            print('Checkpoint does not exist!')
            exit()

    # check if to skip RL training
    if not options.inference:
        print('Start RL policy training...')
    else:
        print('Start RL policy serving...')
        agent.disable_update()

    # start RL training/inference
    agent.train()


if __name__ == "__main__":
    main()
