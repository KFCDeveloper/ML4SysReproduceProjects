import pandas as pd
import random
from util import *


UNSTABLE_WAITING_TIME_UPPER = 900000  # timeout value in serverless platforms, i.e., 15min
UNSTABLE_WAITING_TIME_LOWER = 60000
DEFAULT_MEMORY_LIMIT = 128
ILLEGAL_PENALTY = 100
DEFAULT_NUM_CORES_PER_SERVER = 8
SLO_LATENCY = 9000.0
DATA_PATH = 'tabular_data_single_rl.csv'


class RLEnvironment:
    # initial states
    initial_arrival_rate = 6.0
    initial_memory_limit = 128
    initial_cpu_shares = 1024
    initial_cpu_shares_others = 0

    # registered functions
    functions = []

    # tabular data
    table = {}

    # static
    num_cores_per_server = DEFAULT_NUM_CORES_PER_SERVER

    last_action = {
        'vertical': 0,
        'horizontal': 0
    }
    last_reward = 0

    def __init__(self, function_name='example_app'):
        # add one example function
        self.original_service_rate = 1.5
        self.original_cpu_util = 0.6
        self.slo_latency = SLO_LATENCY
        self.functions.append(function_name)
        self.arrival_rate = self.initial_arrival_rate
        self.cpu_shares_others = self.initial_cpu_shares_others
        self.cpu_shares_per_container = self.initial_cpu_shares
        self.num_containers = 0

    # load all data from traces
    def load_data(self):
        df = pd.read_csv(DATA_PATH)
        for index, row in df.iterrows():
            tabular_item = {
                'avg_cpu_util': row['avg_cpu_util'],
                'slo_preservation': row['slo_preservation'],
                'total_cpu_shares': row['total_cpu_shares'],
                'cpu_shares_others': row['cpu_shares_others'],
                'latency': row['latency']
            }
            key = (row['num_containers'], row['arrival_rate'])
            self.table[key] = tabular_item

    # list all registered functions
    def get_functions(self):
        return self.functions

    # return the states
    # [cpu util percentage, slo preservation percentage, cpu.shares, cpu.shares (others), # of containers, arrival rate]
    def get_rl_states(self, num_containers, arrival_rate):
        if num_containers > MAX_NUM_CONTAINERS:
            return None
        value = self.table[(num_containers, arrival_rate)]
        return [value['avg_cpu_util'], value['slo_preservation'], value['total_cpu_shares']/20480.0,
                value['cpu_shares_others']/92160.0, num_containers/20.0, arrival_rate/10.0, value['latency']]

    # overprovision to num of concurrent containers + 2
    def overprovision(self, function_name):
        # horizontal scaling
        scale_action = {
            'vertical': 0,
            'horizontal': int(self.initial_arrival_rate) + 2
        }
        self.num_containers += int(self.initial_arrival_rate) + 2
        states, _, _ = self.step(function_name, scale_action)
        print('Overprovisioning:',
              '[ Avg CPU utilization: {:.7f} Avg SLO preservation: {:.7f}'.format(states[0], states[1]),
              'Num of containers:', states[4],
              'CPU shares:', states[2], 'CPU shares (others):', states[3],
              'Arrival rate:', states[5], ']')

    # reset the environment by re-initializing all states and do the overprovisioning
    def reset(self, function_name):
        if function_name not in self.functions:
            return KeyError

        self.arrival_rate = self.initial_arrival_rate
        self.cpu_shares_others = self.initial_cpu_shares_others
        self.num_containers = 0
        # randomly set the cpu shares for all other containers on the same server
        cpu_shares_other = random.randint(1, 9) * 1024
        cpu_shares_other = 0
        self.cpu_shares_others = cpu_shares_other

        # randomly set the arrival rate
        arrival_rate = random.choice(range(1, 9))
        # arrival_rate = 6
        self.arrival_rate = arrival_rate

        # overprovision resources to the function initially
        # self.overprovision(function_name)

        return self.get_rl_states(self.num_containers, self.arrival_rate)

    # step function to update the environment given the input actions
    # action: +/- cpu.shares for all function containers; +/- number of function containers with the same cpu.shares
    # states: [cpu util percentage, slo preservation percentage, cpu.shares, cpu.shares (others), # of containers,
    #          arrival rate]
    def step(self, function_name, action):
        if function_name not in self.functions:
            raise KeyError

        curr_state = self.get_rl_states(self.num_containers, self.arrival_rate)

        # action correctness check:
        # - scaling in when there's no containers
        # - scaling up/down when there's no containers
        # - scaling down leading to cpu.shares < 128
        if self.num_containers <= 0:
            if action['vertical'] != 0 or action['horizontal'] < 0:
                self.last_reward = -1
                # return curr_state, -ILLEGAL_PENALTY, False
                return curr_state, -1, False
        else:
            if action['vertical'] + self.cpu_shares_per_container < 128:
                self.last_reward = -1
                # return curr_state, -ILLEGAL_PENALTY, False
                return curr_state, -1, False
        if self.num_containers + action['horizontal'] > MAX_NUM_CONTAINERS:
            self.last_reward = -1
            return curr_state, -1, False

        # perform the action
        if action['vertical'] != 0:
            # vertical scaling of the cpu.shares
            self.cpu_shares_per_container += action['vertical']
        elif action['horizontal'] > 0:
            # scaling out with the existing cpu and memory allocation
            if self.cpu_shares_per_container == -1:
                self.cpu_shares_per_container = self.initial_cpu_shares
            self.num_containers += action['horizontal']
        elif action['horizontal'] < 0:
            # scaling in
            self.num_containers += action['horizontal']
        elif action['scale_to'] > 0:
            # scale to the target number of containers
            if self.cpu_shares_per_container == -1:
                self.cpu_shares_per_container = self.initial_cpu_shares
            self.num_containers = action['scale_to']
        else:
            # no action to perform
            pass

        state = self.get_rl_states(self.num_containers, self.arrival_rate)

        # calculate the reward
        reward = convert_state_action_to_reward(state, action, self.last_action, self.arrival_rate)

        self.last_reward = reward
        self.last_action = action

        # check if done
        done = False

        return state, reward, done

    def reset_arrival_rate(self, function_name, arrival_rate):
        if function_name not in self.functions:
            return KeyError
        self.arrival_rate = arrival_rate
        state = self.get_rl_states(self.num_containers, self.arrival_rate)

        return state


if __name__ == '__main__':
    print('Testing simulated environment...')
    env = RLEnvironment()
    env.load_data()
