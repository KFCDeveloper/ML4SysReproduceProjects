import pandas as pd
import random
from util import *
from AL_module import *

class SimEnvironment:
    # initial states
    initial_arrival_rate = 1
    initial_cpu_shares = 2048
    initial_cpu_shares_others = 0
    memory = 256
    cpu_dictionary = {0:128, 1:256, 2:512, 3:1024, 4:1536, 5:2048}
    # tabular data
    table = {}

    last_action = {
        'vertical': 0,  # last vertical action
        'horizontal': 0,  # last horizontal action
        'last_vertical': 0,  # the vertical action before last action
        'last_horizontal': 0  # the horizontal action before last action
    }
    last_reward = 0

    def __init__(self, input_path, pool=None):
        # add one example function
        ##AL used for controning data size
        ##data labeled
        self.visible_table = {}  
        self.revealed_fraction = 0.1
        ##data unlabeled
        self.unrevealed_table = {}
        ##data used for simulation
        self.simulation_table = {}
        ##data used for training
        self.training_table = {}
        ##based one the function name search for cost
        self.function_names = {}
        self.key_to_function_map = {}
        
        self.original_cpu_util = 0.6
        self.function = 'example_function'
        self.arrival_rate = self.initial_arrival_rate
        self.cpu_shares_others = self.initial_cpu_shares_others
        self.cpu_shares_per_container = self.initial_cpu_shares
        self.num_containers = 1
        self.current_state = None
        self.pool = pool  # a list of csv files for a pool of applications
        if pool:
            self.data_path = input_path + pool[0]
            self.load_data()
        else:
            self.data_path = input_path
            self.load_data()
        self.reveal_initial_data()
    
    ##select a subset of data to reveal
    def reveal_initial_data(self):
        num_initial_entries = max(1, int(len(self.table) * self.revealed_fraction))
        self.visible_table = dict(random.sample(self.table.items(), num_initial_entries))
        self.unrevealed_table = {k: v for k, v in self.table.items() if k not in self.visible_table}
        
        num_simulation_entries = max(1, int(len(self.visible_table) * 0.1))
        self.simulation_table = dict(random.sample(self.visible_table.items(), num_simulation_entries))
        self.training_table = {k: v for k, v in self.visible_table.items() if k not in self.simulation_table}
        print(self.training_table)
        print(self.simulation_table)
        print(self.visible_table)
        print(self.unrevealed_table)
        print(self.table)
        
        valid_cpu, valid_memory = self.get_valid_action(self.cpu_shares_per_container)
        print("valid_cpu, valid_memory")
        print(valid_cpu, valid_memory)
        self.initial_cpu_shares = valid_cpu
        self.initial_memory = valid_memory
        self.memory = self.initial_memory

        
        
    def get_valid_action(self, cpu_value):
        """
        Find the closest valid CPU configuration that exists in training data
        """
        valid_cpus = sorted(set(key[0] for key in self.training_table.keys()))
        valid_memories = sorted(set(key[1] for key in self.training_table.keys()))
        
        # Find closest valid CPU value
        closest_cpu = min(valid_cpus, key=lambda x: abs(x - cpu_value))
        # Find memory value that exists with this CPU configuration
        valid_memories_for_cpu = [key[1] for key in self.training_table.keys() if key[0] == closest_cpu]
        memory = valid_memories_for_cpu[0] if valid_memories_for_cpu else valid_memories[0]
        
        return closest_cpu, memory    
    ##add more data to the training set
    def reveal_new_data(self):
                
        ##1. simulation uncertainty
        env_simulation = Environment_simulation(self.data_path,self.unrevealed_table)
        function_name = env_simulation.get_function_name()
        initial_state = env_simulation.reset(function_name)
        mode = 'test'
        if verbose:
            print('Environment initialized for function', function_name)
            print('Initial state:', initial_state)
        
        folder_path = os.path.dirname(checkpoint_path) + '/' + function_name + '_eval'
            # if not os.path.exists(folder_path):
            #     os.makedirs(folder_path)

        # init an RL agent
        agent_simulation = MetaPPO(env_simulation, function_name, folder_path, mode, device, verbose=verbose, bert_embedding=use_bert)
        checkpoint_file = checkpoint_path
        agent_simulation.load_checkpoint(checkpoint_file)

        agent_simulation.disable_update()
        useless,useless_2,prediction_value = agent_simulation.train(mode=mode)
        
        number_to_select,new_files,cost_utility=active_learning_iteration_RL(table,self.cost_utility,self.iteration,prediction_value, self.unrevealed_table,env)
        for key in new_files:
            if key in self.unrevealed_table:
                # Move from unrevealed_table to visible_table
                self.visible_table[key] = self.unrevealed_table[key]
                # Remove from unrevealed_table
                del self.unrevealed_table[key]
        self.cost_utility = cost_utility
        num_simulation_entries = max(1, int(len(self.visible_table) * 0.1))
        self.simulation_table = dict(random.sample(self.visible_table.items(), num_simulation_entries))
        self.training_table = {k: v for k, v in self.visible_table.items() if k not in self.simulation_table}
        
        
        
        # num_new_entries = max(1, int(len(self.unrevealed_table) * self.revealed_fraction))
        # new_entries = dict(random.sample(self.unrevealed_table.items(), num_new_entries))
        # self.visible_table.update(new_entries)
        # self.unrevealed_table = {k: v for k, v in self.unrevealed_table.items() if k not in new_entries}

    # load all data from traces
    def load_data(self):
        df = pd.read_csv(self.data_path)
        for index, row in df.iterrows():
            tabular_item = {
                'avg_cpu_util': row['avg_cpu_util'],
                'slo_preservation': row['slo_preservation'],
                'total_cpu_shares': row['total_cpu_shares'],
                'cpu_shares_others': row['cpu_shares_others'],
                'num_containers':row['num_containers'],
                'arrival_rate':row['arrival_rate'],
                'latency': row['latency']
            }
            key = (row['cpu'], row['memory'])
            self.table[key] = tabular_item
        self.max_value = df.max().to_dict()
        # for the purpose of sanity check
        # for cpu in self.cpu_dictionary.values():
        #     print('CPU =', cpu)
        #     state = self.get_rl_states(cpu, 256)
        #     print('States:', state)
        #     reward = convert_state_action_to_reward(state, None, self.last_action, None)
        #     print('Reward =', reward, 'CPU util =', state[0], 'SLO preservation =', state[1])
        # exit()
        ##save file name
        function_name = self.data_path.split('/')[-1].split('.')[0] if self.data_path else "default_function"
        print(f"Function name: {function_name}")
        self.key_to_function_map[key] = function_name
        
        random_key,random_value = random.choice(list(self.table.items()))
        if self.pool:
            self.function = 'pool'
        else:
            self.function = self.data_path.split('/')[-1][:-11]
        init_cpu = random_key[0]
        init_cpu_util = random_value['avg_cpu_util']
        self.original_cpu_util = init_cpu_util
        self.cpu_shares_per_container = init_cpu

    # get the function name
    def get_function_name(self):
        return self.function
    
    def get_function_for_key(self, key):
        return self.key_to_function_map.get(key, "unknown_function")

    # return the states
    # [cpu util percentage, slo preservation percentage, cpu.shares, cpu.shares (others), # of containers, arrival rate]
    def get_rl_states(self, cpu, memory):
        # if num_containers > MAX_NUM_CONTAINERS:
        #     return None
        value = self.training_table[(cpu, memory)]
        max_cpu_shares_others = self.max_value['cpu_shares_others']
        max_total_cpu_shares = self.max_value['total_cpu_shares']
        max_num_containers = self.max_value['num_containers']
        max_arrival_rate  = self.max_value['arrival_rate']
        max_latency = self.max_value['latency']
        return [value['avg_cpu_util'], value['slo_preservation'], value['total_cpu_shares']/max_total_cpu_shares,
                value['cpu_shares_others'], value['num_containers']/max_num_containers, value['arrival_rate']/max_arrival_rate, value['latency']/max_latency]

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
        if function_name != self.function:
            return KeyError

        self.arrival_rate = self.initial_arrival_rate
        self.cpu_shares_others = self.initial_cpu_shares_others
        self.num_containers = 1
        # randomly set the cpu shares for all other containers on the same server
        cpu_shares_other = 0  # single-tenant
        self.cpu_shares_others = cpu_shares_other

        # randomly set the arrival rate
        arrival_rate = 1
        self.arrival_rate = arrival_rate

        # overprovision resources to the function initially
        # self.overprovision(function_name)
        self.cpu_shares_per_container = self.initial_cpu_shares
        scale_action = {
            'vertical': random.randint(0, 5),
            'horizontal': 0
        }
        # self.num_containers += 2

        self.cpu_shares_per_container, self.memory = self.get_valid_action(self.cpu_shares_per_container)


        states, _, _ = self.step(function_name, scale_action)

        self.current_state = self.get_rl_states(self.cpu_shares_per_container, self.memory)

        return self.current_state

    # step function to update the environment given the input actions
    # action: +/- cpu.shares for all function containers; +/- number of function containers with the same cpu.shares
    # states: [cpu util percentage, slo preservation percentage, cpu.shares, cpu.shares (others), # of containers,
    #          arrival rate]
    def step(self, function_name, action):
        if function_name != self.function:
            raise KeyError
        
        curr_state = self.get_rl_states(self.cpu_shares_per_container, self.memory)
        self.cpu_shares_per_container = self.cpu_dictionary[action['vertical']]
        
        
        self.cpu_shares_per_container, self.memory = self.get_valid_action(self.cpu_shares_per_container)
        state = self.get_rl_states(self.cpu_shares_per_container, self.memory)
        # calculate the reward
        reward = convert_state_action_to_reward(state, action, self.last_action, self.arrival_rate)
        self.last_reward = reward
        # self.last_action = action
        self.last_action['last_vertical'] = self.last_action['vertical']
        self.last_action['last_horizontal'] = self.last_action['horizontal']
        self.last_action['vertical'] = action['vertical']
        self.last_action['horizontal'] = action['horizontal']
        # check if done
        done = False
        self.current_state = state

        return state, reward, done

    def reset_arrival_rate(self, function_name, arrival_rate):
        if function_name != self.function:
            return KeyError
        self.arrival_rate = arrival_rate
        state = self.get_rl_states(self.num_containers, self.arrival_rate)

        return state

    # print function state information
    def print_info(self):
        print('Function name:', self.function)
        print('Average CPU Util:', self.current_state[0])
        print('SLO Preservation:', self.current_state[1])
        print('Total CPU Shares (normalized):', self.current_state[2])
        print('Total CPU Shares for Other Containers (normalized):', self.current_state[3])
        print('Number of Containers:', self.current_state[4] * 20)
        print('Arrival Rate (rps):', self.current_state[5] * 10)


if __name__ == '__main__':
    print('Testing simulated environment...')
    env = SimEnvironment()
    env.load_data()
