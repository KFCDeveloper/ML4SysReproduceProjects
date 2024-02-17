import numpy as np
import threading
import asyncio


NUM_STATES = 6
NUM_GLOBAL_STATES = 4
NUM_GLOBAL_STATES_WITH_VARIANCE = NUM_GLOBAL_STATES * 2
NUM_MEAN_FIELD_STATES = NUM_STATES
NUM_MEAN_FIELD_STATES_WITH_ACTIONS = NUM_STATES + 2  # 2 for horizontal and vertical actions
NUM_ACTIONS = 5
NUM_TOTAL_ACTIONS = 10
VERTICAL_SCALING_STEP = 128
HORIZONTAL_SCALING_STEP = 1
MAX_NUM_CONTAINERS = 10.0
MAX_CPU_SHARES = 2048.0


# print current state
def print_state(state_list):
    print('Avg CPU util: {:.7f} Avg SLO preservation: {:.7f}'.format(state_list[0], state_list[1]),
          'Num of containers:', state_list[4],
          'CPU shares:', state_list[2], 'CPU shares (others):', state_list[3],
          'Arrival rate:', state_list[4])


# print current action
def print_action(action_dict):
    if action_dict['vertical'] > 0:
        print('Action: Scaling-up by', VERTICAL_SCALING_STEP, 'cpu.shares')
    elif action_dict['vertical'] < 0:
        print('Action: Scaling-down by', VERTICAL_SCALING_STEP, 'cpu.shares')
    elif action_dict['horizontal'] > 0:
        print('Action: Scaling-out by', HORIZONTAL_SCALING_STEP, 'container')
    elif action_dict['horizontal'] < 0:
        print('Action: Scaling-in by', HORIZONTAL_SCALING_STEP, 'container')
    elif action_dict['scale_to'] != -1:
        print('Action: Scaling to', action_dict['scale_to'], ' containers')
    else:
        print('No action to perform')


# print (state, action, reward) for the current step
def print_step_info(step, state_list, action_dict, reward):
    state = 'State: [Avg CPU utilization: {:.7f} Avg SLO preservation: {:.7f}'.format(state_list[0], state_list[1]) +\
            ' Num of containers: ' + str(state_list[4]) +\
            ' CPU shares: ' + str(state_list[2]) + ' CPU shares (others): ' + str(state_list[3]) +\
            ' Arrival rate: ' + str(state_list[5]) + ']'
    action = 'Action: N/A'
    if action_dict['vertical'] > 0:
        action = 'Action: Scaling-up by ' + str(VERTICAL_SCALING_STEP) + ' cpu.shares'
    elif action_dict['vertical'] < 0:
        action = 'Action: Scaling-down by ' + str(VERTICAL_SCALING_STEP) + ' cpu.shares'
    elif action_dict['horizontal'] > 0:
        action = 'Action: Scaling-out by ' + str(HORIZONTAL_SCALING_STEP) + ' container'
    elif action_dict['horizontal'] < 0:
        action = 'Action: Scaling-in by ' + str(HORIZONTAL_SCALING_STEP) + ' container'
    elif action_dict['scale_to'] != -1:
        action = 'Action: Scale to ' + str(action_dict['scale_to']) + ' containers'
    print('Step #' + str(step), '|', state, '|', action, '| Reward:', reward)


def print_step_info_with_function_name(step, state_list, action_dict, reward, function_name):
    state = 'State: [Avg CPU utilization: {:.7f} Avg SLO preservation: {:.7f}'.format(state_list[0], state_list[1]) +\
            ' Num of containers: ' + str(state_list[4]) +\
            ' CPU shares: ' + str(state_list[2]) + ' CPU shares (others): ' + str(state_list[3]) +\
            ' Arrival rate: ' + str(state_list[5]) + ']'
    action = 'Action: N/A'
    if action_dict['vertical'] > 0:
        action = 'Action: Scaling-up by ' + str(VERTICAL_SCALING_STEP) + ' cpu.shares'
    elif action_dict['vertical'] < 0:
        action = 'Action: Scaling-down by ' + str(VERTICAL_SCALING_STEP) + ' cpu.shares'
    elif action_dict['horizontal'] > 0:
        action = 'Action: Scaling-out by ' + str(HORIZONTAL_SCALING_STEP) + ' container'
    elif action_dict['horizontal'] < 0:
        action = 'Action: Scaling-in by ' + str(HORIZONTAL_SCALING_STEP) + ' container'
    elif action_dict['scale_to'] != -1:
        action = 'Action: Scale to ' + str(action_dict['scale_to']) + ' containers'
    print('[', function_name, '] - Step #' + str(step), '|', state, '|', action, '| Reward:', reward)


# calculate the reward based on the current states (after the execution of the current action)
# + cpu utilization percentage [0, 1]
# + slo preservation [0, 1]
# - number of containers (/ arrival rate)
# - penalty
def convert_state_action_to_reward(state, action, last_action, arrival_rate):
    reward_less_containers = np.exp(-state[4])
    # if state[4] > self.arrival_rate:
    #     reward_less_containers = np.exp(self.arrival_rate - state[4])
    # else:
    #     reward_less_containers = np.exp(state[4] - self.arrival_rate)
    alpha = 0.3  # 0.3  - cpu-util
    beta = 0.7  # 0.7  - slo-preservation
    gamma = 10
    reward = alpha * state[0] + beta * state[1] + (1 - alpha - beta) * gamma * reward_less_containers

    # give penalty to illegal or undesired actions
    if arrival_rate > 0 and state[4] == 0:
        # give a large penalty if there's no container created when arriving rate > 0
        reward = -1  # -ILLEGAL_PENALTY

    # give penalty to frequent dangling actions: e.g., scale in and out
    if last_action['horizontal'] * action['horizontal'] < 0:
        reward = -1  # -ILLEGAL_PENALTY

    return reward


# check the correctness of the action
def is_correct(action, num_containers, cpu_shares_per_container):
    # - scaling in when there's no containers
    # - scaling up/down when there's no containers
    # - scaling down leading to cpu.shares < 128
    if num_containers <= 0:
        if action['vertical'] != 0 or action['horizontal'] < 0:
            return False
    else:
        if action['vertical'] + cpu_shares_per_container < 128:
            return False

    # check for maximum number of containers or cpu.shares per container
    if num_containers + action['horizontal'] > MAX_NUM_CONTAINERS:
        return False
    if action['vertical'] + cpu_shares_per_container > MAX_CPU_SHARES:
        return False

    # check the scale_to action
    if action['scale_to'] > MAX_NUM_CONTAINERS or action['scale_to'] < -1:
        return False

    return True


class AsyncioEventLoopThread(threading.Thread):
    def __init__(self, *args, loop=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loop = loop or asyncio.new_event_loop()
        self.running = False
        self.future = None

    def run(self):
        self.running = True
        self.loop.run_forever()

    def run_coro(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, loop=self.loop).result()
        # self.future = asyncio.run_coroutine_threadsafe(coro, loop=self.loop)
        # return self.future

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.join()
        self.running = False


# thread-safe event
class EventThreadSafe(asyncio.Event):
    def set(self):
        # FIXME: The _loop attribute is not documented as public api!
        self._loop.call_soon_threadsafe(super().set)

    def clear(self):
        self._loop.call_soon_threadsafe(super().clear)

    def is_set(self):
        self._loop.call_soon_threadsafe(super().is_set)


# count the number of parameters in a model
def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
