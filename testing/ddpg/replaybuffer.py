import random
import warnings
from collections import deque


class Buffer:
    def __init__(self, buffer_size):
        self.limit = buffer_size
        self.data = deque(maxlen=self.limit)

    def __len__(self):
        return len(self.data)

    def sample_batch(self, batch_size):
        if len(self.data) < batch_size:
            warnings.warn('Not enough entries in the buffer to sample without replacement.')
            return None
        else:
            batch = random.sample(self.data, batch_size)
            curr_state = [element[0] for element in batch]
            action = [element[1] for element in batch]
            next_state = [element[2] for element in batch]
            reward = [element[3] for element in batch]
            terminal = [element[4] for element in batch]
        return curr_state, action, next_state, reward, terminal

    def append(self, element):
        self.data.append(element)
