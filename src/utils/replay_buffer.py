import collections
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            # Not enough samples yet, or indicate this situation appropriately
            return [] # Or raise an error, or return None
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)