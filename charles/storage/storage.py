import random
import torch
import numpy as np
from collections import deque

class Storage:
    def __init__(self, config=None):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        try:
            self.buffer = deque(maxlen=int(config.storage_size))
        except:
            self.buffer = deque()
        self.config = config

    def store(self, data):
        '''stored a single group of data'''

        # make all data 1-dimensional numpy arrays, changing booleans to their corresponding mask value
        def fix(x):
            if isinstance(x, np.bool_): x = 1 - x
            if isinstance(x, torch.Tensor): x = np.array(x.cpu())
            if len(x.shape) == 0: x = np.expand_dims(x, axis=0)
            return x

        # add transitions to the buffer from each agent separately
        if len(data[0].shape) > 1: num_agents = data[0].shape[0]
        else: num_agents = 1
        for agent in range(num_agents):
            transition = tuple(fix(x[agent]) for x in data)
            self.buffer.append(transition)

    def get(self, source):
        '''return all data from the given source'''

        # group together all data of the same type
        n = len(self.buffer[0])
        data = [torch.FloatTensor(np.array([arr[i] for arr in source])).to(self.device) for i in range(n)]

        # expend data dimensions until they all have the same number of dimensions
        max_dim = max([len(d.shape) for d in data])
        for i in range(len(data)):
            while len(data[i].shape) < max_dim:
                data[i].unsqueeze_(2)
        return data

    def get_all(self):
        '''return all stored data'''
        return self.get(self.buffer)

    def sample(self, batch_size=None):
        '''return a random sample from the stored data'''
        if batch_size is None: batch_size = self.config.batch_size
        batch_size = min(len(self.buffer), batch_size)
        batch = random.sample(self.buffer, batch_size)
        return self.get(batch)

    def get_batches(self):
        # N = len(self.buffer)
        # batch_size = min(N, self.config.batch_size)
        # num_batches = len(self.buffer) // batch_size
        #
        # i = list(range(1, N))
        # random.shuffle(i)
        #
        # for batch in range(num_batches):
        #     yield self.get(self.buffer[])

        # TODO: shuffle all data, not just sample randomly multiple times

        num_batches = max(1, len(self.buffer) // self.config.batch_size)
        for _ in range(num_batches):
            yield self.sample()

    def clear(self):
        '''clear stored data'''
        self.buffer.clear()
