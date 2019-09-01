import random
from charles.algorithm import Algorithm
from charles.models import *

class DQN(Algorithm):
    def __init__(self):
        self.name = 'DQN'
        self.type = 'off-policy'
        self.color = [254, 127, 156]

    def setup(self):
        self.Q = Model(DQNNet, self.env, self.config.lr, target=True)

        self.explore()

    def interact(self, s):
        if random.random() < 0.05:
            a = self.random_action()
        else:
            a = self.argmax(self.Q(s), axis=1)

        s2, r, done, _ = self.env.step(a)
        data = (s, a, r, s2, done)
        return s2, r, done, data

    def update(self, storage):
        s, a, r, s2, m = storage.sample()

        max_next_q, _ = torch.max(self.Q.target(s2), dim=-1, keepdim=True)
        y = r + (0.99 * m * max_next_q)

        q_loss = ((self.Q(s).gather(-1, a.long()) - y) ** 2).mean()
        self.Q.optimize(q_loss)

        self.Q.soft_update_target()
