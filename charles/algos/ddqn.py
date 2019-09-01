import random
from charles.algorithm import Algorithm
from charles.models import *

class DDQN(Algorithm):
    def __init__(self):
        self.name = 'DDQN'
        self.type = 'off-policy'
        self.color = [248, 136, 112]

    def setup(self):
        self.Q1 = Model(DQNNet, self.env, self.config.lr, target=True)
        self.Q2 = Model(DQNNet, self.env, self.config.lr, target=True)

        self.explore()

    def interact(self, s):
        if random.random() < 0.05:
            a = self.random_action()
        else:
            a = self.argmax(self.Q1(s))

        s2, r, done, _ = self.env.step(a)
        data = (s, a, r, s2, done)
        return s2, r, done, data

    def update(self, storage):
        s, a, r, s2, m = storage.sample()

        a2 = torch.argmax(self.Q1.target(s2), dim=-1, keepdim=True)
        next_q = self.Q2.target(s2).gather(-1, a2.long())
        y = r + (0.99 * m * next_q)

        q1_loss = ((self.Q1(s).gather(2, a.long()) - y) ** 2).mean()
        q2_loss = ((self.Q2(s).gather(2, a.long()) - y) ** 2).mean()

        self.Q1.optimize(q1_loss)
        self.Q2.optimize(q2_loss)

        self.Q1.soft_update_target()
        self.Q2.soft_update_target()
