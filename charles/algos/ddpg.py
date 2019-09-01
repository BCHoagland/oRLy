from charles.algorithm import Algorithm
from charles.models import *

class DDPG(Algorithm):
    def __init__(self):
        self.name = 'DDPG'
        self.type = 'off-policy'
        self.color = [200, 78, 0]

    def setup(self):
        self.μ = Model(DeterministicPolicy, self.env, self.config.lr, target=True)
        self.Q = Model(Q, self.env, self.config.lr, target=True)

        self.explore()

    def interact(self, s):
        a = self.noisy_action(self.μ(s), 0.15)

        s2, r, done, _ = self.env.step(a)
        data = (s, a, r, s2, done)
        return s2, r, done, data

    def update(self, storage):
        s, a, r, s2, m = storage.sample()

        y = r + (0.99 * m * self.Q.target(s2, self.μ.target(s2)))
        q_loss = ((self.Q(s, a) - y) ** 2).mean()
        self.Q.optimize(q_loss)

        policy_loss = -self.Q(s, self.μ(s)).mean()
        self.μ.optimize(policy_loss)

        self.μ.soft_update_target()
        self.Q.soft_update_target()
