from charles.algorithm import Algorithm
from charles.models import *
from charles.env import *

class SAC(Algorithm):
    def __init__(self):
        self.name = 'SAC'
        self.type = 'off-policy'
        self.color = [51, 152, 152]
        self.env_wrappers = [TanhAction]

    def setup(self):
        self.π = Model(TanhPolicy, self.env, self.config.lr)
        self.Q1 = Model(Q, self.env, self.config.lr, target=True)
        self.Q2 = Model(Q, self.env, self.config.lr, target=True)

        self.α = LearnableParam(0.2, self.config.lr)
        self.target_entropy = -torch.prod(torch.FloatTensor(self.env.action_space.shape)).item()

        self.explore()

    def interact(self, s):
        a = self.π(s)
        s2, r, done, _ = self.env.step(a)
        data = (s, a, r, s2, done)
        return s2, r, done, data

    def update(self, storage):
        s, a, r, s2, m = storage.sample()

        with torch.no_grad():
            a2, p2 = self.π.sample(s2)
            min_next_q = torch.min(self.Q1.target(s2, a2), self.Q2.target(s2, a2)) - (self.α * p2)
            y = r + (0.99 * m * min_next_q)

        q1_loss = ((self.Q1(s, a) - y) ** 2).mean()
        q2_loss = ((self.Q2(s, a) - y) ** 2).mean()
        self.Q1.optimize(q1_loss)
        self.Q2.optimize(q2_loss)

        new_a, p = self.π.sample(s)
        min_q = torch.min(self.Q1(s, new_a), self.Q2(s, new_a))
        policy_loss = (self.α * p - min_q).mean()
        self.π.optimize(policy_loss)

        α_loss = -(self.α * (p + self.target_entropy).detach()).mean()
        self.α.optimize(α_loss)

        self.Q1.soft_update_target()
        self.Q2.soft_update_target()
