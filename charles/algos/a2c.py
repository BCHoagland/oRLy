from charles.algorithm import Algorithm
from charles.models import *

class A2C(Algorithm):
    def __init__(self):
        self.name = 'A2C'
        self.type = 'on-policy'
        self.color = [115, 194, 251]

    def setup(self):
        self.π = Model(LinearPolicy, self.env, self.config.lr)
        self.V = Model(V, self.env, self.config.lr)

    def interact(self, s):
        a = self.π(s)
        s2, r, done, _ = self.env.step(a)
        data = (s, a, r, done)
        return s2, r, done, data

    def update(self, storage):
        s, a, r, m = storage.get_all()

        # calculate returns
        returns = [0] * len(r)
        discounted_next = 0
        for i in reversed(range(len(r))):
            returns[i] = r[i] + discounted_next
            discounted_next = 0.99 * returns[i] * m[i - 1]
        returns = torch.stack(returns)

        # calculate and normalize advantage
        adv = returns - self.V(s)
        mean = adv.mean()
        std = adv.std()
        adv = (adv - mean) / (std + 1e-6)

        # calculate log probabilities
        log_p = self.π.log_prob(s, a)

        # update policy
        policy_loss = -(adv * log_p).mean()
        self.π.optimize(policy_loss)

        # update value network
        value_loss = ((returns - self.V(s)) ** 2).mean()
        self.V.optimize(value_loss)
