from charles.algorithm import Algorithm
from charles.models import *

class PPO(Algorithm):
    def __init__(self):
        self.name = 'PPO'
        self.type = 'on-policy'
        self.color = [153, 51, 153]

    def setup(self):
        self.π = Model(LinearPolicy, self.env, self.config.lr)
        self.V = Model(V, self.env, self.config.lr)

    def interact(self, s):
        a = self.π(s)
        log_p = self.π.log_prob(s, a.type(torch.FloatTensor))
        v = self.V(s)

        s2, r, done, _ = self.env.step(a)
        data = (s, a, r, v, log_p, done)
        return s2, r, done, data

    def update(self, storage):
        # for s, a, r, v, old_log_p, m in storage.sample_all():
        s, a, r, v, old_log_p, m = storage.get_all()

        # calculate returns
        returns = [0] * len(r)
        discounted_next = 0
        for i in reversed(range(len(r))):
            returns[i] = r[i] + discounted_next
            discounted_next = 0.99 * returns[i] * m[i - 1]
        returns = torch.stack(returns)

        # calculate and normalize advantage
        adv = returns - v
        mean = adv.mean()
        std = adv.std()
        adv = (adv - mean) / (std + 1e-6)

        # calculate new log probabilities
        new_log_p = self.π.log_prob(s, a)

        # optimize policy
        ratio = torch.exp(new_log_p - old_log_p)
        surrogate = torch.min(ratio * adv, torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * adv)
        policy_loss = -surrogate.mean()
        self.π.optimize(policy_loss)

        # optimize value network
        value_loss = ((self.V(s) - returns) ** 2).mean()
        self.V.optimize(value_loss)
