import gym
import torch
import numpy as np
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

def make_env(id):
    def _f():
        return gym.make(id)
    return _f

class Env:
    def __init__(self, env_name, actors=1):
        self.env = SubprocVecEnv([make_env(env_name) for _ in range(actors)])
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.actors = actors

        try:
            self.action_space_low = torch.FloatTensor(self.env.action_space.low)
            self.action_space_high = torch.FloatTensor(self.env.action_space.high)
        except:
            self.action_space_low = None
            self.action_space_high = None

    def reset(self):
        s = self.env.reset()
        if len(np.array(s).shape) == 0:
            s = np.expand_dims(s, axis=0)
        return s

    def explore_step(self, a):
        s2, r, done, info = self.env.step(a)
        if len(np.array(s2).shape) == 0:
            s2 = np.expand_dims(s2, axis=0)
        return s2, r, done, info

    def step(self, a):
        if isinstance(a, torch.Tensor):
            a = a.cpu().numpy()
        s2, r, done, info = self.env.step(a)
        if len(np.array(s2).shape) == 0:
            s2 = np.expand_dims(s2, axis=0)
        return s2, r, done, info
    
    def random_action(self):
        return np.stack([self.env.action_space.sample() for _ in range(self.actors)])

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

class TanhAction(Env):
    def __init__(self, env):
        self.env = env

    def __getattr__(self, k):
        return getattr(self.env, k)

    def step(self, a):
        a = ((torch.FloatTensor(a.cpu()) + 1) / 2) * (self.action_space_high - self.action_space_low) + self.action_space_low
        return self.env.step(a)
    
    def explore_step(self, a):
        s2, r, done, info = self.step(a)
        if len(np.array(s2).shape) == 0:
            s2 = np.expand_dims(s2, axis=0)
        return s2, r, done, info
    
    def random_action(self):
        a = torch.FloatTensor(np.stack([self.env.action_space.sample() for _ in range(self.actors)]))
        a = (a - self.action_space_low) / (self.action_space_high - self.action_space_low) * 2 - 1
        return a