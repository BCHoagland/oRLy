import torch
import numpy as np
from charles.env import *
from charles.storage import *
from charles.visualize import *

class Agent:
    def __init__(self, algo, config):
        self.env = Env(config.env, config.actors)
        try:
            for env_wrapper in self.algo.env_wrappers:
                self.env = env_wrapper(self.env)
        except:
            pass

        self.visualizer = Visualizer(config.env)
        self.config = config

        self.storage = Storage(config)

        self.algo = algo()
        self.algo.agent = self
        self.algo.setup()

    def explore(self):
        s = self.env.reset()
        for step in range(int(10000)):
            a = np.stack([self.env.action_space.sample() for _ in range(self.config.actors)])
            s2, r, done, _ = self.env.step(a)
            self.storage.store((s, a, r, s2, done))
            s = s2

    def train(self):
        total_timesteps = 0

        ep_reward = np.zeros(self.config.actors)
        final_ep_reward = np.zeros(self.config.actors)

        s = self.env.reset()
        while total_timesteps < self.config.max_timesteps:

            # collect trajectory
            for t in range(int(self.config.trajectory_length)):

                # collect transition
                with torch.no_grad():
                    s2, r, done, data = self.algo.interact(s)
                self.storage.store(data)
                s = s2

                # update stored rewards
                ep_reward += r
                mask = 1 - done
                final_ep_reward *= mask
                final_ep_reward += (1 - mask) * ep_reward
                ep_reward *= mask

                # visualize progress occasionally
                total_timesteps += 1
                if total_timesteps == 0 or total_timesteps % self.config.vis_iter == 0:
                    self.visualizer.plot(self.algo.name, 'Episodic Reward', 'Timesteps', total_timesteps, final_ep_reward, self.algo.color)

            # run updates after trajectory has been collected
            for _ in range(self.config.epochs):
                self.algo.update(self.storage)
            if self.algo.type == 'on-policy':
                self.storage.clear()
