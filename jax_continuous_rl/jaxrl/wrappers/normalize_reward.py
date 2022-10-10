import gym
import numpy as np

from jaxrl.wrappers.common import TimeStep


class NormalizeReward(gym.Wrapper):

    def __init__(self, env, shift=0.0, scale=1.0):
        super().__init__(env)
        self.shift = shift
        self.scale =  scale

    def step(self, action: np.ndarray) -> TimeStep:
        obs, reward, done, info = self.env.step(action)
        
        norm_reward = (reward - self.shift) * self.scale
        info.update({'raw_reward': reward})

        return obs, norm_reward, done, info
