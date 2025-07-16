import gymnasium as gym
from gymnasium import RewardWrapper


class SparseMujoco(RewardWrapper):

    def __init__(self, env, accumulation_time=None):
        super().__init__(env)
        self.accumulation_time = accumulation_time
        self.current_step = 0
        self.current_accumulated_reward = 0

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.current_step = 0
        self.current_accumulated_reward = 0
        return obs, info

    def reward(self, _dense_r):
        if self.current_step < self.accumulation_time:
            self.current_step += 1
            self.current_accumulated_reward += _dense_r
            return 0.0
        else:
            self.current_step = 0
            return_value = self.current_accumulated_reward + _dense_r
            self.current_accumulated_reward = 0
            return return_value
