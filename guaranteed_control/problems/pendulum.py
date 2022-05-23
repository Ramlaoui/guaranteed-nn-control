import gym
from gym.envs.classic_control.pendulum import PendulumEnv
import numpy as np


class Pendulum(PendulumEnv):
    
    def __init__(self, g=10.0):
        super().__init__(g)

    def reset(self, input_interval=None):

        high = np.array([np.pi, 1])

        if input_interval == None:
            self.state = self.np_random.uniform(low=-high, high=high)
        else:
            
            try:
                interval = input_interval.intervals
                self.state = np.array([self.np_random.uniform(low = interval[0][0], high = interval[0][1]), self.np_random.uniform(low = interval[1][0], high = interval[1][1])])
            except RuntimeError:
                print("Bad shape for input interval, need one dimensional interval object")

        self.last_u = None
        return self._get_obs()