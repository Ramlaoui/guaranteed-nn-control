import gym
from gym.envs.classic_control.pendulum import PendulumEnv
from gym import spaces
from gym.utils import seeding
import numpy as np
import types


def reset(self, input_interval=None):

    self._elapsed_steps = 1
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


def pendulum():

    pendulumv1 = gym.make("Pendulum-v1")
    pendulumv1.unwrapped.reset = types.MethodType(reset, pendulumv1.unwrapped)
    

    return pendulumv1



