import gym
from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
import numpy as np
import types


def reset(self, input_interval=None):
    
    if input_interval == None:
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
    else:
        interval = input_interval.intervals
        try:
            self.state = np.array([self.np_random.uniform(low = interval[0][0], high = interval[0][1]), self.np_random.uniform(low = interval[1][0], high = interval[1][1])])
        except RuntimeError:
            print("Bad shape for input interval, need one dimensional interval object")
    return np.array(self.state, dtype=np.float32)


def mountain_car():

    car = gym.make("MountainCarContinuous-v0")
    car.unwrapped.reset = types.MethodType(reset, car.unwrapped)
    

    return car


