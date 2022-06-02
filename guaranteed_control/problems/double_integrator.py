import numpy as np
import gym
from guaranteed_control.intervals.interval import Interval
import time
from guaranteed_control.ddpg.ddpg import train, DDPG, play
import tensorflow as tf
import matplotlib.pyplot as plt


class double_integrator():

    def __init__(self, state_obj=np.array([0, 0]), cost=None):

        self.obj = state_obj
        
        self.reset()

        self.action_space = Interval(interval = [[-1, 1]])
        self.low_action, self.high_action = self.action_space.high_low()

        self.delta_t = 1
        self.min_x, self.max_x = -2, 2
        self.min_y, self.max_y = -2, 2
        self.observation_space = Interval(interval = [[-10, 10], [-10, 10]])
        self.pos_space = Interval(interval = [[self.min_x, self.max_x], [self.min_y, self.max_y]])

        self.max_iter = 50

        self.viewer = None
        self.cost = cost

        
    def step(self, action):
        done = False
        self.reward = 0

        action = np.clip(action, self.low_action, self.high_action)

        state_ = self.state

        state_[0] = state_[0] + state_[1]
        state_[1] = state_[1] + action[0]
        
        self.state = state_

        self.iteration += 1
        time.sleep(0.5)
        if self.iteration >= self.max_iter:
            done = True

        return self.state, self.reward, done, None

    
    def reset(self, input_interval=None):

        if input_interval == None:
            self.iteration = 0
            self.reward = 0
            self.state = np.array([0, 0])
        else:
            self.iteration = 0
            self.reward = 0
            interval = input_interval.intervals
            try:
                self.state = np.array([np.random.uniform(low = interval[0][0], high = interval[0][1]), np.random.uniform(low = interval[1][0], high = interval[1][1])])
                print(self.state)
            except RuntimeError:
                print("Bad shape for input interval, need one dimensional interval object")

        return np.array(self.state, dtype=np.float32)

        
    def render(self, mode="human"):
        screen_width = 600
        screen_height = 600

        world_width = self.max_x - self.min_x
        scale = screen_width / world_width
        car_side = 10
        car_y = 15

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)

            clearance = 10

            car = rendering.make_circle(10/2)
            # car = rendering.FilledPolygon([(-car_side/2, car_side/2), (car_side/2, car_side/2), (car_side/2, -car_side/2), -car_side/2, -car_side/2])
            # car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.cartrans.set_rotation(0)
           

            flagx = (self.obj[0] - self.min_x) * scale
            flagy = (self.obj[1] - self.min_y) * scale
            flag_offset = (1/10) * scale
            flag = rendering.FilledPolygon(
                [(flagx+flag_offset, flagy+flag_offset), (flagx+flag_offset, flagy-flag_offset), (flagx - flag_offset, flagy-flag_offset), (flagx-flag_offset, flagy + flag_offset)]
            )
            flag.set_color(0.8, 0.8, 0)
            self.viewer.add_geom(flag)
            self.viewer.add_geom(car)

        #This calls glRotate which implements a rotation using a rotation matrix or the rotation angle in degrees around a vector
        #RAD2DEG is dealth with in the render env
        pos_x = self.state[0]
        pos_y = 0
        self.cartrans.set_translation(
            (pos_x - self.min_x) * scale, (pos_y - self.min_y) * scale
        )
        

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None