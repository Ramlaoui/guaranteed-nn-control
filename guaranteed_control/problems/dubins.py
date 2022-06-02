import numpy as np
import gym
from guaranteed_control.intervals.interval import Interval
import time
from guaranteed_control.ddpg.ddpg import train, DDPG, play
import tensorflow as tf
import matplotlib.pyplot as plt

def cost1(self, state_, action, a, b):

        vector_to_obj = self.obj - state_[:2]

        theta_to_obj = angle_normalize(np.pi + np.arctan2(vector_to_obj[1], vector_to_obj[0]))

        if self.iteration == self.max_iter:
            print(theta_to_obj - state_[2])
            
        return np.square(np.linalg.norm(state_[:2])) + a*np.sum(np.linalg.norm(action)) + b*np.square(np.linalg.norm(theta_to_obj-state_[2]))
    

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

class Dubin():

    def __init__(self, state_obj=np.array([0, 0]), cost=None):

        self.obj = state_obj
        
        self.reset()

    #Careful, what is the condition on v2? depends on how much torque we give the car v1 (can't fully rotate absurdly)
        self.action_space = Interval(interval = [[-0.05, 0.05], [-0.2, 0.2]])
        self.low_action, self.high_action = self.action_space.high_low()

        self.delta_t = 1
        self.min_x, self.max_x = -1, 1
        self.min_y, self.max_y = -1, 1
        self.observation_space = Interval(interval = [[self.min_x, self.max_x], [self.min_y, self.max_y], [-np.pi, np.pi]])
        self.pos_space = Interval(interval = [[self.min_x, self.max_x], [self.min_y, self.max_y]])

        self.max_iter = 500

        self.viewer = None
        self.cost = cost

        
    def step(self, action):
        done = False


        action = np.clip(action, self.low_action, self.high_action)

        self.iteration +=1
        state = self.state
        state_ = np.zeros(state.shape)
    
        state_[0] = state[0] + action[0]*np.cos(state[2])*self.delta_t
        state_[1] = state[1] + action[0]*np.sin(state[2])*self.delta_t
        state_[2] = angle_normalize(state[2] + self.delta_t*action[1])


        state_[:2] = np.clip(state_[:2], *self.pos_space.high_low())

        vector_to_obj = self.obj - state_[:2]

        theta_to_obj = angle_normalize(np.pi + np.arctan2(vector_to_obj[1], vector_to_obj[0]))

        if self.iteration == self.max_iter:
            print(theta_to_obj - state_[2])
            
        self.state = np.array(state_)
        cost = np.square(np.linalg.norm(vector_to_obj)) + np.sum(np.square(action))
        cost = np.square(np.linalg.norm(vector_to_obj))
        cost = self.cost((self, state_, action))
        self.reward = -np.array(cost).astype(np.float32)

        # if np.linalg.norm(vector_to_obj) < 1e-2:
        #     print("here")
        #     self.reward = np.max([self.reward, 1000])

        if self.iteration >= self.max_iter:
            done = True

        return self.state, self.reward, done, None

    
    def reset(self, input_interval=None):

        if input_interval == None:
            self.iteration = 0
            self.reward = 0
            self.state = np.array([0, 0, 0])
        else:
            self.iteration = 0
            self.reward = 0
            interval = input_interval.intervals
            try:
                self.state = np.array([np.random.uniform(low = interval[0][0], high = interval[0][1]), np.random.uniform(low = interval[1][0], high = interval[1][1]), np.random.uniform(low = interval[2][0], high = interval[2][0])])
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

            # car = rendering.make_circle(10/2)
            car = rendering.FilledPolygon([(-car_side/2, -car_y/2), (0, car_y/2), (car_side/2, -car_y/2)])
            # car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.cartrans.set_rotation(0)
            self.viewer.add_geom(car)

            flagx = (self.obj[0] - self.min_x) * scale
            flagy = (self.obj[1] - self.min_y) * scale
            flag_offset = (1/10) * scale
            flag = rendering.FilledPolygon(
                [(flagx+flag_offset, flagy+flag_offset), (flagx+flag_offset, flagy-flag_offset), (flagx - flag_offset, flagy-flag_offset), (flagx-flag_offset, flagy + flag_offset)]
            )
            flag.set_color(1, 0, 0)
            self.viewer.add_geom(flag)

        #This calls glRotate which implements a rotation using a rotation matrix or the rotation angle in degrees around a vector
        #RAD2DEG is dealth with in the render env
        pos_x = self.state[0]
        pos_y = self.state[1]
        self.cartrans.set_translation(
            (pos_x - self.min_x) * scale, (pos_y - self.min_y) * scale
        )
        self.cartrans.set_rotation(self.state[2]+np.pi/2)
        

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None