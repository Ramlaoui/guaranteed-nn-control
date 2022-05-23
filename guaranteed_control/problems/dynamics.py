import tensorflow as tf
import numpy as np
from guaranteed_control.intervals.interval import Interval
import matplotlib.pyplot as plt


def F_car(state_interval,action_interval):

    power = 0.0015

    min_action, max_action = -1., 1.
    min_position, max_position = -1.2, 0.6
    max_speed = 0.07

    position = state_interval.extract(axis=0)
    velocity = state_interval.extract(axis=1)

    #Careful, p index
    # action_interval = action_interval.clip(min_action, max_action, axis=0)
    velocity = velocity + action_interval.alpha(power) - position.alpha(3).cos(axis=0).alpha(0.0025)
    
    velocity = velocity.clip(-max_speed, max_speed)

    position = position + velocity
    position = position.clip(min_position, max_position)
    
    

    #How to add that?? is it important for intervals??
    # if position == min_position and velocity < 0:
    #     velocity = 0

    return position.combine(velocity)


def F_car_point(x,action):
    
    power = 0.0015

    min_action, max_action = -1., 1.
    min_position, max_position = -1.2, 0.6
    max_speed = 0.07

    position = x[0]
    velocity = x[1]
    
    #Careful, p index
    # action_interval = action_interval.clip(min_action, max_action, axis=0)
    velocity = velocity + power*action - 0.0025*np.cos(3*position)
    
    velocity = np.clip(velocity, -max_speed, max_speed)

    position = position + velocity
    position = np.clip(position, min_position, max_position)
   
    #How to add that?? is it important for intervals??
    # if position == min_position and velocity < 0:
    #     velocity = 0

    return np.array([position, velocity])



def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def angle_normalize_interval(theta):
    theta_underline = theta.intervals[0][0]
    theta_overline = theta.intervals[0][1]
    return Interval(interval=[[min(angle_normalize(theta_underline), angle_normalize(theta_overline)), max(angle_normalize(theta_underline), angle_normalize(theta_overline))]])


def F_pendulum(state_interval,action_interval):

    max_speed = 8
    max_torque = 2.0
    dt = 0.05
    g = 10.0
    m = 1.0
    l = 1.0

    th = state_interval.extract(axis=0)
    thdot = state_interval.extract(axis=1)

    action_interval.clip(-max_torque, max_torque)
    
    # costs = (th.apply_incr(angle_normalize))**2 + (thdot**2).alpha(0.1) + (action_interval ** 2).alpha(0.001)
    
    newthdot = thdot + (th.sin()).alpha(3 *dt* g / (2 * l)) + action_interval.alpha((3.0 / (m * l ** 2)) * dt)
    newthdot = newthdot.clip(-max_speed, max_speed)
    
    newth = angle_normalize_interval(th + newthdot.alpha(dt))
    # print(th.intervals, action_interval.intervals)
    # print(thdot.intervals, newthdot.intervals)
    return newth.combine(newthdot)


def F_double_integrator(state_interval, action_interval):

    min_action, max_action = -1, 1

    x0 = state_interval.extract(axis=0)
    x1 = state_interval.extract(axis=1)

    # action_interval = action_interval.clip(min_action, max_action)

    newx0 = x0 + x1
    newx1 = x1 + action_interval

    return newx0.combine(newx1)