# Problem environments

This module implements various problems that we have presented in the project report, alongside their dynamics. In the "dynamics.py" file, you can find the interval arithmetic implementation of such dynamics.

## Moutain Car

The goal here is to wrap the "MountainCarContinuous-v0" environment from OpenAI's gym with a function to reset the problem inside a specified interval.

```python
from guaranteed_control.problems.mountain_car import mountain_car
from guaranteed_control.intervals.interval import Interval

start_interval = Interval(interval=[[-0.6, -0.4], [-0.07, 0.07]])
env = mountain_car()
env.reset()
agent = DDPG(env.observation_space.shape[0], 1, upper_bounds=env.action_space.high, n_layer1=16, n_layer2=16, batch_size=16, noise_std=0.4, epsilon_s=0.05, lambda_smooth=0, D_s=10)
agent.start_training(env, agent, input_interval=start_interval, n_episodes=600)
```

## Pendulum

The Pendulum environment is also extracted from gym with the possibility to call it on a specific start interval after reset.

## Double integrator

The Double Integrator is a simple linear system that is presented in the project report. We provide a way to vizualise the dynamics using pygl and gym's rendering objects.

## Dubins car

We also provide visualizations for the Dubins car problem.

```python
from guaranteed_control.problems.dubins import Dubin, cost1

a,b = 0.01, 0.06
env = Dubin(cost = lambda x : cost1(*x, a, b))
state = env.reset()
```
