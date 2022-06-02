
# DDPG Agent

The DDPG agent is based on DeepMind's paper. It consists in two actor networks and two critic networks used to output actions through one actor network, while the critics give a value that is related to how good the actions are to increase the rewards on the long term. We also add smooth regularization conditions for the actions of the actors in order to make the different states of the system be more continuous as described in the project's report.

# Creating a DDPG Agent

```python
from guaranteed_control.ddpg.ddpg import DDPG
from guaranteed_control.ddpg.training import train
from guaranteed_control.problems.pendulum import pendulum
from guaranteed_control.intervals.interval import Interval
import tensorflow as tf

env = pendulum()
n_obs = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]
agent = DDPG(n_obs, n_actions, upper_bounds=env.action_space.high, n_layer1=16, n_layer2=16, batch_size=32, noise_std=0.05, learning_rate_actor=1e-5, leraning_rate_critic=2e-5, tau=0.005)
input_interval = Interval(interval=[[3.14, 3.14+1e-4], [1, 1+1e-4]])
train(env, agent, input_interval=input_interval, n_episodes=200, plot_every=10, plot=True)
tf.keras.models.save_model(agent.actor, filepath="./models/pendulum_actor.tf", save_format="tf")
```

This code allows to create a simple DDPG agent and to train it on the Pendulum environment of Gym, while specifying the interval we want to train it on. 

# Adding smooth Regularization to the DDPG agent

```python
env = pendulum()
n_obs = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]
agent = DDPG(n_obs, n_actions, upper_bounds=env.action_space.high, n_layer1=16, n_layer2=16, batch_size=32, noise_std=0.05, learning_rate_actor=1e-5, leraning_rate_critic=2e-5, tau=0.005, epsilon_s=0.05, lambda_smooth=0.2, D_s=10)
input_interval = Interval(interval=[[3.14, 3.14+1e-4], [1, 1+1e-4]])
train(env, agent, input_interval=input_interval, n_episodes=200, plot_every=10, plot=True)
tf.keras.models.save_model(agent.actor, filepath="./models/pendulum_actor.tf", save_format="tf")
```

In order to add the smooth regularization to the agent, we need to specify the parameters $\epsilon_s, \lambda_s$ and $D_s$ to the DDPG class.

# Loading a trained DDPG controller and play in the corresponding environment

```python
from guaranteed_control.ddpg.ddpg import DDPG
from guaranteed_control.ddpg.training import play
from guaranteed_control.problems.pendulum import pendulum
from guaranteed_control.intervals.interval import Interval
import tensorflow as tf

env = pendulum()
n_obs = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]
agent = DDPG(n_obs, n_actions, upper_bounds=env.action_space.high, n_layer1=16, n_layer2=16)
agent.actor = tf.keras.models.load_model(".models/pendulum_actor.tf")
input_interval = Interval(interval=[[3.14, 3.14+1e-4], [1, 1+1e-4]])
eps_rewards, observations, actions = play(env, agent, n_games=16, input_interval=input_interval, plot=False, watch=True)
```

We can then load the trained agent and make it play a specified number of games. This stores the corresponding rewards, the states of the system during every iteration of every epsiodes and the actions returned by the controller.