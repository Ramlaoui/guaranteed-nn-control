# Neural Network Reachability

In this module, we implement many functions that allow to do reachability analysis of intervals using neural networks. All the functions are implemented assuming that the neural networks are feed-forward NNs and that the activation functions are monotonous (ReLU, tanh, sigmoids...) and that they are implemented on TensorFlow.

## Arithmetic of intervals of a neural network

We provide a function that directly takes a neural network, and an interval and outputs the output interval over-approximation provided by as proven in the references of this project.

```python
from guarnateed_control.intervals.interval import Interval
from guaranteed_control.nn_reachability.nn_reachability_tf import nn_interval
import tensorflow as tf

x = Interval(interval=[[3.14, 3.14+1e-4], [1, 1+1e-4]])
model = tf.models.load_model(filepath="./models/pendulum_actor.tf")
u = nn_interval(model, x)
```

## Compute reachability set of a neural network with minimum division of input interval

The idea here is, given a $\epsilon$ value that is the maximum size we allow the input interval to be divided in, to compute the output reachability interval of a neural network, by minimizing the over-approximation error, using a simulated interval for control of that error metric.

```python
from guaranteed_control.intervals.interval import Interval
from guaranteed_control.nn_reachability.nn_reachability_tf import reachMLP_epsilon

model = tf.models.load_model(filepath="./models/car_actor.tf")
N = 5000
state_interval = Interval(interval=[[-0.45, -0.4495], [1, 1+1e-4]])
low, high = state_interval.high_low()
random_points = [np.random.uniform(low[i], high[i], N) for i in range(len(high))]
H = np.concatenate(random_points, axis=1)
epsilon = 0.1
actions_interval = reachMLP_epsilon(H, epsilon, N)
```

Here the variable H is a set of points extracted from the state_interval that allows to compute the control simulated interval (Note that we could have implemented the computation of H inside the function reachMLP without having to provide it, by using the state interval directly).

## Compute reachability set of a neural network by specifying a maximum error

Given a specified value $\delta$, the goal is for the estimated interval to not have a length that is higher than $\delta$ than the control interval. Note that this algorithm might not converge for very low values of $\delta$, so we just added a timeout condition that will just output the maximal precision it got during that runtime.

```python
from guaranteed_control.intervals.interval import Interval
from guaranteed_control.nn_reachability.nn_reachability_tf import reachMLP
import tensorflow as tf

model = tf.models.load_model(filepath="./models/pendulum_actor.tf")
N = 5000
state_interval = Interval(interval=[[-0.45, -0.4495], [1, 1+1e-4]])
low, high = state_interval.high_low()
random_points = [np.random.uniform(low[i], high[i], N) for i in range(len(high))]
H = np.concatenate(random_points, axis=1)
delta = 0.2
actions_interval = reachMLP(model, H, epsilon, N)
```

## Other functions

We also provided tools to plot multiple intervals inside this module.

```python
from guaranteed_control.nn_reachability.nn_reachability_tf import plot_interval, add_to_plot
import matplotlib.pyplot as plt


state_interval1 = Interval(interval=[[-0.45, -0.4495], [1, 1+1e-4]])
ax_state = plot_interval(state_interval1, x_axis=0, y_axis=1, color=[1, 0, 0])
state_interval2 = Interval(interval=[[-0.44, -0.4395], [1+1e-4, 1.2]])
ax_state = add_to_plot(ax_state, state_interval2, x_axis=0, y_axis=1, color=[0, 0, 1])
plt.show()
```
