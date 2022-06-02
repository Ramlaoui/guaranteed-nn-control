# Reachability analysis

This is the module in which we implement the functions for the closed loop reachability analysis of the system. 

## Naive Reachability analysis

This is how to use the naive reachability analysis algorithm that progressively implements the closed loop analysis without controlling the size of the action interval.

```python
from guaranteed_control.intervals.interval import Interval
from guaranteed_control.sys_reachability.reach import interval_approximation
from guaranteed_control.nn_reachability.nn_reachability_tf import reachMLP
import tensorflow as tf

# For small neural networks, the reachability analysis is way faster with the CPU because of memory exchanges with GPU taking a long time
with tf.device('/cpu:0'):
    model = keras.models.load_model("./models/car_actor_speed_interval.tf")
    state_interval = Interval(interval = [(-0.48, -0.475), (0.0095,0.01)])
    epsilon= 0.0001
    state_interval  = interval_approximation(35, model, F_car, state_interval, None, epsilon, f=reachMLP, epsilon_actions=None, plot_jumps=1, plot=True, verbose=2)
```

## Stability Reachability analysis

When we need the state intervals to contract for stability reachability analysis, we need to present the second version of the reachability algorithm proposed in the project report. Here, we need to provide a value for epsilon_actions that will depend on the problem. epsilon here is the error tolerance for the Neural Network reachability algorithm. In order to get optimal computation time, you can try increasing progressively the iteration number T, and set verbose to 2, in order to understand how the divisions are being made. This will guide you in tuning your parameters epsilon_actions, epsilon, threshold, and state_interval size.

```python
from guaranteed_control.intervals.interval import Interval
from guaranteed_control.sys_reachability.reach import interval_approximation
from guaranteed_control.nn_reachability.nn_reachability_tf import reachMLP_pendulum
import tensorflow as tf
import numpy as np

with tf.device('/cpu:0'):
    model = keras.models.load_model("./models/pendulum_smooth_actor3.tf")
    state_interval = Interval(interval = [(np.pi, np.pi+0.0001), (1, 1.0001)])
    epsilon = 0.2
    state_interval  = interval_approximation(80, model, F_pendulum, state_interval, None, epsilon,f=reachMLP_pendulum, epsilon_actions=0.2, plot_jumps=1, plot=True, threshold=0.2, verbose=2, epsilon_increase="", actions_increase="uniform")
```