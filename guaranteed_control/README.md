This library implements all of the following modules that interact with each other. The end goal is to be able to visulize all of the reachable sets of a dynamical system starting at a given interval after a given amount of time. Although we try to generalize the problems that can be studied using these modules, it is still very limited and you can try to personnalize the functions to your problems.

# Problems
This module implements the main problems that are used as example in the article. They were the ones used to test all of the functions in the code. 

# DDPG
We implement a DDPG controller in TensorFlow that can be quickly deployed to any environment that has the basic functions of Gym environments and trained. It is also possible to simulate the controller without having to make it train and visualize the outputs if your environment provides rendering.

# Intervals
Since the methods used are all based on interval over-approximations, we provide an Interval class that allows to manipulate these intervals in a suitable way for all the functions. This implementation can be improved by adding more interval functions. We also provide utility functions that are not directly inside the class.

# NN Reachability
This is where the neural network output over-approximation is computed. We also provide functions to plot the inputs and outputs of the NN reachability functions. Different display modes are provided in order to understand what the functions are doing to the intervals.

# Reachability
The main reachability functions are implemented here. The goal is to provide the controller as a neural network, and the dynamical system as a function of intervals that outputs a new state interval and to see what the reachability states are.

