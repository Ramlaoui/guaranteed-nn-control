import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import time
import logging
import matplotlib.pyplot as plt
from guaranteed_control.intervals.interval import Interval, create_interval, Interval_tf, over_appr_union

# --------------------------------------------------------------------------
#      Functions for the reachability of the neural network
# --------------------------------------------------------------------------

#These functions only support Tensorflow push-forward neural networks

with open('./logs/nn_reachability.log', 'w'):
    pass

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

filehandler = logging.FileHandler("./logs/nn_reachability.log")

formatter = logging.Formatter(' %(asctime)s - %(name)s - %(message)s')
filehandler.setFormatter(formatter)

logger.addHandler(filehandler)


def generate_nn_outputs(model, H, N):

    """
    Input: Neural Network, set of points H, and number of simulations
    Samples N points from the input set H, and returns the N outputs 
    of the actions of the neural network on those inputs
    """

    indexes = np.random.choice(H.shape[0], size=N, replace=True)

    input_points = H[indexes]

    output_points = model(input_points)

    return output_points.numpy()


def nn_interval_py(model, input_interval):

    """
    Input: Neural Network, and input_interval (Interval.intervals)
    Returns the over-approximation of the image of the input interval 
    by the neural network, using interval approximation
    """

    layers = model.layers
    layers_weights = []
    activations = []
    biases = []

    for i in range(len(layers)):
        if i == len(layers) - 1:
            clipper = layers[i]
            break
        layers_weights.append(layers[i].weights[0].numpy())
        biases.append(layers[i].bias.numpy())
        activations.append(layers[i].activation)

    input_interval = Interval(interval=input_interval)

    for layer, bias, activation in zip(layers_weights, biases, activations):

        lows, highs = input_interval.high_low()
        layer_bool = layer >= 0

        input_low = np.where(np.transpose(layer_bool) == True, np.repeat(
            [lows], layer.shape[1], axis=0), np.repeat([highs], layer.shape[1], axis=0))
        input_high = np.where(np.transpose(layer_bool) == True, np.repeat(
            [highs], layer.shape[1], axis=0), np.repeat([lows], layer.shape[1], axis=0))

        output_low = np.zeros(shape=(layer.shape[1]))
        output_high = np.zeros(shape=(layer.shape[1]))
        for i in range(layer.shape[1]):

            output_low[i] = np.dot(layer[:, i], input_low[i, :]) + bias[i]
            output_high[i] = np.dot(layer[:, i], input_high[i, :]) + bias[i]

        output = np.concatenate([activation(output_low.reshape(1, output_low.shape[0])).numpy(
        ), activation(output_high.reshape(1, output_high.shape[0])).numpy()], axis=0)

        input_interval = create_interval(output)

    return (clipper(tf.convert_to_tensor(output)).numpy())


#Tensorflow implementation of the same function
@tf.function
def nn_interval_tf(model, eta):
    start_time = time.time()
    layers = model.layers
    clipper = layers[len(layers)-1]

    input_interval = eta

    for full_layer in layers[:-1]:

        layer, bias, activation = full_layer.weights[0], full_layer.bias, full_layer.activation
        lows, highs = input_interval[:, 0], input_interval[:, 1]
        # lows, highs = tf.convert_to_tensor(lows), tf.convert_to_tensor(highs)
        layer_bool = tf.cast(tf.transpose(layer >= 0), tf.float32)
        layer_bool_neg = 1-layer_bool
        # tf.print(time.time() - start_time)

        repeat_low = tf.tile(tf.reshape(
            lows, (1, lows.shape[0])), [layer.shape[1], 1])
        repeat_high = tf.tile(tf.reshape(
            highs, (1, lows.shape[0])), [layer.shape[1], 1])
        # tf.print(time.time() - start_time)
        # This might slow things down
        input_low = tf.math.multiply(
            repeat_low, layer_bool) + tf.math.multiply(repeat_high, layer_bool_neg)
        input_high = tf.math.multiply(
            repeat_high, layer_bool) + tf.math.multiply(repeat_low, layer_bool_neg)
        # tf.print(time.time()-start_time)
        # output_low = tf.Variable(tf.zeros(shape=(layer.shape[1])))
        # output_high = tf.Variable(tf.zeros(shape=(layer.shape[1])))

        # https://stackoverflow.com/questions/2301046/how-to-compute-only-the-diagonal-of-a-matrix-product-in-octave
        output_low = tf.math.reduce_sum(tf.math.multiply(
            tf.transpose(layer), (input_low)), 1) + bias
        output_high = tf.math.reduce_sum(tf.math.multiply(
            tf.transpose(layer), (input_high)), 1) + bias
        # tf.print(time.time() - start_time)
        # for i in range(layer.shape[1]):

        #     output_low[i] = tf.tensordot(layer[:,i], tf.cast(input_low[i,:],tf.float32), 0) + bias[i]
        #     output_high[i] = tf.tensordot(layer[:,i], tf.cast(input_high[i,:], tf.float32),0) + bias[i]

        output = tf.concat([activation(tf.reshape(output_low, (1, output_low.shape[0]))), activation(
            tf.reshape(output_high, (1, output_high.shape[0])))], axis=0)

        input_interval = tf.transpose(output)

    return (clipper((output)))



def nn_interval(mod, eta):
    input_interval = tf.convert_to_tensor(eta.intervals, tf.float32)
    return create_interval(nn_interval_tf(mod, input_interval))



# We start by an input set, and we want to know what the ouput set from the NN is
# We approximate that input set by an interval, and get an overapproximation of the output of NN using the interval approximation
# The goal is then to find a finer approximation of the output set: Use N simulations and approximate the output with an interval
# Cut the first input interval as many times as needed for it's output to be inside the simulated interval
# This will lead to a set smaller than the first interval, but we are still sure that it will be reached too!

# To do: add verbose and more or less display options

# We start by an input set, and we want to know what the ouput set from the NN is
# We approximate that input set by an interval, and get an overapproximation of the output of NN using the interval approximation
# The goal is then to find a finer approximation of the output set: Use N simulations and approximate the output with an interval
# Cut the first input interval as many times as needed for it's output to be inside the simulated interval
# This will lead to a set smaller than the first interval, but we are still sure that it will be reached too!

# To do: add verbose and more or less display options


#idea: stop cutting when we have a precision of delta

def reachMLP_py_(H, epsilon, N, epsilon_stop, input_x=0, input_y=1, output_x=0, output_y=1, over_appr=True):
    plot=False
    usim_set = generate_nn_outputs(model, H, N)
    usim = create_interval(usim_set)
    eta = create_interval(H)

    u = nn_interval(model, eta)

    ue = []
    etae = []

    if usim.length() > epsilon_stop:
        return u.intervals, usim.intervals, usim_set

    M = [(eta, u)]

    while len(M) > 0:
        (eta, u) = M.pop(0)
    #or u.length() - u.intersect(usim).length() < gamma
        if u.is_included(usim):
            ue.append(u)
            etae.append(eta)
            continue

        else:
            if eta.length() > epsilon:
                eta1, eta2 = eta.bissection()

                u1 = nn_interval(model, eta1)
                u2 = nn_interval(model, eta2)

                M.append((eta1, u1))
                M.append((eta2, u2))
            else:
                ue.append(u)
                etae.append(eta)
                # break
                # Why don't we continue?? It clearly improves the estimation

    ue = ue + [u for (eta, u) in M]
    etae = etae + [eta for (eta, u) in M]

    # for interv, inp_interv in zip(ue, etae):
    #     color = np.random.rand(3,)
    #     ax_output = add_to_plot(ax_output, interv, output_x, output_y, color)
    #     ax_input = add_to_plot(ax_input, inp_interv, input_x, input_y, color)

    if over_appr:
        ue = over_appr_union(ue).intervals

    # if plot:
    #     ax_output = add_to_plot(ax_output, ue, output_x, output_y)
    #     plt.show()

    return ue, usim.intervals, usim_set

def reachMLP_py(H, epsilon, N, epsilon_stop, input_x=0, input_y=1, output_x=0, output_y=1, over_appr=True, time_timeout=0.2):

    start_time = time.time()
    plot=False
    usim_set = generate_nn_outputs(model, H, N)
    usim = create_interval(usim_set)
    eta = create_interval(H)

    u = nn_interval(model, eta)

    ue = []
    etae = []

    if usim.length() > epsilon_stop:
        return u.intervals, usim.intervals, usim_set

    M = [(eta, u)]

    while len(M) > 0:
        (eta, u) = M.pop(0)
    #or u.intersect(usim).length() < gamma
    #add a control on ue to check if the length is fine?
    
        if not(u.intersect(usim)):
            if u.length() > 1e-2:
                logger.warning(f"estimated not included, size is {u.length()}")
            cond = u.is_included(usim) or u.length() < 1e-2
        else:
            logger.debug(f"{u.is_included(usim)} or {u.length() - u.intersect(usim).length()}")
            cond = u.is_included(usim) or u.length() - u.intersect(usim).length() < epsilon

        if cond or time.time() - start_time < time_timeout:
            ue.append(u)
            etae.append(eta)
            continue

        else:
            eta1, eta2 = eta.bissection()
    
            th1 = eta1.extract(axis=0)
            torque1 = eta1.extract(axis=1)
            eta1_augmented = (th1.cos().combine(th1.sin())).combine(torque1)
            th2 = eta2.extract(axis=0)
            torque2 = eta2.extract(axis=1)
            eta2_augmented = (th2.cos().combine(th2.sin())).combine(torque2)

            u1 = nn_interval(model, eta1_augmented)
            u2 = nn_interval(model, eta2_augmented)

            M.append((eta1, u1))
            M.append((eta2, u2))

    ue = ue + [u for (eta, u) in M]
    etae = etae + [eta for (eta, u) in M]

    # for interv, inp_interv in zip(ue, etae):
    #     color = np.random.rand(3,)
    #     ax_output = add_to_plot(ax_output, interv, output_x, output_y, color)
    #     ax_input = add_to_plot(ax_input, inp_interv, input_x, input_y, color)

    if over_appr:
        ue = over_appr_union(ue).intervals

    # if plot:
    #     ax_output = add_to_plot(ax_output, ue, output_x, output_y)
    #     plt.show()

    return ue, usim.intervals, usim_set




def reachMLP(mod, H, epsilon, N, epsilon_stop, over_appr=True, input_x=0, input_y=1, output_x=0, output_y=1, plot=0, verbose=1):
    global model
    model = mod
    ue, usim, usim_set = tf.numpy_function(reachMLP_py, [H, epsilon, N, epsilon_stop], [
                                           tf.Tensor, tf.Tensor, tf.Tensor, tf.float32])
    ue = Interval_tf(interval=ue.numpy())
    usim, usim_set = Interval_tf(interval=usim.numpy()), usim_set.numpy()

    if plot==1:
        ax_output = plot_interval(usim, output_x, output_y, 'r')
        ax_output.scatter(usim_set.T, usim_set.T)
        ax_output = add_to_plot(ax_output, ue, output_x, output_y)
        plt.show()

    if verbose == 2:
        error = ue.length() - usim.length()
        logger.debug(f"Length of output: {ue.length()} - Length of sim:{usim.length()}")
        return ue, error

    return ue


def reachMLP_pendulum_py_(H, epsilon, N, epsilon_stop, input_x=0, input_y=1, output_x=0, output_y=1):

    eta = create_interval(H)
 
    H = np.concatenate([np.cos(H[:,0]).reshape(H.shape[0],1), np.sin(H[:,0]).reshape(H.shape[0],1), H[:,1].reshape(H.shape[0], 1)], axis=1)
    usim_set = generate_nn_outputs(model, H, N)
    
    usim = create_interval(usim_set)
    
    th = eta.extract(axis=0)
    torque = eta.extract(axis=1)
    eta_augmented = (th.cos().combine(th.sin())).combine(torque)

    u = nn_interval(model, eta_augmented)

    if usim.length() > epsilon_stop:
        return u.intervals, usim.intervals, usim_set

    ue = []
    etae = []
    
    M = [(eta, u)]

    while len(M) > 0:
        (eta, u) = M.pop(0)
        if u.is_included(usim):
            ue.append(u)
            etae.append(eta)
            continue
        else:
            eta1, eta2 = eta.bissection()
    
            th1 = eta1.extract(axis=0)
            torque1 = eta1.extract(axis=1)
            eta1_augmented = (th1.cos().combine(th1.sin())).combine(torque1)
            th2 = eta2.extract(axis=0)
            torque2 = eta2.extract(axis=1)
            eta2_augmented = (th2.cos().combine(th2.sin())).combine(torque2)

            u1 = nn_interval(model, eta1_augmented)
            u2 = nn_interval(model, eta2_augmented)

            M.append((eta1, u1))
            M.append((eta2, u2))

    ue = ue + [u for (eta, u) in M]
    etae = etae + [eta for (eta, u) in M]

    ue = over_appr_union(ue).intervals


    return ue, usim.intervals, usim_set

def reachMLP_pendulum_py(H, epsilon, N, epsilon_stop, input_x=0, input_y=1, output_x=0, output_y=1, time_timeout=0.2):

    start_time = time.time()

    eta = create_interval(H)
 
    H = np.concatenate([np.cos(H[:,0]).reshape(H.shape[0],1), np.sin(H[:,0]).reshape(H.shape[0],1), H[:,1].reshape(H.shape[0], 1)], axis=1)
    usim_set = generate_nn_outputs(model, H, N)
    
    usim = create_interval(usim_set)
    
    th = eta.extract(axis=0)
    torque = eta.extract(axis=1)
    eta_augmented = (th.cos().combine(th.sin())).combine(torque)

    u = nn_interval(model, eta_augmented)

    if usim.length() + 2*epsilon > epsilon_stop:
        return u.intervals, usim.intervals, usim_set

    ue = []
    etae = []
    
    M = [(eta, u)]

    while len(M) > 0:
        (eta, u) = M.pop(0)

        if not(u.intersect(usim)):
            if u.length() > 1e-2:
                logger.warning(f"estimated not included, size is {u.length()}")
            cond = u.is_included(usim) or u.length() < 1e-2
        else:
            logger.debug(f"{u.is_included(usim)} or {u.length() - u.intersect(usim).length()}")
            cond = u.is_included(usim) or u.length() - u.intersect(usim).length() < epsilon

        if cond or time.time() - start_time < time_timeout:
            ue.append(u)
            etae.append(eta)
            continue

        else:
            eta1, eta2 = eta.bissection()
    
            th1 = eta1.extract(axis=0)
            torque1 = eta1.extract(axis=1)
            eta1_augmented = (th1.cos().combine(th1.sin())).combine(torque1)
            th2 = eta2.extract(axis=0)
            torque2 = eta2.extract(axis=1)
            eta2_augmented = (th2.cos().combine(th2.sin())).combine(torque2)

            u1 = nn_interval(model, eta1_augmented)
            u2 = nn_interval(model, eta2_augmented)

            M.append((eta1, u1))
            M.append((eta2, u2))

    ue = ue + [u for (eta, u) in M]
    etae = etae + [eta for (eta, u) in M]

    ue = over_appr_union(ue).intervals


    return ue, usim.intervals, usim_set

#add interval directly in the future (how with tensorflow?)
#How to make falcutative return arguments?

def reachMLP_pendulum(mod, H, epsilon, N, epsilon_stop, over_appr=True, input_x=0, input_y=1, output_x=0, output_y=1, plot=0, verbose=1):
    global model
    model = mod
    ue, usim, usim_set = tf.numpy_function(reachMLP_pendulum_py, [H, epsilon, N, epsilon_stop], [
                                           tf.Tensor, tf.Tensor, tf.Tensor])
    ue = Interval_tf(interval=ue.numpy())
    usim, usim_set = Interval_tf(interval=usim.numpy()), usim_set.numpy()

    if plot==1:
        ax_output = plot_interval(usim, output_x, output_y, 'r')
        ax_output.scatter(usim_set.T, usim_set.T)
        ax_output = add_to_plot(ax_output, ue, output_x, output_y)
        plt.show()

    if verbose == 2:
        error = ue.length() - usim.length()
        logger.debug(f"Length of output: {ue.length()} - Length of sim:{usim.length()}")
        return ue, error

    
    return ue



# --------------------------------------------------------------------------
#                           Interval plotting
# --------------------------------------------------------------------------

def plot_interval(interval, x_axis, y_axis, color=[0, 0, 1]):
    intervals = interval.intervals

    x_bounds = intervals[x_axis]
    y_bounds = intervals[y_axis]
    x_interval = np.linspace(x_bounds[0], x_bounds[1], 100)
    y_interval = np.linspace(y_bounds[0], y_bounds[1], 100)

    f, ax = plt.subplots()

    x = np.concatenate([x_interval, x_bounds[1] * np.ones(100),
                       np.flip(x_interval), x_bounds[0] * np.ones(100)])
    y = np.concatenate([y_bounds[0] * np.ones(100), y_interval,
                       y_bounds[1] * np.ones(100), np.flip(y_interval)])

    ax.plot(x, y, color=color)
    # plt.show()

    return ax


def add_to_plot(ax, interval, x_axis, y_axis, color=[0, 0, 1]):
    intervals = interval.intervals

    x_bounds = intervals[x_axis]
    y_bounds = intervals[y_axis]

    x_interval = np.linspace(x_bounds[0], x_bounds[1], 100)
    y_interval = np.linspace(y_bounds[0], y_bounds[1], 100)

    x = np.concatenate([x_interval, x_bounds[1] * np.ones(100),
                       np.flip(x_interval), x_bounds[0] * np.ones(100)])
    y = np.concatenate([y_bounds[0] * np.ones(100), y_interval,
                       y_bounds[1] * np.ones(100), np.flip(y_interval)])

    ax.plot(x, y, color=color)
    # plt.show()

    return ax
