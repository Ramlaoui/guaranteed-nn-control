import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from interval import Interval, create_interval



def generate_nn_outputs(model, H, N):

    indexes = np.random.choice(H.shape[0], size=N, replace=True)

    input_points = H[indexes]
    
    output_points = model(input_points)

    return output_points.numpy()




def nn_interval(model, eta):
    
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
    
    input_interval = eta
    
    for layer, bias, activation in zip(layers_weights, biases, activations):

        lows, highs = input_interval.high_low()
        layer_bool = layer >= 0

        input_low = np.where(np.transpose(layer_bool) == True, np.repeat([lows], layer.shape[1], axis=0), np.repeat([highs], layer.shape[1], axis=0))
        input_high = np.where(np.transpose(layer_bool) == True, np.repeat([highs], layer.shape[1], axis=0), np.repeat([lows], layer.shape[1], axis=0))

        output_low = np.zeros(shape=(layer.shape[1]))
        output_high = np.zeros(shape=(layer.shape[1]))
        for i in range(layer.shape[1]):
            
            output_low[i] = np.dot(layer[:,i], input_low[i,:]) + bias[i]
            output_high[i] = np.dot(layer[:,i], input_high[i,:]) + bias[i]
        
        output = np.concatenate([activation(output_low.reshape(1, output_low.shape[0])).numpy(), activation(output_high.reshape(1, output_high.shape[0])).numpy()], axis=0)

        input_interval = create_interval(output)
    
    input_interval = create_interval(clipper(tf.convert_to_tensor(output)).numpy())

    return input_interval
            

def plot_interval(interval, x_axis, y_axis, color='b'):
    intervals = interval.intervals
    
    x_bounds = intervals[x_axis]
    y_bounds = intervals[y_axis]
    x_interval = np.linspace(x_bounds[0], x_bounds[1], 100)
    y_interval = np.linspace(y_bounds[0], y_bounds[1], 100)

    f, ax = plt.subplots()

    x = np.concatenate([x_interval, x_bounds[1] * np.ones(100), np.flip(x_interval), x_bounds[0] * np.ones(100)])
    y = np.concatenate([y_bounds[0] * np.ones(100), y_interval, y_bounds[1] * np.ones(100), np.flip(y_interval)])

    ax.plot(x, y, color)
    #plt.show()

    return ax

def add_to_plot(ax, interval, x_axis, y_axis, color='b'):
    intervals = interval.intervals
    
    x_bounds = intervals[x_axis]
    y_bounds = intervals[y_axis]

    x_interval = np.linspace(x_bounds[0], x_bounds[1], 100)
    y_interval = np.linspace(y_bounds[0], y_bounds[1], 100)


    x = np.concatenate([x_interval, x_bounds[1] * np.ones(100), np.flip(x_interval), x_bounds[0] * np.ones(100)])
    y = np.concatenate([y_bounds[0] * np.ones(100), y_interval, y_bounds[1] * np.ones(100), np.flip(y_interval)])

    ax.plot(x, y, color)
    #plt.show()

    return ax


def over_appr_union(u):


    low, high = u[0].high_low()

    for interval in u:
        i_low, i_high = interval.high_low()
        low = np.where(i_low < low, i_low, low)
        high = np.where(i_high > high, i_high, high)
    
    return create_interval([low, high])

#We start by an input set, and we want to know what the ouput set from the NN is
#We approximate that input set by an interval, and get an overapproximation of the output of NN using the interval approximation
#The goal is then to find a finer approximation of the output set: Use N simulations and approximate the output with an interval
#Cut the first input interval as many times as needed for it's output to be inside the simulated interval
#This will lead to a set smaller than the first interval, but we are still sure that it will be reached too!

def reachMLP(model, H, epsilon, N, input_x=0, input_y=1, output_x=0, output_y=1):

    usim_set = generate_nn_outputs(model, H, N)
    print(usim_set)
    usim = create_interval(usim_set)
    eta = create_interval(H)

    u = nn_interval(model, eta)
    
    ue = []
    print(u.intervals, usim.intervals)
    ax_input = plot_interval(eta, input_x, input_y)
    ax_output = plot_interval(usim, output_x, output_y, 'r')
    #ax_output = add_to_plot(ax_output, u, 0, 0)
    
    ax_output.scatter(usim_set.T, usim_set.T)
    #plt.show()

    M = [(eta, u)]
    axis=0

    while len(M) > 0:
        (eta, u) = M.pop(0)
    
        if u.is_included(usim):
            ue.append(u)
            continue
        else:
            if eta.length() > epsilon:
                eta1, eta2 = eta.bissection()
                ax_input = add_to_plot(ax_input, eta1, input_x, input_y, 'b')
                ax_input = add_to_plot(ax_input, eta2, input_x, input_y, 'b')
                axis +=1
                u1 = nn_interval(model, eta1)
                u2 = nn_interval(model, eta2)

                M.append((eta1, u1))
                M.append((eta2, u2))
            else:
                ue.append(u)
                #break
                #Why don't we continue?? It clearly improves the estimation

    ue = ue + [u for (eta, u) in M]

    for interv in ue:
        ax_output = add_to_plot(ax_output, interv, output_x, output_y)

    ue = over_appr_union(ue)

    plt.show()


    return ue
