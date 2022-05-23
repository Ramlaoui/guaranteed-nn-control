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
            

def plot_interval(interval, x_axis, y_axis, color=[0,0,1]):
    intervals = interval.intervals
    
    x_bounds = intervals[x_axis]
    y_bounds = intervals[y_axis]
    x_interval = np.linspace(x_bounds[0], x_bounds[1], 100)
    y_interval = np.linspace(y_bounds[0], y_bounds[1], 100)

    f, ax = plt.subplots()

    x = np.concatenate([x_interval, x_bounds[1] * np.ones(100), np.flip(x_interval), x_bounds[0] * np.ones(100)])
    y = np.concatenate([y_bounds[0] * np.ones(100), y_interval, y_bounds[1] * np.ones(100), np.flip(y_interval)])

    ax.plot(x, y, color=color)
    #plt.show()

    return ax

def add_to_plot(ax, interval, x_axis, y_axis, color=[0,0,1]):
    intervals = interval.intervals
    
    x_bounds = intervals[x_axis]
    y_bounds = intervals[y_axis]

    x_interval = np.linspace(x_bounds[0], x_bounds[1], 100)
    y_interval = np.linspace(y_bounds[0], y_bounds[1], 100)


    x = np.concatenate([x_interval, x_bounds[1] * np.ones(100), np.flip(x_interval), x_bounds[0] * np.ones(100)])
    y = np.concatenate([y_bounds[0] * np.ones(100), y_interval, y_bounds[1] * np.ones(100), np.flip(y_interval)])

    ax.plot(x, y, color=color)
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

#To do: add verbose and more or less display options

def reachMLP(model, H, epsilon, N, input_x=0, input_y=1, output_x=0, output_y=1, plot=True, over_appr=True):

    usim_set = generate_nn_outputs(model, H, N)
    
    usim = create_interval(usim_set)
    eta = create_interval(H)

    u = nn_interval(model, eta)
    
    ue = []
    etae = []

    if plot:
        print(u.intervals, usim.intervals)
        ax_input = plot_interval(eta, input_x, input_y)
        ax_output = plot_interval(usim, output_x, output_y, 'r')
        #ax_output = add_to_plot(ax_output, u, 0, 0)
        
        ax_output.scatter(usim_set.T, usim_set.T)
        #plt.show()

    M = [(eta, u)]

    while len(M) > 0:
        (eta, u) = M.pop(0)
        if u.is_included(usim):
            ue.append(u)
            etae.append(eta)

            if plot:
                color = np.random.rand(3,)
                ax_output = add_to_plot(ax_output, u, output_x, output_y, color)
                ax_input = add_to_plot(ax_input, eta, input_x, input_y, color)
            
            continue
        else:
            if eta.length() > epsilon:
                eta1, eta2 = eta.bissection()

                if plot:
                    ax_input = add_to_plot(ax_input, eta1, input_x, input_y, "b")
                    ax_input = add_to_plot(ax_input, eta2, input_x, input_y, "b")
               
                u1 = nn_interval(model, eta1)
                u2 = nn_interval(model, eta2)

                M.append((eta1, u1))
                M.append((eta2, u2))
            else:
                ue.append(u)
                etae.append(eta)
                #break
                #Why don't we continue?? It clearly improves the estimation

    ue = ue + [u for (eta, u) in M]
    etae = etae + [eta for (eta, u) in M]

    # for interv, inp_interv in zip(ue, etae):
    #     color = np.random.rand(3,)
    #     ax_output = add_to_plot(ax_output, interv, output_x, output_y, color)
    #     ax_input = add_to_plot(ax_input, inp_interv, input_x, input_y, color)

    if over_appr:
        ue = over_appr_union(ue)

    if plot:
        ax_output = add_to_plot(ax_output, ue, output_x, output_y)
        plt.show()


    return ue



def reachMLP_pendulum(model, H, eta, epsilon, N, input_x=0, input_y=1, output_x=0, output_y=1):

    usim_set = generate_nn_outputs(model, H, N)
    
    usim = create_interval(usim_set)
    
    th = eta.extract(axis=0)
    torque = eta.extract(axis=1)
    eta_augmented = (th.cos().combine(th.sin())).combine(torque)

    u = nn_interval(model, eta_augmented)
    
    ue = []
    etae = []
    print(u.intervals, usim.intervals)
    ax_input = plot_interval(eta, input_x, input_y)
    ax_output = plot_interval(usim, output_x, output_y, 'r')
    #ax_output = add_to_plot(ax_output, u, 0, 0)
    
    ax_output.scatter(usim_set.T, usim_set.T)
    #plt.show()

    M = [(eta, u)]

    while len(M) > 0:
        (eta, u) = M.pop(0)
        if u.is_included(usim):
            ue.append(u)
            etae.append(eta)
            color = np.random.rand(3,)
            ax_output = add_to_plot(ax_output, u, output_x, output_y, color)
            ax_input = add_to_plot(ax_input, eta, input_x, input_y, color)
            continue
        else:
            if eta.length() > epsilon:
                eta1, eta2 = eta.bissection()
                # ax_input = add_to_plot(ax_input, eta1, input_x, input_y, "b")
                # ax_input = add_to_plot(ax_input, eta2, input_x, input_y, "b")

                #Adjusting the inputs in order to have cos and sine instead of just the angle (that's how the agent was trained)
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
            else:
                ue.append(u)
                etae.append(eta)
                #break
                #Why don't we continue?? It clearly improves the estimation

    ue = ue + [u for (eta, u) in M]
    etae = etae + [eta for (eta, u) in M]

    for interv, inp_interv in zip(ue, etae):
        color = np.random.rand(3,)
        ax_output = add_to_plot(ax_output, interv, output_x, output_y, color)
        ax_input = add_to_plot(ax_input, inp_interv, input_x, input_y, color)

    ue = over_appr_union(ue)
    ax_output = add_to_plot(ax_output, ue, output_x, output_y)

    plt.show()


    return ue