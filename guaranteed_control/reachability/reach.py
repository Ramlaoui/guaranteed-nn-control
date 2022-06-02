import numpy as np
import scipy
import matplotlib.pyplot as plt
import tensorflow as tf
from guaranteed_control.intervals.interval import Interval, regroup_close, over_appr_union, cut_state_interval
import tensorflow.keras as keras
from guaranteed_control.nn_reachability.nn_reachability_tf import plot_interval, add_to_plot, reachMLP, reachMLP_pendulum
from tqdm import tqdm
import time
from guaranteed_control.problems.dynamics import F_car, F_double_integrator, F_pendulum
import os 



def interval_approximation_naive(T, model, F, state_interval, epsilon, N= 5000, f=reachMLP, plot_jumps = 1, plot=True):
    
    # specification = True
    # if specification_interval == None:
    #     print("No specification interval")
    #     specification=False

    low, high = state_interval.high_low()
    random_points = [np.random.uniform(low[i], high[i], N) for i in range(len(high))]
    H = np.concatenate(random_points, axis=1)
    action_interval = f(model, H, epsilon, 5000, input_y=1, output_y=0)

    if plot:
        ax_state = plot_interval(state_interval, 0, 1)
        ax_state.set_xlabel(r"Position x")
        ax_state.set_ylabel(r"Speed $\dot{x}$")

    for i in tqdm(range(T), disable=not(plot)):

        state_interval = F(state_interval, action_interval)

        if plot:
            ax_state = add_to_plot(ax_state, state_interval, 0, 1)

        # if specification:
        #     if state_interval.is_included(specification_interval):
        #         if plot:
        #             ax_state = add_to_plot(ax_state, state_interval, 0, 1, 'r')
        #         break


        low, high = state_interval.high_low()
   
        random_points = [np.random.uniform(low[i], high[i], N) for i in range(len(high))]
        H = np.concatenate(random_points, axis=1)
    
        action_interval = f(model, H, epsilon, 1000, input_y=1, output_y=0, plot=plot_jumps-1)

        if plot_jumps - 1==1:
            state_plot = plot_interval(state_interval, 0, 1)
            plt.show()

    if (plot_jumps - 1 != 1) and plot:
        plt.show()

    return state_interval


# First attempt at improving the above function
def interval_approximation_beta(T, model, F, state_interval, specification_interval, epsilon_nn, epsilon_states=None, f= reachMLP, N=5000, plot_jumps=1, plot=True):

    """
    Awful complexity, it increases accuracy by a lot, but takes a long time to compute. The goal will be to improve that complexity
    """

    if epsilon_states == None:
        low, high = state_interval.high_low()
        random_points = [np.random.uniform(low[i], high[i], N) for i in range(len(high))]
        H = np.concatenate(random_points, axis=1)
        action_interval = f(model, H, epsilon_nn, N, input_y=1, output_y=0)

        return interval_approximation_naive(T, model, F, state_interval, epsilon_nn, specification_interval, f=f, plot_jumps=plot_jumps, plot=plot)
    
    state_intervals = cut_state_interval(state_interval, epsilon_states)
    outputs = []
    
    for state_interval in state_intervals:
        t = 1

        low, high = state_interval.high_low()
        random_points = [np.random.uniform(low[i], high[i], N).reshape(N, 1) for i in range(len(high))]
        H = np.concatenate(random_points, axis=1)
        action_interval = f(model, H, epsilon_nn, N, input_y=1, output_y=0, plot=plot_jumps-1)
        state_interval = F(state_interval, action_interval)

        
        while (action_interval.length() <= epsilon_states) and (t<T):
            t +=1
            low, high = state_interval.high_low()
            random_points = [np.random.uniform(low[i], high[i], N).reshape(N, 1) for i in range(len(high))]
            H = np.concatenate(random_points, axis=1)

            action_interval = f(model, H, epsilon_nn, N, input_y=1, output_y=0, plot=plot_jumps-1)
            state_interval = F(state_interval, action_interval)
        
        if t < T:
            outputs.append(interval_approximation(T-t, model, F, state_interval, specification_interval, epsilon_nn, epsilon_states=epsilon_states, f=f, plot_jumps=1, plot=False))
        else:
            outputs.append([state_interval])
    
    outputs = np.concatenate(outputs)
    
    return outputs
        

#Use verbose 2 to understand what happends with the interval size and the divisions
#What is happening, is that as we advance, we need to divide more the interval
#Could changing epsilon_actions as the iterations increase be the solution? 
# Final Closed loop reachability analysis algorithm
def interval_approximation(T, model, F, state_interval, specification_interval, epsilon_nn, epsilon_actions=None, threshold=0.2, f= reachMLP, N=5000, plot_jumps=1, plot=False, verbose=0, epsilon_increase = "uniform", actions_increase="uniform"):

    """
    Awful complexity, it increases accuracy by a lot, but takes a long time to compute. The goal will be to improve that complexity
    """

    states_iterations = []

    title = str(state_interval.intervals.tolist()).replace(" ", '')
    epsilons_nn = np.ones(T)*epsilon_nn
    epsilons_actions = np.ones(T)*epsilon_actions

    if actions_increase == "uniform":
        p = T//2
        mid_value = 1.5*epsilon_actions
        epsilons_actions[:p] = np.linspace(epsilon_actions, mid_value, p)
        epsilons_actions[p:] = np.linspace(mid_value, epsilon_actions, T-p)

    if epsilon_increase == "uniform":
        epsilons_nn = np.linspace(epsilon_nn, 1.5*epsilon_nn, T)



    if verbose == 2:
        time_reachability = []
        state_size = []
        errors = []
        action_size = []
        iteration = []


    if epsilon_actions == None:
        low, high = state_interval.high_low()
        random_points = [np.random.uniform(low[i], high[i], N) for i in range(len(high))]
        H = np.concatenate(random_points, axis=1)
        action_interval = f(model, H, epsilon_nn, N, input_y=1, output_y=0)

        return interval_approximation_naive(T, model, F, state_interval, epsilon_nn, specification_interval, f=f, plot_jumps=plot_jumps, plot=plot)
    
    if plot:
        try:
            os.mkdir(f"./plots/plot_interval_approx/{title}_{epsilon}_{epsilon_actions}/")
        except:
            print("folder already exists")
        ax_state = plot_interval(state_interval, 0, 1, "r")

    state_intervals = [state_interval]
    
    for t in tqdm(range(1, T+1)):
        state_intervals_new_step = []

        while len(state_intervals) != 0:
            state_interval = state_intervals.pop()
            low, high = state_interval.high_low()
            random_points = [np.random.uniform(low[i], high[i], N).reshape(N, 1) for i in range(len(high))]
            H = np.concatenate(random_points, axis=1)

            if verbose == 2:
                start_time = time.time()

            if verbose == 2:
                action_interval, error = f(model, H, epsilons_nn[t-1], N, epsilons_actions[t-1], input_y=1, output_y=0, plot=plot_jumps-1, verbose=2)
            else:
                action_interval = f(model, H, epsilons_nn[t-1], N, epsilons_actions[t-1], input_y=1, output_y=0, plot=plot_jumps-1)

            if verbose == 2:
                time_reachability.append(time.time() - start_time)
                state_size.append(state_interval.length())
                action_size.append(action_interval.length())
                iteration.append(t)
                errors.append(error)

            
            if action_interval.length() <= epsilon_actions:
                state_intervals_new_step.append(F(state_interval, action_interval))

            else:
                state_interval1, state_interval2 = state_interval.bissection()
                state_interval1, state_interval3 = state_interval1.bissection()
                state_interval2, state_interval4 = state_interval2.bissection()
                state_intervals.append(state_interval1)
                state_intervals.append(state_interval2)
                state_intervals.append(state_interval3)
                state_intervals.append(state_interval4)
        
        state_intervals = state_intervals_new_step

        if plot:
            ax_state = add_to_plot(ax_state, over_appr_union(state_intervals), 0, 1)
            plt.savefig(f"./plots/plot_interval_approx/{title}_{epsilon}_{epsilon_actions}/{title}_{epsilon}_{epsilon_actions}_{t}.png")

        # if t!=T:
            # state_intervals_temp = [over_appr_union(state_intervals)]
            # if state_intervals_temp[0].length() <= 0.2:
            #     state_intervals = state_intervals_temp
            
        state_intervals = regroup_close(state_intervals, threshold=threshold)
        states_iterations.append(state_intervals.copy())
  
    if plot:
        plt.show()
        for i, interv in enumerate(states_iterations[max(-10, -len(states_iterations)):]):
            if i == 0:
                ax_out = plot_interval(over_appr_union(interv), 0, 1)
            else:
                ax_out = add_to_plot(ax_out, over_appr_union(interv), 0, 1)
        plt.savefig(f'./plots/plot_interval_approx/{title}_{epsilon}_{epsilon_actions}/final_{title}_{epsilon}_{epsilon_actions}.jpg')
    plt.show()

    if verbose == 2:
        plt.scatter(np.arange(len(time_reachability)), time_reachability)
        plt.title("Time for NN reachability")
        plt.show()
        plt.scatter(np.arange(len(time_reachability)), iteration)
        plt.title("N iteration")
        plt.show()
        plt.scatter(np.arange(len(time_reachability)), action_size)
        plt.title("Action interval size")
        plt.show()
        plt.scatter(np.arange(len(time_reachability)), state_size)
        plt.title("State interval size")
        plt.show()
        plt.scatter(state_size, action_size)
        plt.title("Action over State sizes")
        plt.show()
        plt.scatter(np.arange(len(time_reachability)), state_size)
        plt.title("State size function of time")
        plt.show()
        plt.scatter(np.arange(len(time_reachability)), errors)
        plt.title("Error size")
        plt.show()
        plt.scatter(state_size, errors)
        plt.title("Error size function of state size")
        plt.show()
        plt.scatter(time_reachability, errors)
        plt.title("Error size function of time taken")
        plt.show()

    return states_iterations

