import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from guaranteed_control.intervals.interval import Interval, over_appr_union, regroup_close, cut_state_interval
import tensorflow.keras as keras
from guaranteed_control.nn_reachability.nn_reachability_tf import reachMLP, reachMLP_pendulum, add_to_plot, plot_interval
from tqdm import tqdm
# from system_functions import F_car, F_double_integrator, F_pendulum



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
        
    

def interval_approximation(T, model, F, state_interval, specification_interval, epsilon_nn, epsilon_actions=None, threshold=0.2, f= reachMLP, N=5000, plot_jumps=1, plot=False):

    """
    Awful complexity, it increases accuracy by a lot, but takes a long time to compute. The goal will be to improve that complexity
    """

    if epsilon_actions == None:
        low, high = state_interval.high_low()
        random_points = [np.random.uniform(low[i], high[i], N) for i in range(len(high))]
        H = np.concatenate(random_points, axis=1)
        action_interval = f(model, H, epsilon_nn, N, input_y=1, output_y=0)

        return interval_approximation_naive(T, model, F, state_interval, epsilon_nn, specification_interval, f=f, plot_jumps=plot_jumps, plot=plot)
    
    if plot:
        ax_state = plot_interval(state_interval, 0, 1)

    state_intervals = [state_interval]
    
    for t in tqdm(range(1, T+1)):
        state_intervals_new_step = []

        while len(state_intervals) != 0:
            state_interval = state_intervals.pop()
            low, high = state_interval.high_low()
            random_points = [np.random.uniform(low[i], high[i], N).reshape(N, 1) for i in range(len(high))]
            H = np.concatenate(random_points, axis=1)
            action_interval = f(model, H, epsilon_nn, N, epsilon_actions, input_y=1, output_y=0, plot=plot_jumps-1)
            
            if action_interval.length() <= epsilon_actions:
                state_intervals_new_step.append(F(state_interval, action_interval))

            else:
                state_interval1, state_interval2 = state_interval.bissection()
                state_intervals.append(state_interval1)
                state_intervals.append(state_interval2)
        
        state_intervals = state_intervals_new_step

        if plot:
            ax_state = add_to_plot(ax_state, over_appr_union(state_intervals), 0, 1)

        # if t!=T:
            # state_intervals_temp = [over_appr_union(state_intervals)]
            # if state_intervals_temp[0].length() <= 0.2:
            #     state_intervals = state_intervals_temp

        state_intervals = regroup_close(state_intervals, threshold=threshold)
            
    if plot:
        plt.show()

    return state_intervals
    