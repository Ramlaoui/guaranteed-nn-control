import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import gym


def train(env, agent, input_interval=None, n_episodes=200, plot=True, plot_every=100, verbose=1):

    def env_step(action):
        state, reward, done, info = env.step(action)
        return (state.astype(np.float32), reward, int(done))

    eps_rewards = []

    for i in range(n_episodes):
        observation = env.reset(input_interval=input_interval)
        done = False
        episode_reward = 0

        while not done:
            
            if plot and i%plot_every==0:
                env.render()
            action = agent([observation]) 
            action = action
            observation_, reward, done = tf.numpy_function(env_step, [action], [tf.float32, tf.float32, tf.int8])
            
            agent.buffer.add_observation(observation, action, reward.numpy(), observation_.numpy())
            episode_reward += reward
            if agent.buffer.position >= agent.buffer.batch_size:
                agent.train()
                #agent.update_target()
            observation = observation_
            if done:
                break
        if plot and i%plot_every == 0:
            env.close()
        eps_rewards.append(episode_reward)

        if verbose:
            print(f"Episode {i} finished with reward: {episode_reward}, average reward {np.mean(eps_rewards[-40:])}")

    plt.plot(eps_rewards)
    plt.show()

    #env.close()
    return


def play(env, agent, n_games=10, input_interval=None, watch=False, plot=True):

    def env_step(action):
        state, reward, done, info = env.step(action)
        return (state.astype(np.float32), reward, int(done))

    eps_rewards = []
    observations = []
    actions = []
    
    for i in range(n_games):

        observation = env.reset(input_interval=input_interval)
        done = False
        episode_reward = 0

        episode_reward = 0
        obs_array = [observation]
        action_array = []

        while not done:

            if watch:
                env.render()

            action = agent([observation]) 
            action = action
            observation_, reward, done = tf.numpy_function(env_step, [action], [tf.float32, tf.float32, tf.int8])
  
            episode_reward += reward
            observation = observation_

            obs_array.append(observation.numpy())
            action_array.append(action.numpy())

            if done:
                break

        eps_rewards.append(episode_reward)
        print("Episode {} finished with reward: {}".format(i, episode_reward))

        observations.append(obs_array)
        actions.append(action_array)
        eps_rewards.append(episode_reward)

        if watch:
            env.close()

    if plot:
        n = int(np.ceil(np.sqrt(n_games)))
        fig, axs = plt.subplots(n, n, figsize=(20, 20))
        for i in range(n_games):
            axs[i//n][i%n].plot(np.array(actions[i])[:,0])
            axs[i//n][i%n].set_title(f"episode {i}")
            axs[i//n][i%n].set_xlabel("k")
            axs[i//n][i%n].set_ylabel("u(k)")
        plt.show()


    return eps_rewards, observations, actions