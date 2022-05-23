import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import gym
from guaranteed_control.ddpg.training import train, play

class Actor(keras.Model):

    def __init__(self, n_states, n_actions, upper_bounds, n_layer1 = 512, n_layer2=512):
        super().__init__()
        self.n_states= n_states
        self.n_layer1 = n_layer1
        self.n_layer2 = n_layer2
        self.n_actions = n_actions
        self.upper_bounds = upper_bounds

        self.layer1 = keras.layers.Dense(n_layer1, activation="relu")
        self.layer2 = keras.layers.Dense(n_layer2, activation="relu")
        self.layer3 = keras.layers.Dense(n_actions, activation="tanh")
        self.bound_layer = keras.layers.Lambda(lambda x: x * upper_bounds)

    def call(self, state):
        x = self.layer1(state)
        x = self.layer2(x)
        action = self.bound_layer(self.layer3(x))
        
        return action


class Critic(keras.Model):

    def __init__(self, n_states, n_actions, n_layer1=512, n_layer2=512):
        super().__init__()

        self.n_states = n_states
        self.n_layer1 = n_layer1
        self.n_layer2 = n_layer2
        self.n_actions = n_actions

        self.concat_layer = keras.layers.Concatenate()
        self.layer1 = keras.layers.Dense(n_layer1, activation="relu")
        self.layer2 = keras.layers.Dense(n_layer2, activation="relu")
        #No activation for the Critic output!
        self.layer3 = keras.layers.Dense(1)

    def call(self, state, action):
        
        x = self.concat_layer([state, action])
        x = self.layer1(x)
        x = self.layer2(x)
        q = self.layer3(x)

        return q



class MemoryBuffer():

    def __init__(self, n_states, n_actions, size=100000, batch_size=32):
        
        self.size = size
        self.batch_size = batch_size

        self.buffer_states = np.zeros((self.size, n_states))
        self.buffer_actions = np.zeros((self.size, n_actions))
        self.buffer_rewards = np.zeros((self.size, 1))
        self.buffer_states_ = np.zeros((self.size, n_states))

        self.position = 0

    def add_observation(self, state, action, reward, state_):

        index = self.position % self.size

        self.buffer_states[index] = state
        self.buffer_actions[index] = action
        self.buffer_rewards[index] = reward
        self.buffer_states_[index] = state_

        self.position += 1

    def extract_batch(self):

        if self.position < self.batch_size:
            return self.buffer_states, self.buffer_actions, self.buffer_rewards, self.buffer_states

        batch_indexes = np.random.choice(range(min(self.size, self.position)), size=self.batch_size)

        return self.buffer_states[batch_indexes], self.buffer_actions[batch_indexes], self.buffer_rewards[batch_indexes], self.buffer_states_[batch_indexes]

    

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)



class DDPG(keras.Model):

    def __init__(self, n_states, n_actions, upper_bounds, noise_std=0.2, learning_rate_critic = 0.002, learning_rate_actor= 0.001, gamma=0.99, tau = 0.005, n_layer1=32, n_layer2=32, batch_size=64, lambda_smooth=None, epsilon_s=0.2, D_s=10):
        super().__init__()

        self.tau = tau
        self.gamma = gamma
        self.lambda_s = lambda_smooth
        self.upper_bounds = upper_bounds
        self.n_states = n_states
        self.n_actions = n_actions
        self.epsilon_s, self.D_s = epsilon_s, D_s
        self.actions_perturb = tf.Variable(tf.zeros((batch_size,n_actions)), dtype=tf.float32)

        self.actor = Actor(n_states, n_actions,upper_bounds, n_layer1, n_layer2)
        self.critic = Critic(n_states, n_actions, n_layer1, n_layer2)
        self.target_actor = Actor(n_states, n_actions,upper_bounds, n_layer1, n_layer2)
        self.target_critic = Critic(n_states, n_actions, n_layer1, n_layer2)

        self.play = play
        self.start_training = train

        self.buffer = MemoryBuffer(n_states, n_actions, batch_size=batch_size)
        self.noise = OUActionNoise(np.zeros(n_actions), np.array(noise_std))

        self.actor.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate_actor))
        self.target_actor.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate_actor))
        self.critic.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate_critic))
        self.target_critic.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate_critic))


        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        self.optimizer = keras.optimizers.Adam()

    def update_target(self):

        weights = []
        target_weights = self.target_actor.trainable_variables
        for i, weight in enumerate(self.actor.trainable_variables):
            weights.append(self.tau * weight + (1-self.tau)* target_weights[i])
            target_weights[i].assign(weights[i])
        #tf.numpy_function(self.target_actor.set_weights, tf.Variable(weights), tf.float32)

        weights = []
        target_weights = self.target_critic.trainable_variables
        for i, weight in enumerate(self.critic.trainable_variables):
            weights.append(self.tau * weight + (1-self.tau)* target_weights[i])
            target_weights[i].assign(weights[i])
        #tf.numpy_function(self.target_critic.set_weights, tf.Varibale(weights), tf.float32)

        return

    def call(self, states):
        states = tf.convert_to_tensor(states)
        actions = tf.squeeze(self.actor(states))
        noise = self.noise()
        legal_action = tf.clip_by_value(actions + noise, -self.upper_bounds, self.upper_bounds)

        return legal_action

    @tf.function
    def learn(self, states, actions, rewards, states_):  

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_)
            target_critics = self.target_critic(states_, target_actions)
            states_i_critics = self.critic(states, actions)

            y = rewards + self.gamma*target_critics

            loss_critic = tf.math.reduce_mean(tf.math.square(y - states_i_critics))

        critic_gradients_wrt_weights = tape.gradient(loss_critic, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_gradients_wrt_weights, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            actor_actions = self.actor(states)
            critic_value = self.critic(states, actor_actions)

            actor_loss = -tf.math.reduce_mean(critic_value)

            if self.lambda_s != None:

                # delta = tf.Variable(tf.random.uniform((self.n_states, ), dtype=tf.float64), trainable=True)

                for i in range(states.shape[0]):
                    
                    delta = tf.cast(tf.random.uniform((self.n_states, )), dtype=tf.float64)

                    for l in range(self.D_s):

                        with tf.GradientTape() as tape2:
                            tape2.watch(delta)
                            state_pert = tf.expand_dims(states[i] + delta, 0)
                            action_state, action_perturb = actor_actions[i], self.actor(state_pert)
                            smoothness = tf.math.square(tf.norm(action_state - action_perturb))
                
                        delta = tf.cast(delta + self.epsilon_s*0.2*tape2.gradient(smoothness, delta), dtype=tf.float32)
                        #Projection on the ball of radius epsilon_s, centered around 0
                        delta = tf.cast(self.epsilon_s * delta * tf.math.minimum(1/self.epsilon_s, 1/tf.norm(delta)), dtype=tf.float64)
                    
                    
                    self.actions_perturb = self.actions_perturb[i, :].assign(action_perturb[0])

                actor_loss = -tf.math.reduce_mean(critic_value) + tf.math.reduce_mean(self.lambda_s*tf.math.square(tf.norm(self.actions_perturb - actor_actions, axis=1)))

        actor_gradients_wrt_weights = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_gradients_wrt_weights, self.actor.trainable_variables))
        
        self.update_target()
        
        return

    #One step train
    def train(self):

        states, actions, rewards, states_ = self.buffer.extract_batch()
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards)
        rewards = tf.cast(rewards, dtype=tf.float32)
        states_ = tf.convert_to_tensor(states_)

        self.learn(states, actions, rewards, states_)
        return

    def load_actor(self, path):
        self.actor = keras.models.load_model(path)

        return
