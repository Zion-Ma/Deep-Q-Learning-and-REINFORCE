import os
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class DeepQModel(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        super(DeepQModel, self).__init__()
        self.num_actions = num_actions
        self.state_size = state_size
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.state_size,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.num_actions),
        ])
        self.model.build(input_shape = (None, self.state_size))
        # Create the target model by cloning the prediction model
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
        # Define the optimizerx
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    def call(self, states):
        return self.model(states) 
    def loss_func(self, batch, discount_factor = 0.99):
        # Compute the loss for the agent
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_done = batch
        # Convert batch data to tensors for vectorized computation
        batch_states = tf.convert_to_tensor(batch_states, dtype=tf.float32)
        batch_actions = tf.convert_to_tensor(batch_actions, dtype=tf.int32)
        batch_rewards = tf.convert_to_tensor(batch_rewards, dtype=tf.float32)
        batch_next_states = tf.convert_to_tensor(batch_next_states, dtype=tf.float32)
        batch_done = tf.convert_to_tensor(batch_done, dtype=tf.bool)
        # Compute Q-Values
        q_values = self.model(batch_states)
        next_q_values = self.target_model(batch_next_states)
        max_next_q_values = tf.reduce_max(next_q_values, axis=1)
        # Select the Q-Values for Actions Taken
        action_indices = tf.stack([tf.range(batch_actions.shape[0]), batch_actions], axis = 1)
        q_values_taken = tf.gather_nd(q_values, action_indices)
        # Compute the target Q-value
        target_q_values = batch_rewards + discount_factor * tf.cast(batch_done, tf.float32) * max_next_q_values
        # Compute the TD error
        td_error = q_values_taken - target_q_values
        # Compute the mean squared error
        loss = tf.reduce_mean(tf.square(td_error))
        return loss
