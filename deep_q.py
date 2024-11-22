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

        # TODO: Define network parameters and optimizer
        raise NotImplementedError

    def call(self, states):
        raise NotImplementedError
    

    def loss_func(self, batch, discount_factor = 0.99):
        
        # Compute the loss for the agent
        
        raise NotImplementedError