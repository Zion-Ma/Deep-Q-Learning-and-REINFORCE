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

        # We require that you use tf.keras.Sequential to define the model and call it self.model
        #   (This is for auto-grading purposes)
        self.model = ???
        
        # We require that you call your target model self.target_model
        #    (This is for auto-grading purposes)
        #    Hints: You can clone the model using tf.keras.models.clone_model
        #           You can get the weights of model using get_weights
        #           You can set the weights of target_model using set_weights

        self.target_model = ???
        raise NotImplementedError

    def call(self, states):
        raise NotImplementedError
    

    def loss_func(self, batch, discount_factor = 0.99):
        
        # Compute the loss for the agent
        
        raise NotImplementedError