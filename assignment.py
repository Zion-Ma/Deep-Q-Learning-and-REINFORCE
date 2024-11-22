import os
import sys
import gymnasium as gym
import numpy as np
import tensorflow as tf
try:
    from reinforce import Reinforce
    from reinforce_with_baseline import ReinforceWithBaseline
    from deep_q import DeepQModel
except:
    print("Please make sure to have the 'reinforce.py', 'reinforce_with_baseline.py', 'deep_q.py' files in the same directory as this file.")
from matplotlib.pyplot import plot, xlabel, ylabel, title, grid, show
from numpy import arange

def visualize_episode(model, env_name):
    """
    HELPER - do not edit.
    Takes in an enviornment and a model and visualizes the model's actions for one episode.
    We recomend calling this function every 20 training episodes. Please remove all calls of 
    this function before handing in.

    :param env: The cart pole enviornment object
    :param model: The model that will decide the actions to take
    """

    done = False
    env = gym.make(env_name, render_mode="human")
    state = env.reset()[0]
    while not done:
        newState = np.reshape(state, [1, state.shape[0]])
        prob = model.call(newState)
        newProb = np.reshape(prob, prob.shape[1])
        # if sum of probabilities is not 1, take max
        if np.sum(newProb) != 1:
            action = np.argmax(newProb)
        else:
            action = np.random.choice(np.arange(newProb.shape[0]), p = newProb)
        state, _, term, trunc, _ = env.step(action)
        if term or trunc:
            done = True
    

def visualize_data(total_rewards):
    """
    HELPER - do not edit.
    Takes in array of rewards from each episode, visualizes reward over episodes

    :param total_rewards: List of rewards from all episodes
    """

    x_values = arange(0, len(total_rewards), 1)
    y_values = total_rewards
    plot(x_values, y_values)
    xlabel('episodes')
    ylabel('cumulative rewards')
    title('Reward by Episode')
    grid(True)
    show()


def discount(rewards, discount_factor=.99):
    """
    Takes in a list of rewards for each timestep in an episode, and
    returns a list of the discounted rewards for each timestep, which
    are calculated by summing the rewards for each future timestep, discounted
    by how far in the future it is.
    For example, in the simple case where the episode rewards are [1, 3, 5] 
    and discount_factor = .99 we would calculate:
    dr_1 = 1 + 0.99 * 3 + 0.99^2 * 5 = 8.8705
    dr_2 = 3 + 0.99 * 5 = 7.95
    dr_3 = 5
    and thus return [8.8705, 7.95 , 5].
    Refer to the slides for more details about how/why this is done.

    :param rewards: List of rewards from an episode [r_{t1},r_{t2},...]
    :param discount_factor: Gamma discounting factor to use, defaults to .99
    :returns: discounted_rewards: list containing the discounted rewards for each timestep in the original rewards list
    """
    # TODO: Compute discounted rewards
    raise NotImplementedError

def generate_trajectory(env, model):
    """
    Generates lists of states, actions, and rewards for one complete episode.

    :param env: The openai gym environment
    :param model: The model used to generate the actions
    :returns: A tuple of lists (states, actions, rewards), where each list has length equal to the number of timesteps in the episode
    """
    states = []
    actions = []
    rewards = []
    state = env.reset()[0]
    done = False

    while not done:
        # TODO:
        # 1) use model to generate probability distribution over next actions
        # 2) sample from this distribution to pick the next action
        raise NotImplementedError
    
    return states, actions, rewards


def train_reinforce_episode(env, model):
    """
    This function should train your model for one episode.
    Each call to this function should generate a complete trajectory for one
    episode (lists of states, action_probs, and rewards seen/taken in the episode), and
    then train on that data to minimize your model loss.
    Make sure to return the total reward for the episode

    :param env: The openai gym environment
    :param model: The model
    :returns: The total reward for the episode
    """

    # TODO:
    # 1) Use generate trajectory to run an episode and get states, actions, and rewards.
    # 2) Compute discounted rewards.
    # 3) Compute the loss from the model and run backpropagation on the model.
    raise NotImplementedError

def train_deep_q_episode(env, model, batch_size, memory, epsilon=.1):

    # train_model for one episode
    state = env.reset()[0]
    done = False
    ep_rwd = []
    num_batches = 10.
    i=0
    raise NotImplementedError



def train(env, model, memory = None, epsilon=.1): 
    """
    This function should train your model for one episode.
    Each call to this function should generate a complete trajectory for one
    episode (lists of states, action_probs, and rewards seen/taken in the episode), and
    then train on that data to minimize your model loss.
    Make sure to return the total reward for the episode

    :param env: The openai gym environment
    :param model: The model
    :returns: The total reward for the episode
    """
    raise NotImplementedError

def main():
    if len(sys.argv) <2 or sys.argv[1] not in {"REINFORCE", "REINFORCE_BASELINE", "DEEP_Q"}:
        print("USAGE: python assignment.py <Model Type> <optional: env_name>")
        print("<Model Type>: [REINFORCE/REINFORCE_BASELINE/DEEP_Q]")
        exit()
    if len(sys.argv) == 3:
        try:
            env = gym.make(sys.argv[2])
            env_name = sys.argv[2]
        except Exception as e:
            print(f"Incorrect Environment Name: {sys.argv[2]}, make sure the environment is exactly as written in the gym documentation")
    else:
        env = gym.make("CartPole-v1")
        env_name = "CartPole-v1"

    state_size = env.observation_space.shape
    print("State size: ", state_size)
    state_size = state_size[0]
    num_actions = env.action_space.n

    # Initialize model
    if sys.argv[1] == "REINFORCE":
        model = Reinforce(state_size, num_actions)
    elif sys.argv[1] == "REINFORCE_BASELINE":
        model = ReinforceWithBaseline(state_size, num_actions)
    elif sys.argv[1] == "DEEP_Q":
        model = DeepQModel(state_size, num_actions)

    # Pipeline (we wrote this for you):
    # 1) Train your model for 650 episodes, passing in the environment and the agent.
    # 1a) OPTIONAL: Visualize your model's performance every 50 or so episodes.
    # 2) Append the total reward of the episode into a list keeping track of all of the rewards.
    # 3) After training, print the average of the last 50 rewards you've collected.
    totalReward = []
    num_episodes = 650
    viz_every = 50
    memory=None
    for episode in range(num_episodes):
        print(episode, end = "\r")
        if sys.argv[1] == "DEEP_Q":
            reward, memory = train(env, model, memory=memory, epsilon = 1-episode/num_episodes)
        else:
            reward = train(env, model)
        if episode in range(0, num_episodes):
            totalReward.append(reward)
        if episode%viz_every == 0:
            visualize_episode(model, env_name)
    visualize_episode(model, env_name)
    env.close()
    print(sum(totalReward)/len(totalReward))    
    visualize_data(totalReward)
if __name__ == '__main__':
    main()
