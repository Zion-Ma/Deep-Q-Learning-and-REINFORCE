# Deep-Q-Learning-and-REINFORCE
This is a project to use TensorFlow to build DeepQ and REINFORCE neural network for a Cartpole agent.

## Overview:
In this assignment we will be using OpenAI's gym to try out some techniques to train agents in the Cartpole Environment.

## Assignment Goals
In this assignment you are going to implement the Deep-Q, REINFORCE and REINFORCE with BASELINE training algorithms. You'll also get familiar with the gym library through the Cartpole environment.

## Getting Started
Please click [here](https://classroom.github.com/a/1N46ElxC) to get the stencil code. 


:::danger 
**Do not change the stencil except where specified.** You are welcome to write your own helper functions, however, changing the stencil's method signatures **will** break the autograder
:::

## Roadmap


### 1. Deep-Q Learning

#### Motivation
Q-Learning is the method of training an agent to learn so-called "q values" associated with taking an action when in a particular state. A q-value is a **proxy for reward** that also incorporates predicted future returns as well. To maximize cumulative reward, it makes sense to take the action with the highest predicted q-value at every timestep, which becomes our **policy**. Refer to the lecture slides for more details and a formal statement of the problem. 

### Implementation

Head over to `deep_q.py`, here you will design your net for Deep Q learning

:::info
__Task 1.1 [DeepQModel Forward Pass]:__ Fill out the `__init__` and `call` methods for DeepQModel. The model should take a state/batch of states as input and output the predicted q-value of taking each action.

We recommend constructing an MLP using `tf.keras.Sequential`. 

You should initialize your _target model_ to be a copy of your prediction model in the constuctor. 

You'll need to build your prediction model $Q$ (check out the `.build` hint below), make a copy of $Q$ to be your target model, then copy the weights to the new target model. 

You may find `tf.keras.models.clone_model`, `.set_weights`, and `.get_weights` helpful for this part!
:::

:::warning
For this assignment, optimizers should be created in the model `__init__`, you can then access the optimizer when training by using `model.optimizer`. 
:::

:::warning
**About `.build`**

Tensorflow doesn't actually make the weights for a layer until it is called on an input. This way, it can stay generalizable to several different inputs (think about how dense layers don't need to have an input shape specified!)

We can force TF to build weights by calling `<insert model/layer_name>.build(input_shape=<insert shape iterable>)`
:::


:::success
__Note:__ In our case, the forward pass should take in a state and return the predicted reward __for each action__. The Q-value for a particular action can then be acquired by _indexing the output vector_.
:::

:::warning
An equally valid way to implement the network is to _append the action to the state as input_ and have the network return the predicted q-value of the state action pair. You __should not__ use this implementation for this assignment since it will not play nice with the autograder but feel free to come back to this assignment later on and make the necessary adjustments to train the network in this way. 
:::
:::info
__Task 1.2 [DeepQModel Loss]:__ Implement the loss function for a DQN. 
- Keep in mind each batch is a list of 5 items: _batch_states, batch_actions, batch_rewards, batch_next_states, batch_done_. Each is a list of length batch_size. Each example in the batch is independent so each example is non-consecutive and can be from different episodes entirely.
- The loss function _per example_ is as follows: 
    $$\mathcal{L}(Q, \hat{Q}, s, a, r, s^{`}, d) = \left(Q(s,a)-\left[r + \gamma*(1-\mathbb{1}_{d})*\max_{a^{`}}\{\hat{Q}(s^{`},a^{`})\}\right] \right)^2$$
    Where $s$ is the current state, $a$ is the action taken, $s^{`  }$ is the next state, $r$ is the reward, $\gamma$ is the discount factor (usually .99), $d$ is a boolean which is True if the game terminated after this action, false otherwise. $Q$ is the prediction network and $\hat{Q}$ is the target network. 
    - $\mathbb{1}_{d}$ is 1 if d is True and 0 otherwise.  
    
- As typical, you should vectorize this function and __return the mean loss over each example in the batch.__
:::

Now that we have a model, we need a way to train it!

:::success
__Note:__ `assignment.py` has several functions, many of which you will implement, some of which you won't have to worry about. Here is a breakdown of the different functions:
  - `visualize_data` and `visualize_episode` are utility methods we will use to visualize your trained agents and rewards acquired over the course of training. You don't have to worry about these, but you should look at their function signatures if you want to use them during/after training.
  - `discount` and `train_trajectory` are helper functions for REINFORCE and REINFORCE with BASELINE. You'll implement these later and use them to train with those algorithms but for now you can skip past them.
  - `train_reinforce_episode` and `train_deep_q_episode` should train the passed model on the passed environment for 1 episode. `train_deep_q_episode` has extra args for `batch_size`, `memory` and `epsilon` which are applied in the training algorithm for deep Q learning. You will be filling out `train_deep_q_episode` shortly.
  - `train` is used to run a single training episode, it will call either `train_reinforce/deep_q_episode` depending on what model was passed in. 
:::
:::info
__Task 1.3 [Prepare to train a DeepQModel part 1]:__ Implement `train_deep_q_episode`.

`train_deep_q_episode` should
1. Reset the environment and prepare any variables you need
2. Simulate a whole episode, append the information from each step into the memory bank. Each item in `memory` should be a `(state, action, reward, next_state, done)` tuple.
3. For n batches (around 10) train the model by:
    - Randomly choosing `batch_size` many examples from the memory
    - Convert the batch into to tensors and arrays
        - Since each item in memory has a different datatype, you'll need to split the values of your batch before converting to tensors or arrays. 
    - Compute the loss and update the weights of the model
4. Update the target network parameters
5. Return the sum total of rewards earned during the episode simulation and the updated memory bank
:::

:::success
__Note:__ When simulating an episode, you can interact with the environment with 
    - `env.step(action)` and `env.sample()`
:::
:::info
__Task 1.4 [Prepare to train a DeepQModel part 2]:__ Implement train.


`train` should
1. Check which model is being trained (use `isinstance` for this)
2. If training a `DeepQModel` and the memory is empty, initialize a memory of around 50 samples using random actions. Before calling `train_deep_q_episode`, be sure to **truncate the memory to 1000 examples**. 
3. If training either `Reinforce` or `ReinforceWithBaseline`, just call `train_reinforce_episode`
4. (Stencil code is incorrect): Return both the total reward and memory! This can be `None` for `Reinforce` and `ReinforceWithBaseline`
:::

:::success
**You should know:** Truncating the memory might at first seem like a bad idea since we are losing out on training data. This method has lots of merits however. Consider that our oldest examples are likely to contain worse state-action pairs that are not helpful to the training process. We'd prefer not to keep learning from bad examples and focus on state-action pairs that are closer to optimal. 
:::
:::info
__Task 1.4 [Train a DeepQModel]:__ You can run `main` by using the command line `python assignment.py DEEP_Q`. You can adjust the main function to tweek the `num_episodes`, how often episodes are visualized, etc. but the default values should work well. Ensure you can train a `DeepQModel` with average reward > 110. 
:::
---
### 2. REINFORCE
#### Motivation
REINFORCE is a policy gradient algorithm so instead of trying to learn the reward values it directly optimizes for the policy function.
:::info
__Task 2.1 [Reinforce initialization and call]:__ Implement the init and `call` method for `Reinforce`, we recommend a small MLP for this.
:::

:::info
__Task 2.2 [Reinforce loss function]:__ Implement `loss_func` for `REINFORCE`. 

The loss is given by
$$\mathcal{L}(s, a, r) = -\log(A(s,a))*\gamma r $$
Where $A$ is the predicted log probability of taking action, $a$, from state, $s$. $\gamma$ is the discount applied to the reward $r$.
:::
:::success
__Note:__ A couple hints when implementing `Reinforce.loss_func`

- You won't need to worry about the discount rate, $\gamma$, here since that'll be taken care later
- You'll need to call the model and use `gather_nd` to get the probabilities associated with each action
:::

:::info
__Task 2.3 [Prepare to train REINFORCE, discount]:__ Head over to `assignment.py` here you will find the `discount` function. Given a list of rewards, $r$, and discount rate $\gamma$, you should return a list, $d$, where

$$d[i] = \sum_{j=0}^i \gamma^j*r[j] $$

An example of expected behavior is provided in the handout.
:::

:::info
__Task 2.4 [Prepare to train REINFORCE, generate_trajectory]:__ 

Here you will simulate 1 episode using the model to generate the policy, you should return lists of all states, actions and rewards experienced during the episode. 
:::

:::warning
__Note:__ Recall that policy gradient methods output a _probability distribution_ over the action space and that this distribution __is a representation of our policy__. Thus, you should sample from this distribution rather than take a maximum predicted action.
:::

:::info
__Task 2.5 [Train REINFORCE]:__ Put your model, `generate_trajectory` and `discount` together to implement `train_reinforce_model` which should train on one simulated episode. 

Be sure to adjust `train` to call `train_reinforce_model` when using `Reinforce` and `ReinforceWithBaseline` models if you didn't do so in Task 1.4. 
:::

:::success
You should now be able to train REINFORCE by running `python assignment.py REINFORCE`
:::

### 3. REINFORCE with BASELINE

:::info
__Task 3.1 [ReinforceWithBaseline]:__ At this point, all you have to do are fill in the methods for `ReinforceWithBaseline` then you can run `python assignment.py REINFORCE_BASELINE` to train using the new model. We leave the details for this, the last task for 2470, to you (with a few helpful hints below).
:::

:::success
__Hints:__
- In our loss functions, instead of directly using discounted rewards, we use the so-called advantage function given by $$Adv(s) = \gamma r - V(s)$$
where $V$ is the value network (Critc network)
- The Actor loss for a single step is given by $$-(Adv(s)*\log(A(s,a))$$ where $A$ is the actor network
    - Be sure to use `tf.stop_gradient` around the $Adv$ when computing actor loss, if you do not, then the actor loss gradient will continue to propagate through the critic network. Consult the box below if you are unsure why this is necessary
- The critic loss for a single step is given by $$Adv(s)^2$$
- As with all loss functions in this course, you should return one value, the sum of the Critic and Actor losses
:::

:::success
__You should know:__
When you completed BERAS, the gradient method may have seem overly generalized and it may have been hard to imagine cases in which a search algorithm would have been useful. The above loss function is a perfect example of why the generalization is important. With a single GradientTape we can compute every gradient we need across both networks at once.
  
Think back to your gradient method and consider how it would have handled the $Adv$ term in the Actor loss. Since $Adv$ is parameterized, its weight gradients would have been computed and its weights then updated had we not used `tf.stop_gradient`. 
:::

## Thank you

Congratulations you have completed your last assignment for CSCI 2470! 

After submitting your work, there is some extra heft built into the stencil code to be able to train these networks using any `gym` environment, just add the name as a command line argument, such as `python assignment.py REINFORCE LunarLander-v2`. Cartpole is a very easy environment so you will likely need to run more episodes when training on other environments. That said, it can be a lot of fun to watch a network learn a more complicated environment with better graphics!
