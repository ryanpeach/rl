#!/usr/bin/env python
# coding: utf-8
"""
An Implementation of A3C for Cartpole.

Made to look similar to my A2C code for ease of comparison.

Not an original implementation, but heavily modified from around the net,
and made to suit my experiements.

References:
    https://github.com/yc930401/Actor-Critic-pytorch
    https://github.com/MorvanZhou/pytorch-A3C

"""
# ---------
# Imports
# ---------

from itertools import count

import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.optim as optim
from torch.distributions import Categorical

# -----------
# Parameters
# -----------

## Pytorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(2)

## Neural Network
LR = .001
BETAS = (0.92, 0.999)
HIDDEN = [128, 256]

## Gym
global_env = gym.make("CartPole-v0")
CATEGORICAL_ACTION = True
ACTION_DIM = global_env.action_space.n
STATE_DIM = global_env.observation_space.shape[0]

## Reinforcement Learning
N_ITERS = 3000
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.99

## Other
MA_RATE = 0.99  # Used for the moving average at the end

# -------------------
# The Neural Networks
# -------------------

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        
        # We are going to make a feed forward network of depth len(HIDDEN)
        self.layers = []
        self.input_layer = nn.Linear(STATE_DIM, HIDDEN[0])
        for i, j in zip(HIDDEN, HIDDEN[1:]):
            self.layers.append(nn.Linear(i,j))
            
        # These will be our output layers
        # If the policy is categorical we need to use a softmax output
        self.policy_output_layer = nn.Linear(HIDDEN[-1], ACTION_DIM)

    def forward(self, state):
        temp = F.relu(self.input_layer(state))
        for layer in self.layers:
            temp = F.relu(layer(temp))
            
        # The policy network 
        # If the policy is categorical we need to sample from the softmax distribution
        policy = self.policy_output_layer(temp)
        if CATEGORICAL_ACTION:
            policy = Categorical(F.softmax(policy, dim=0))
            
        return policy


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        
        # We are going to make a feed forward network of depth len(HIDDEN)
        self.layers = []
        self.input_layer = nn.Linear(STATE_DIM, HIDDEN[0])
        for i, j in zip(HIDDEN, HIDDEN[1:]):
            self.layers.append(nn.Linear(i,j))
            
        # These will be our output layers
        # If the policy is categorical we need to use a softmax output
        self.value_output_layer = nn.Linear(HIDDEN[-1], 1)

    def forward(self, state):
        temp = F.relu(self.input_layer(state))
        for layer in self.layers:
            temp = F.relu(layer(temp))
            
        # The value output is a simple linear layer
        value = self.value_output_layer(temp)
        
        # Return as a tuple
        return value


# --------------
# Shared Adam
# --------------

class SharedAdam(optim.Adam):
    """
    An Adam optimizer with shared parameters for multiprocessing.

    Reference:
        Copied from https://github.com/MorvanZhou/pytorch-A3C/blob/master/shared_adam.py
    """
    def __init__(self, 
                 params, 
                 lr=LR, 
                 betas=BETAS, 
                 eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params,
                                         lr=lr,
                                         betas=betas, 
                                         eps=eps, 
                                         weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


# --------------
# Training Loop
# --------------

def create_worker(gnet_actor: Actor,
                  gnet_critic: Critic,
                  opt: SharedAdam,
                  global_episode: mp.Value,
                  global_results_queue: mp.Queue,
                  name: str) -> None:
    """
    This is our main function.

    It is in a function so that it can be spread over multiple processes.

    :param gnet_actor: Our global Actor network.
    :param gnet_critic: Our global Critic network.
    :param opt: Our shared Adam optimizer.
    :param global_episode: A shared value that tells us what episode we are on over all workers.
    :param global_results_queue: A shared queue that workers can put rewards onto.
    :param name: A name for this worker.
    :return: None
    """
    lnet_actor, lnet_critic = Actor(), Critic()
    lnet_critic.load_state_dict(gnet_critic.state_dict())
    lnet_actor.load_state_dict(gnet_actor.state_dict())
    lenv = gym.make('CartPole-v0')
    if int(name[1:]) == 0:
        video_recorder = VideoRecorder(lenv, './output/04_Cartpole_A3C.mp4', enabled=True)
    else:
        video_recorder = None

    print(f"Worker {name} starting run...")
    
    total_step = 1
    while global_episode.value < N_ITERS:
        buffer_state, buffer_log_probs, buffer_rewards = [], [], []
        episode_reward = 0
        state = lenv.reset()

        # Render the environment if you are the zeroth worker every 100 steps
        if (total_step+1) % 100 == 0 and int(name[1:]) == 0:
            if video_recorder:
                video_recorder.capture_frame()

        for _ in count():
            state = torch.FloatTensor(state).to(device)
            policy = lnet_actor(state)
            action = policy.sample()
            next_state, reward, done, _ = lenv.step(action.cpu().numpy())

            log_prob = policy.log_prob(action).unsqueeze(0)

            episode_reward += reward
            buffer_log_probs.append(log_prob[None, :])
            buffer_state.append(state[None, :])
            buffer_rewards.append(torch.FloatTensor([reward])[None, :].to(device))

            state = next_state

            if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                # sync
                final_value = lnet_critic(next_state) if not done else 0

                # Concatenate buffers
                buffer_state = torch.cat(buffer_state, dim=0)
                buffer_log_probs = torch.cat(buffer_log_probs, dim=0)
                buffer_rewards = torch.cat(buffer_rewards, dim=0)

                # Calculate the cumulative rewards using the final predicted value as the terminal value
                cum_reward = final_value
                discounted_future_rewards = torch.FloatTensor(len(buffer_rewards)).to(device)
                for i in range(len(buffer_rewards)):
                    cum_reward = buffer_rewards[-i] + GAMMA * cum_reward
                    discounted_future_rewards[-i] = cum_reward

                # Calculate the local losses for the states in the buffer
                values = lnet_critic(buffer_state)

                # Now we calculate the advantage function
                advantage = discounted_future_rewards - values

                # And the loss for both the actor and the critic
                actor_loss = -(buffer_log_probs * advantage.detach())
                critic_loss = advantage.pow(2)

                # calculate local gradients and push local parameters to global
                # We are going to couple these losses so that on each episode they are related together
                opt.zero_grad()
                (actor_loss + critic_loss).mean().backward()
                for lp, gp in zip(lnet_actor.parameters(), gnet_actor.parameters()):
                    gp._grad = lp.grad
                for lp, gp in zip(lnet_critic.parameters(), gnet_critic.parameters()):
                    gp._grad = lp.grad
                opt.step()

                # pull global parameters
                lnet_critic.load_state_dict(gnet_critic.state_dict())
                lnet_actor.load_state_dict(gnet_actor.state_dict())
                buffer_state, buffer_log_probs, buffer_rewards = [], [], []

                if done:
                    # Increment the global episode
                    with global_episode.get_lock():
                        global_episode.value += 1

                    # Update the results queue
                    # print(episode_reward)
                    global_results_queue.put(episode_reward)

                    # End this batch
                    break
        
    # This indicates its time to join all workers
    global_results_queue.put(None)
    print("DONE!")

    if video_recorder is not None:
        video_recorder.close()
    lenv.close()


# -----------
# Main
#
# This is where we will mostly be spawning workers, initializing networks, and plotting our rewards.
# -----------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # global networks
    gnet_actor = Actor()
    gnet_critic = Critic()

    # share the global parameters in multiprocessing
    gnet_actor.share_memory()         
    gnet_critic.share_memory()

    # Shared optimizers
    opt = SharedAdam(list(gnet_actor.parameters()) + list(gnet_critic.parameters()))

    # Some global variables and queues
    global_episode, global_results_queue = mp.Value('i', 0), mp.Queue()
    
    # parallel training
    workers = []
    for i in range(mp.cpu_count()):
        name = "w"+str(i).zfill(2)
        worker = mp.Process(target=create_worker,
                            args=(gnet_actor, gnet_critic, opt,
                                  global_episode, global_results_queue,
                                  name))
        worker.start()
        workers.append(worker)
    
    results = []                    # record episode reward to plot
    results_ma_ = 0
    results_ma = []

    # Similar to while True but with a counter.
    nb_ended_workers = 0
    for i in count():
        r = global_results_queue.get()
        if r is not None:
            results.append(r)

            # Create a moving average for results
            if results_ma_ == 0:
                results_ma_ = r
            else:
                results_ma_ = results_ma_ * MA_RATE + r * (1-MA_RATE)
            results_ma.append(results_ma_)

            # Print our moving average rewards over time
            if (i+1) % 100 == 0:
                print(f"Reward MA {str(i).zfill(4)}: {results_ma_}")

        else:
            nb_ended_workers += 1
            if nb_ended_workers == len(workers):
                break

    # Join all the workers
    for w in workers:
        w.join()

    # Plotting
    fig, ax = plt.subplots(1)
    ax.set_title("Cartpole A3C - Rewards over Training")
    ax.plot(results, label="rewards")
    ax.plot(results_ma_, label="moving average")
    fig.savefig('./output/Cartpole_A3C.png')

