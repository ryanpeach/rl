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

from typing import List

from collections import deque
import random

import numpy as np

import torch
import torch.nn

from tqdm import trange

import gym
import matplotlib.pyplot as plt

# -----------
# Parameters
# -----------

## Gym
N_STATES = 5
N_ACTIONS = 2  # Left, Right

## Neural Network
BATCH_SIZE = 64
LEARNING_RATE = 1e-4

## Reinforcement Learning
MEMORY_SIZE = int(1e4)
STARTUP_SIZE = 100
TOTAL_RUNTIME = int(MEMORY_SIZE*2)
EPSILON_DECAY = 0.999
GAMMA = 0.9999

## Other
MOVING_AVERAGE = 100  # This is the window for our moving average

assert STARTUP_SIZE < TOTAL_RUNTIME

# ----------------
# Useful Functions
# ----------------

def moving_average(a: np.ndarray, n: int) -> np.ndarray:
    """
    A moving average function.
    Returns the moving average over a with a window length n.

    Reference: https://stackoverflow.com/a/14314054
    :param a: The array to average over.
    :param n: The length of the window to use to average.
    :return: np.ndarray
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# --------
# States
# --------
def angle_to_vector(pole_angle: float, n: int) -> torch.Tensor:
    """
    Turns the pole angle into a digitized hot one encoded vector.

    :param pole_angle:
    :param n:
    :return:
    """
    bins = np.linspace(-np.pi/2, np.pi/2, n-1)
    idx = np.digitize(pole_angle, bins)
    out = np.zeros(n, dtype="float32")
    out[idx] = 1
    return torch.from_numpy(out)

# -------------------
# The Neural Networks
# -------------------

class QNetwork(torch.nn.Module):
    inputlayer: torch.nn.Linear
    hidden: List[torch.nn.Linear]
    output: torch.nn.Linear

    def __init__(self):
        super(QNetwork, self).__init__()
        H = [10, 100, 10]
        self.hidden = []
        self.inputlayer = torch.nn.Linear(N_STATES, H[0])
        for h0, h1 in zip(H, H[1:]):
            self.hidden.append(torch.nn.Linear(h0, h1))
        self.output = torch.nn.Linear(H[-1], N_ACTIONS)
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.SGD(self.parameters(), lr=LEARNING_RATE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.inputlayer(x).clamp(min=0)
        for h in self.hidden:
            y = h(y).clamp(min=0)
        y = self.output(y)
        return y

    def fit_once(self, state: torch.Tensor, reward: torch.Tensor) -> None:
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = self(state)

        # Compute and print loss
        loss = self.criterion(y_pred, reward)

        # Zero gradients, perform a backward pass, and update the weights.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# --------
# Memory
# --------

class Memory(deque):
    """
    An implementation of Q Learning's Replay Memory.
    """
    def push(self, state: float, action: int, reward: float, next_state: float, done: bool) -> None:
        super(Memory, self).append((state, action, reward, next_state, done))

    ## ======= Bellman Equation =======
    def experience_replay(self, model):
        if len(self) < BATCH_SIZE:
            return
        batch = random.sample(self, BATCH_SIZE)
        for state, action, reward, state_next, done in batch:
            q_update = reward
            if not done:
                q_update = (reward + GAMMA * np.argmax(model(angle_to_vector(state_next, N_STATES)).detach().numpy()))
            state_vec = angle_to_vector(state, N_STATES)
            q_values = model(state_vec).detach().numpy()
            q_values[action] = q_update
            model.fit_once(state_vec, torch.from_numpy(q_values))



# -----------
# Main
#
# This is where we will mostly be spawning workers, initializing networks, and plotting our rewards.
# -----------

if __name__ == "__main__":
    memory = Memory([], MEMORY_SIZE)
    model = QNetwork()
    epsilon = 1
    all_epsilons, all_steps, all_rewards = [], [], []
    for i in trange(TOTAL_RUNTIME):
        env = gym.make("CartPole-v0")
        total_reward = 0.0
        total_steps = 0
        state = env.reset()
        while True:
            if i < STARTUP_SIZE:
                action = env.action_space.sample()
            else:
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(model(angle_to_vector(state, N_STATES)).detach().numpy())
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            total_steps += 1
            all_rewards.append(total_reward)
            all_steps.append(i)
            all_epsilons.append(epsilon)
            memory.push(state, action, reward, next_state, done)
            if done:
                break
            state = next_state
        if i >= STARTUP_SIZE:
            memory.experience_replay(model)
        epsilon *= EPSILON_DECAY

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(all_steps, all_rewards, color='red')
    ax1.plot(all_steps[MOVING_AVERAGE-1:], moving_average(all_rewards, n=MOVING_AVERAGE), color='blue')
    ax1.set_title('Cartpole Q Learning - Rewards over Training')
    ax1.set_xlabel('total_reward')
    ax1.set_ylabel('total_steps')
    ax2.plot(all_steps, all_epsilons, color='red')
    ax2.set_title('Epsilon vs Steps')
    ax2.set_xlabel('epsilon')
    ax2.set_ylabel('total_steps')
    fig.savefig('./output/Cartpole_Q_Learning.png')
