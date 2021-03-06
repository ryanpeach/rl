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

import math

from collections import deque
import random

import numpy as np

import torch
import torch.nn

from tqdm import trange

import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder

import matplotlib.pyplot as plt

if torch.cuda.is_available():
    print("NVIDIA")
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

# -----------
# Parameters
# -----------

## Pytorch
torch.manual_seed(0)

## Gym
N_STATES = 2
N_ACTIONS = 2  # Left, Right

## Neural Network
BATCH_SIZE = 64
LEARNING_RATE = 1e-4

## Reinforcement Learning
MEMORY_SIZE = int(2e4)
STARTUP_SIZE = 100
TOTAL_RUNTIME = int(MEMORY_SIZE*2)
EPSILON_DECAY = 0.9999
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
def angle_to_vector(pole_angle_rad: float, n: int) -> torch.Tensor:
    """
    Turns the pole angle into a digitized hot one encoded vector.

    :param pole_angle:
    :param n:
    :return:
    """
    bins = np.linspace(-np.pi/2, np.pi/2, n+1)[1:-1]
    idx = np.digitize(pole_angle_rad, bins)
    out = np.zeros(n, dtype="float32")
    out[idx] = 1
    return out

def test_angle_to_vector():
    assert (angle_to_vector(0, 2) == [0, 1]).all()
    assert (angle_to_vector(math.radians(-15), 2) == [1, 0]).all()
    assert (angle_to_vector(math.radians(15), 2) == [0, 1]).all()

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
        self.inputlayer = torch.nn.Linear(N_STATES, H[0]).to(device)
        for h0, h1 in zip(H, H[1:]):
            self.hidden.append(torch.nn.Linear(h0, h1).to(device))
        self.output = torch.nn.Linear(H[-1], N_ACTIONS).to(device)
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
        all_state_vec = []
        all_qvalues = []
        for state, action, reward, state_next, done in batch:
            q_update = reward
            if not done:
                q_update = (reward + GAMMA * np.argmax(model(torch.from_numpy(angle_to_vector(state_next, N_STATES)).to(device)).cpu().detach().numpy()))
            state_vec = torch.from_numpy(angle_to_vector(state, N_STATES)).to(device)
            all_state_vec.append(state_vec)
            q_values = model(state_vec)
            q_values[action] = q_update
            all_qvalues.append(q_values)
        model.fit_once(torch.stack(all_state_vec), torch.stack(all_qvalues))



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
    env = gym.make("CartPole-v0")
    video_recorder = VideoRecorder(env, './output/01_Cartpole_Q_Learning_Video.mp4', enabled=True)
    for i in trange(TOTAL_RUNTIME):
        total_reward = 0.0
        total_steps = 0
        state = env.reset()[2]
        while True:
            if (i+1) % 100 == 0:
                video_recorder.capture_frame()
            if i < STARTUP_SIZE:
                action = env.action_space.sample()
            else:
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(model(torch.from_numpy(angle_to_vector(state, N_STATES)).to(device)).cpu().detach().numpy())
            next_state, reward, done, _ = env.step(action)
            next_state = next_state[2]
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

    video_recorder.close()
    env.close()

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(all_steps, all_rewards, color='red')
    ax1.plot(all_steps[MOVING_AVERAGE-1:], moving_average(all_rewards, n=MOVING_AVERAGE), color='blue')
    ax1.set_title('Cartpole Q Learning - Rewards over Training')
    ax1.set_xlabel('total_steps')
    ax1.set_ylabel('total_reward')
    ax2.plot(all_steps, all_epsilons, color='red')
    ax2.set_title('Epsilon vs Steps')
    ax2.set_xlabel('total_steps')
    ax2.set_ylabel('epsilon')
    fig.tight_layout()
    fig.savefig('./output/Cartpole_Q_Learning.png')
