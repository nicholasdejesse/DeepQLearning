# Some sections inspired by code found here: https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#dqn-algorithm

from collections import namedtuple
from collections import deque
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
DISCOUNT = 0.99
MEMORY_CAPACITY = 100000
TARGET_NET_UPDATE = 1000

EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 2500

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

class Memory:
    def __init__(self, capacity):
        self.mem = deque(maxlen=capacity)

    def append(self, *args):
        self.mem.append(Transition(*args))

    def sample(self, k):
        return random.sample(self.mem, k)
    
    def __len__(self):
        return len(self.mem)

class DeepQNetwork:
    def __init__(self, env, device, network, net_input, net_output, num_context_observations = 1):
        self.env = env
        self.device = device
        self.q_net = network(net_input, net_output).to(device)
        self.target_net = network(net_input, net_output).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)
        self.memory = Memory(MEMORY_CAPACITY)

        self.num_context_observations = num_context_observations     # Number of consecutive observations to consider. Default = 1 (only consider the current observation)
        self.context = deque(maxlen=self.num_context_observations)
        self.transforms = None                # Transforms to apply to observation before storing (used in Atari environments)
        self.frame_skipping = 1

        self.frames_trained = 0
        self.rewards = []

    def train(self, episodes):
        self.q_net.train()
        self.target_net.train()
        frame_skip = FrameSkipper(self.frame_skipping)
        for _ in tqdm(range(episodes)):
            frame_skip.reset()
            observation, _ = self.env.reset()
            if self.transforms is not None:
                observation = self.transforms(observation)
            else:
                observation = torch.tensor(observation, device=self.device)

            if self.num_context_observations > 1:
                self.context.clear()
                for _ in range(self.num_context_observations):
                    self.context.append(observation.detach().clone())     # Fill queue with initial observation

            terminated, truncated = False, False
            reward_this_episode = 0

            while not terminated and not truncated:
                if self.num_context_observations > 1:
                    action = self.__select_action(torch.stack(tuple(self.context)).squeeze().to(self.device)).item()
                else:
                    action = self.__select_action(observation).item()
                action = frame_skip.get_action(action)

                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                if self.transforms is not None:
                    next_observation = self.transforms(next_observation)


                reward_this_episode += reward
                if terminated or truncated:
                    self.rewards.append(reward_this_episode)
                    reward_this_episode = 0

                action = torch.tensor([action], device=self.device)
                reward = torch.tensor([reward], device=self.device)
                if self.transforms is None:
                    next_observation = torch.tensor(next_observation, device=self.device)
                done = torch.tensor([terminated], device=self.device, dtype=torch.bool)

                if self.num_context_observations > 1:
                    old_context = torch.stack(tuple(self.context))
                    self.context.append(next_observation)
                    next_context = torch.stack(tuple(self.context))
                    self.memory.append(old_context, action, reward, next_context, done)
                else:
                    self.memory.append(observation, action, reward, next_observation, done)

                observation = next_observation

                self.__optimize()

                self.frames_trained += 1
                if self.frames_trained % TARGET_NET_UPDATE == 0:
                    self.target_net.load_state_dict(self.q_net.state_dict())

    def __optimize(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = Transition(*zip(*self.memory.sample(BATCH_SIZE)))

        states = torch.stack(batch.state).squeeze().to(self.device)
        actions = torch.stack(batch.action)
        rewards = torch.stack(batch.reward).squeeze()
        next_states = torch.stack(batch.next_state).squeeze().to(self.device)
        is_done = torch.stack(batch.done).squeeze()

        # print(f"State shape: {self.q_net(states).shape}")
        state_action_output = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            next_state_action_max = self.target_net(next_states).max(1).values
        
        next_state_action_max = rewards + (~is_done) * DISCOUNT * next_state_action_max
                
        loss = nn.MSELoss()(state_action_output, next_state_action_max.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
                
    def __select_action(self, observation):
        r = random.random()
        eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * self.frames_trained / EPS_DECAY)
        if r < eps:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                # # print(observation)
                # print(self.q_net(observation).shape)
                return torch.argmax(self.q_net(observation))

    # Only used during testing once training is complete
    def evaluate(self, observation):
        return torch.argmax(self.q_net(observation)).item()
    
class FrameSkipper:
    def __init__(self, k):
        self.k = k
        self.count = 0
        self.previous_action = None

    def reset(self):
        self.count = 0
        self.previous_action = None

    # Returns either the action passed in or the previous action depending on the count
    def get_action(self, action):
        if self.count % self.k == 0:
            self.previous_action = action
            self.count = 0
            res = action
        else:
            res = self.previous_action
        self.count += 1
        return res