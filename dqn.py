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
    def __init__(self, env, device, network, net_input, net_output, vectorized = False):
        self.envs = env
        self.device = device
        self.q_net = network(net_input, net_output).to(device)
        self.target_net = network(net_input, net_output).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)
        self.memory = Memory(MEMORY_CAPACITY)

        self.transforms = None                # Transforms to apply to observation before storing (used in Atari environments)
        self.vectorized = vectorized
        
        self.frames_trained = 0
        self.rewards = []

    def train_vector(self, episodes):
        self.q_net.train()
        self.target_net.train()
        episode_count = 0

        observations, _ = self.envs.reset()
        if self.transforms is not None:
            self.__transform_vector_observation(observations)
            observations = torch.tensor(observations, device=self.device).float()
        else:
            observations = torch.tensor(observations, device=self.device)

        rewards_per_episode = np.zeros(self.envs.num_envs)
        episode_started = np.zeros(self.envs.num_envs, dtype=bool)

        pbar = tqdm(total=episodes)
        while episode_count < episodes:
            actions = self.__select_action(observations)

            next_observations, rewards, terminations, truncations, _ = self.envs.step(actions)

            if self.transforms is not None:
                self.__transform_vector_observation(next_observations)

            # rewards_per_episode += rewards
            # if terminations or truncations:
            #     self.rewards.append(rewards_per_episode)
            #     rewards_per_episode = 0
            next_observations = torch.tensor(next_observations, device=self.device).float()
            actions = torch.tensor(np.array(actions), device=self.device)
            rewards = torch.tensor(np.array(rewards), device=self.device)
            if self.transforms is None:
                next_observations = torch.tensor(next_observations, device=self.device)

            for i in range(self.envs.env_num):
                if episode_started[i]:  # True if this is the first frame of the episode
                    episode_count += 1
                    pbar.update(1)
                else:
                    self.memory.append(observations[i], actions[i], rewards[i], next_observations[i], terminations[i])

            observations = next_observations

            self.__optimize()

            episode_started = np.logical_or(terminations, truncations)

            self.frames_trained += 1
            if self.frames_trained % TARGET_NET_UPDATE == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())
        pbar.close()

    def __optimize(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = Transition(*zip(*self.memory.sample(BATCH_SIZE)))

        states = torch.stack(batch.state).squeeze().to(self.device).float()
        actions = torch.stack(batch.action)
        rewards = torch.stack(batch.reward).squeeze()
        next_states = torch.stack(batch.next_state).squeeze().to(self.device).float()
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
            # print(f"Random: {self.env.action_space.sample()}")
            return np.array(self.envs.action_space.sample())
        else:
            with torch.no_grad():
                # # print(observation)
                # print(self.q_net(observation).shape)
                if self.vectorized:
                    # print(f"Not random: {self.q_net(observation).argmax().cpu().numpy()}")
                    return self.q_net(observation).detach().cpu().argmax().unsqueeze(0).numpy()
                return torch.argmax(self.q_net(observation))

    def __transform_vector_observation(self, observation):
        for env in observation:
            for stack in observation:
                for img in stack:
                    img = self.transforms(img)
        
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