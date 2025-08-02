from collections import namedtuple
from collections import deque
import random
import gymnasium as gym
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

ENV_ID = "CartPole-v1"

BATCH_SIZE = 32
LEARNING_RATE = 0.0001
DISCOUNT = 0.99
MEMORY_CAPACITY = 10000
TARGET_NET_UPDATE = 100

EPS_START = 1
EPS_END = 0.01
EPS_DECAY = 2500

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

class Memory:
    def __init__(self, capacity):
        self.mem = deque(maxlen=capacity)

    def append(self, *args):
        self.mem.append(Transition(args))

    def sample(self, k):
        return random.sample(self.mem, k)

class DeepQNetwork(nn.Module):
    def __init__(self, input_features, actions):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, actions)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return torch.argmax(self.linear_relu_stack(x))

class Solver:
    def __init__(self, env, device):
        self.env = env
        self.q_net = DeepQNetwork(train_env.observation_space.shape[0], train_env.action_space.n).to(device)
        self.target_net = DeepQNetwork(train_env.observation_space.shape[0], train_env.action_space.n).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.SGD(self.q_net.parameters(), lr=LEARNING_RATE)
        self.memory = Memory(MEMORY_CAPACITY)

        self.frames_trained = 0
        self.rewards = []

    def train(self, episodes):
        for e in range(episodes):
            observation, _ = self.env.reset()
            terminated, truncated = False, False
            reward_this_episode = 0

            while not terminated and not truncated:
                action = self.__select_action(observation)
                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                self.memory.append(observation, action, reward, next_observation, terminated or truncated)

                observation = next_observation

                reward_this_episode += reward
                if terminated or truncated:
                    self.rewards.append(reward_this_episode)
                    reward_this_episode = 0

                self.__optimize()

                self.frames_trained += 1
                if self.frames_trained % TARGET_NET_UPDATE == 0:
                    self.target_net.load_state_dict(self.q_net.state_dict())

    def __optimize(self):
        samples = self.memory.sample(BATCH_SIZE)

        states = torch.Tensor([s.state for s in samples])

        state_action_output = self.q_net(states)

        loss = F.cross_entropy(state_action_output, state_action_expected)

        self.q_net.zero_grad()
        loss.backward()
        self.optimizer.step()
                
    def __select_action(self, observation):
        r = random.random()
        eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * self.frames_trained / EPS_DECAY)
        if r < eps:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                return self.q_net(observation)


device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"
print(f"Using {device}")

train_env = gym.make(ENV_ID)
solver = Solver(train_env, device)
solver.train(1000)