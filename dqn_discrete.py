# Some sections inspired by code found here: https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#dqn-algorithm

from collections import namedtuple
from collections import deque
import random
import gymnasium as gym
import math
import matplotlib.pyplot as plt
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
        self.mem.append(Transition(*args))

    def sample(self, k):
        return random.sample(self.mem, k)
    
    def __len__(self):
        return len(self.mem)

class LinearRelu(nn.Module):
    def __init__(self, input_features, actions):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, actions)
        )
    
    def forward(self, x):
        return self.linear_relu_stack(x)

class DeepQNetwork:
    def __init__(self, env, device):
        self.env = env
        self.device = device
        self.q_net = LinearRelu(env.observation_space.shape[0], env.action_space.n).to(device)
        self.target_net = LinearRelu(env.observation_space.shape[0], env.action_space.n).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)
        self.memory = Memory(MEMORY_CAPACITY)

        self.frames_trained = 0
        self.rewards = []

    def train(self, episodes):
        self.q_net.train()
        self.target_net.train()
        for _ in range(episodes):
            observation, _ = self.env.reset()
            observation = torch.tensor(observation, device=self.device)
            terminated, truncated = False, False
            reward_this_episode = 0

            while not terminated and not truncated:
                action = self.__select_action(observation).item()
                next_observation, reward, terminated, truncated, _ = self.env.step(action)

                reward_this_episode += reward
                if terminated or truncated:
                    self.rewards.append(reward_this_episode)
                    reward_this_episode = 0

                action = torch.tensor([action], device=self.device)
                reward = torch.tensor([reward], device=self.device)
                next_observation = torch.tensor(next_observation, device=self.device)
                done = torch.tensor([terminated], device=self.device, dtype=torch.bool)

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

        states = torch.stack(batch.state)
        actions = torch.stack(batch.action)
        rewards = torch.stack(batch.reward).squeeze()
        next_states = torch.stack(batch.next_state)
        is_done = torch.stack(batch.done).squeeze()

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
                return torch.argmax(self.q_net(observation))

    # Only used during testing once training is complete
    def evaluate(self, observation):
        return torch.argmax(self.q_net(observation)).item()

device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"

parser = argparse.ArgumentParser(
    prog="Deep Q Learning",
    description="A deep q learning implementation for solving some environments in Farama's Gymnasium"
)
parser.add_argument("environment", help="The name of the environment to use.")
parser.add_argument("--train", nargs=2, help="Flag to train the model. Specify the number of episodes to train for and the filename of the model. Will render a graph of rewards after training completes.")
parser.add_argument("--load", help="Loads the model at the given filepath and renders the environment to use it on for 10 episodes.")
args = parser.parse_args()

if args.train:
    print("Beginning training.")
    train_env = gym.make(args.environment)
    network = DeepQNetwork(train_env, device)
    network.train(int(args.train[0]))
    train_env.close()
    torch.save(network.q_net.state_dict(), args.train[1])
    print("Training complete.")

    plt.plot(network.rewards)
    plt.title("Rewards over episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()

if args.load:
    human_env = gym.make(args.environment, render_mode="human")
    network = DeepQNetwork(human_env, device)
    network.q_net.eval()
    network.q_net.load_state_dict(torch.load(args.load, weights_only=True))
    for _ in range(10):
        observation, info = human_env.reset()

        terminated, truncated = False, False
        while True:
            action = network.evaluate(torch.tensor(observation, device=device))
            observation, reward, terminated, truncated, info = human_env.step(action)
            if terminated or truncated:
                break

    human_env.close()