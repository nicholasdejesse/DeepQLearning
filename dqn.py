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
    def __init__(self, env, device, network, net_input, net_output,
            vectorized = False,
            memory_capacity = 10_000,
            min_memory_to_train = 100,
            target_net_update = 100,
            eps_start = 1,
            eps_end = 0.01,
            eps_frame_to_end = 10_000
        ):
        self.envs = env
        self.device = device
        self.q_net = network(net_input, net_output).to(device)
        self.target_net = network(net_input, net_output).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.RMSprop(self.q_net.parameters(), lr=LEARNING_RATE)

        self.memory_capacity = memory_capacity
        self.min_memory_to_train = min_memory_to_train
        self.target_net_update = target_net_update
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_frame_to_end = eps_frame_to_end

        self.memory = Memory(memory_capacity)

        self.transforms = None                # Transforms to apply to observation before storing (used in Atari environments)
        self.vectorized = vectorized
        
        self.frames_trained = 0
        self.rewards = []

    def train(self, episodes):
        self.q_net.train()
        self.target_net.train()
        for _ in tqdm(range(episodes)):
            observation, _ = self.envs.reset()
            observation = torch.tensor(observation, device=self.device)
            terminated, truncated = False, False
            reward_this_episode = 0

            while not terminated and not truncated:
                action = self.__select_action(observation).item()
                next_observation, reward, terminated, truncated, _ = self.envs.step(action)

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
                if self.frames_trained % self.target_net_update == 0:
                    self.target_net.load_state_dict(self.q_net.state_dict())

    def train_vector(self, episodes):
        self.q_net.train()
        self.target_net.train()
        episode_count = 0

        observations, _ = self.envs.reset()
        if self.transforms is not None:
            self.__transform_vector_observation(observations)
            observations = torch.tensor(observations, device=self.device)
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

            rewards_per_episode += rewards
            for i in range(self.envs.num_envs):
                if terminations[i] or truncations[i]:
                    self.rewards.append(rewards_per_episode[i])
                    rewards_per_episode[i] = 0

            next_observations = torch.tensor(next_observations, device=self.device)
            # actions = torch.tensor(actions, device=self.device)
            rewards = torch.tensor(rewards, device=self.device)
            terminations = terminations.tolist()
            if self.transforms is None:
                next_observations = torch.tensor(next_observations, device=self.device)

            for i in range(self.envs.num_envs):
                if episode_started[i]:  # True if this is the first frame of the episode
                    episode_count += 1
                    pbar.update(1)
                else:
                    self.memory.append(observations[i], torch.tensor([actions[i]], device=self.device), rewards[i], next_observations[i], torch.tensor(terminations[i], device=self.device))

            observations = next_observations

            self.__optimize()

            episode_started = np.logical_or(terminations, truncations)

            self.frames_trained += 1
            if self.frames_trained % self.target_net_update == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())
        pbar.close()

    def __optimize(self):
        if len(self.memory) < self.min_memory_to_train:
            return
        batch = Transition(*zip(*self.memory.sample(BATCH_SIZE)))

        states = torch.stack(batch.state).squeeze().to(self.device).float()
        actions = torch.stack(batch.action, 0)
        rewards = torch.stack(batch.reward).squeeze()
        next_states = torch.stack(batch.next_state).squeeze().to(self.device).float()
        is_done = torch.stack(batch.done).squeeze()

        # print(f"Input shape: {self.q_net(states).shape}")
        # print(f"Index shape: {actions.shape}")
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
        eps = self.epsilon_schedule(self.frames_trained)
        if r < eps:
            # print(f"Random: {self.env.action_space.sample()}")
            return np.array(self.envs.action_space.sample())
        else:
            with torch.no_grad():
                # # print(observation)
                # print(self.q_net(observation).shape)
                if self.vectorized:
                    # print(f"Not random: {self.q_net(observation).argmax().cpu().numpy()}")
                    return self.q_net(observation.float()).detach().cpu().argmax(axis=1).numpy()
                return torch.argmax(self.q_net(observation))

    def __transform_vector_observation(self, observation):
        for _ in observation:
            for stack in observation:
                for img in stack:
                    img = self.transforms(img)
    
    def epsilon_schedule(self, frame):
        return max(self.eps_end, self.eps_start - frame * (self.eps_start - self.eps_end) / self.eps_frame_to_end)

    # Only used during testing once training is complete
    def evaluate(self, observation):
        return torch.argmax(self.q_net(observation)).item()