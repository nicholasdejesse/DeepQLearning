import gymnasium as gym
from NeuralNetwork.network import Network
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import math

class CartPoleSolver:
    def __init__(self, env):
        network_shape = (4, 64, 64, 2)
        self.q_network = Network(network_shape)
        self.target_network = copy.deepcopy(self.q_network)
        self.env = env

        self.eps_start = 1 # Probability of selecting a random action at the start
        self.eps_end = 0.15   # Probability of selecting a random action at the end
        self.eps_decay = 3000  # Rate of epsilon decay

        self.target_network_delay = 1000 # How long until the target network's weights gets reset to the Q network's weights
        self.discount = 0.99 # Discount factor
        self.memory_batch_size = 32
        self.memory_buffer_length = 10000
        self.learning_rate = 0.01
        self.num_actions = 2
    
    def train(self, episodes):
        self.loss = []
        self.rewards = []
        memory = []
        target_network_delay_count = 0
        frame_count = 0

        for _ in range(episodes):
            observation, info = self.env.reset()
            terminated, truncated = False, False
            reward_this_episode = 0

            while not (terminated or truncated):
                r = random.random()
                eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1 * frame_count / self.eps_decay)
                action = self.env.action_space.sample() if r < eps else np.argmax(self.q_network.result(observation))
                old_observation = observation
                observation, reward, terminated, truncated, info = self.env.step(action)
                frame_count += 1

                reward_this_episode += reward
                if terminated or truncated:
                    self.rewards.append(reward_this_episode)
                    reward_this_episode = 0

                memory.append((old_observation, action, reward, observation, terminated, truncated))
                if len(memory) > self.memory_buffer_length:
                    memory.pop(0)
                batch = random.sample(memory, k=min(self.memory_batch_size, len(memory)))

                input_observations = []
                targets = []
                actions = []
                for mem in batch:
                    input_observations.append(mem[0])
                    actions.append(mem[1])
                    target_output = np.zeros((self.num_actions, 1))
                    if mem[4] or mem[5]: # If memory terminates or truncates
                        target_output[mem[1]] = mem[2]
                    else:
                        target_output[mem[1]] = mem[2] + self.discount * np.max(self.target_network.result(mem[3]))
                    targets.append(target_output)

                self.q_network.train(input_observations, targets, len(batch), 1, self.learning_rate, actions, self.loss)

                target_network_delay_count += 1
                if target_network_delay_count >= self.target_network_delay:
                    self.target_network = copy.deepcopy(self.q_network)
                    target_network_delay_count = 0
        
    def result(self, observation):
        return np.argmax(self.q_network.result(observation))


train_env = gym.make("CartPole-v1")
solver = CartPoleSolver(train_env)
solver.train(10000)
train_env.close()

plt.plot(solver.loss)
plt.title("Loss over frames")
plt.ylabel("Loss")
plt.show()

plt.plot(solver.rewards)
plt.title("Rewards over episodes")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()

# Show visual after training complete
human_env = gym.make("CartPole-v1", render_mode="human")
for _ in range(100):
    observation, info = human_env.reset()

    terminated, truncated = False, False
    while True:
        action = solver.result(observation)
        observation, reward, terminated, truncated, info = human_env.step(action)
        if terminated:
            break
        if truncated:
            print("Episode truncated")
            break

human_env.close()