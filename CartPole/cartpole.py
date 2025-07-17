import gymnasium as gym
from NeuralNetwork.network import Network
import numpy as np
import random
import copy

class CartPoleSolver:
    def __init__(self, env):
        network_shape = (4, 8, 4, 2)
        self.q_network = Network(network_shape)
        self.target_network = copy.deepcopy(self.q_network)
        self.env = env

        self.eps = 0.15 # Probability of selecting a random action
        self.target_network_delay = 5
        self.discount = 0.9
        self.memory_batch_size = 1
        self.memory_buffer_length = 100
        self.learning_rate = 0.001
        self.num_actions = 2
    
    def train(self, episodes):
        memory = []
        target_network_delay_count = 0

        for _ in range(episodes):
            observation, info = self.env.reset()
            terminated, truncated = False, False

            while not (terminated or truncated):
                r = random.random()
                action = self.env.action_space.sample() if r < self.eps else np.argmax(self.q_network.result(observation))
                old_observation = observation
                observation, reward, terminated, truncated, info = self.env.step(action)

                memory.append((old_observation, action, reward, observation))
                if len(memory) > self.memory_buffer_length:
                    memory.pop(0)
                batch = random.sample(memory, k=min(self.memory_batch_size, len(memory)))
                targets = []
                for mem in batch:
                    if terminated:
                        targets.append(np.full((self.num_actions, 1), mem[2]))
                    else:
                        targets.append(np.full((self.num_actions, 1), mem[2] + self.discount * np.max(self.target_network.result(mem[0]))))
                # b[0] = observation before taking the action
                self.q_network.train([b[0] for b in batch], targets, len(batch), 1, self.learning_rate)
                target_network_delay_count += 1
                if target_network_delay_count >= self.target_network_delay:
                    self.target_network = copy.deepcopy(self.q_network)
                    target_network_delay_count = 0
    
    def result(self, observation):
        return np.argmax(self.q_network.result(observation))


train_env = gym.make("CartPole-v1")
solver = CartPoleSolver(train_env)
solver.train(2000)
train_env.close()

# Show visual after training complete
human_env = gym.make("CartPole-v1", render_mode="human")
for _ in range(100):
    observation, info = human_env.reset()

    terminated, truncated = False, False
    while not (terminated or truncated):
        action = solver.result(observation)
        observation, reward, terminated, truncated, info = human_env.step(action)

human_env.close()