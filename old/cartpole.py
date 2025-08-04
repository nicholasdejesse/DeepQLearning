import gymnasium as gym
from collections import deque
from network import Network
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import math

class CartPoleSolver:
    def __init__(self, env):
        network_shape = (4, 128, 128, 2)
        self.q_network = Network(network_shape)
        self.target_network = Network(network_shape)
        self.target_network.load_weights_and_biases(self.q_network.copy_weights_and_biases())
        self.env = env

        self.eps_start = 0.9 # Probability of selecting a random action at the start
        self.eps_end = 0.01   # Probability of selecting a random action at the end
        self.eps_decay = 10000  # Rate of epsilon decay

        self.target_network_delay = 500 # How long until the target network's weights gets reset to the Q network's weights
        self.discount = 0.99 # Discount factor
        self.memory_batch_size = 32
        self.memory_buffer_length = 10000
        self.learning_rate = 0.001
        self.num_actions = 2
    
    def train(self, episodes):
        self.loss = []
        self.rewards = []
        memory = deque(maxlen=self.memory_buffer_length)
        target_network_delay_count = 0
        frame_count = 0

        for e in range(episodes):
            observation, _ = self.env.reset()
            terminated, truncated = False, False
            reward_this_episode = 0

            while not (terminated or truncated):
                r = random.random()
                eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1 * frame_count / self.eps_decay)
                action = self.env.action_space.sample() if r < eps else np.argmax(self.q_network.result(observation))
                old_observation = np.copy(observation)
                observation, reward, terminated, truncated, _ = self.env.step(action)
                frame_count += 1

                reward_this_episode += reward
                if terminated or truncated:
                    self.rewards.append(reward_this_episode)
                    # print(f"Episode: {e}, Reward: {reward_this_episode}, Epsilon: {eps}")
                    reward_this_episode = 0

                memory.append((old_observation, action, reward, observation, terminated, truncated))

                if len(memory) > self.memory_batch_size:
                    batch = random.sample(memory, k=self.memory_batch_size)
                    input_observations = []
                    targets = []
                    for mem in batch:
                        input_observations.append(np.copy(mem[0]))
                        target_output = self.target_network.result(mem[0])
                        if mem[4] or mem[5]: # If memory terminates or truncates
                            target_output[int(mem[1])] = mem[2]
                        else:
                            target_output[int(mem[1])] = mem[2] + self.discount * np.max(self.target_network.result(mem[3]))
                        targets.append(target_output)

                    before = self.q_network.copy_weights_and_biases()
                    self.q_network.train(input_observations, targets, len(batch), 1, self.learning_rate, self.loss)
                    after = self.q_network.copy_weights_and_biases()

                    target_network_delay_count += 1
                    if target_network_delay_count >= self.target_network_delay:
                        self.target_network.load_weights_and_biases(self.q_network.copy_weights_and_biases())
                        target_network_delay_count = 0
        
    def result(self, observation):
        return np.argmax(self.q_network.result(observation))

random_seed = 10
random.seed(random_seed)
np.random.seed(random_seed)

train_env = gym.make("CartPole-v1")
train_env.action_space.seed(random_seed)
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
human_env.action_space.seed(random_seed)
for _ in range(5):
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