import gymnasium as gym
from NeuralNetwork.network import Network

class CartPoleSolver:
    def __init__(self, env):
        self.q_function = Network((4, 8, 8, 2))
        self.env = env
        self.memory = []
    
    def train(self, episodes):
        pass


env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()