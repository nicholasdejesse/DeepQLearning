import gymnasium as gym
from NeuralNetwork.network import Network

class CartPoleSolver:
    def __init__(self, env):
        self.q_function = Network((4, 8, 8, 2))
        self.env = env
        self.memory = []
    
    def train(self, episodes):
        for _ in episodes:
            observation, info = env.reset()
            terminated, truncated = False

            while not terminated or truncated:
                action = self.q_function.result(observation)
                observation, reward, terminated, truncated, info = env.step(action)



env = gym.make("CartPole-v1", render_mode="human")
solver = CartPoleSolver(env)

solver.train(10)
env.close()