import time
import matplotlib.pyplot as plt
import gymnasium as gym
import argparse

import torch
import torch.nn as nn

import dqn

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

device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"

parser = argparse.ArgumentParser(
    prog="DQN Classic Control",
    description="A deep q learning implementation for solving some of the classic control environments (Acrobot, Cart Pole, Mountain Car, Pendulum) in Farama's Gymnasium"
)
parser.add_argument("environment", help="The name of the environment to use.")
parser.add_argument("--train", nargs=2, help="Flag to train the model. Specify the number of episodes to train for and the filename of the model. Will render a graph of rewards after training completes.")
parser.add_argument("--load", help="Loads the model at the given filepath and renders the environment to use it on for 30 episodes.")
args = parser.parse_args()

# Graph epsilon decay
# x_vals = np.linspace(0, 10000, 10000)
# y_vals = EPS_END + (EPS_START - EPS_END) * np.exp(x_vals * -1 / EPS_DECAY)
# plt.plot(y_vals)
# plt.xlabel("Frame")
# plt.ylabel("Epsilon")
# plt.title("Epsilon over frames")
# plt.show()

if args.train:
    print("Beginning training.")
    start = time.time()
    train_env = gym.make(args.environment)
    network = dqn.DeepQNetwork(train_env, device, LinearRelu, train_env.observation_space.shape[0], train_env.action_space.n)
    network.num_context_observations = 1
    network.train(int(args.train[0]))
    train_env.close()
    torch.save(network.q_net.state_dict(), args.train[1])
    end = time.time()
    print(f"Training complete after {int((end - start) // 60)} mins {round((end - start) % 60, 2)} secs.")

    plt.plot(network.rewards)
    plt.title("Rewards over episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()

if args.load:
    human_env = gym.make(args.environment, render_mode="human")
    network = dqn.DeepQNetwork(human_env, device, LinearRelu, train_env.observation_space.shape[0], train_env.action_space.n)
    network.q_net.eval()
    network.q_net.load_state_dict(torch.load(args.load, weights_only=True))
    for _ in range(30):
        observation, _ = human_env.reset()

        terminated, truncated = False, False
        while True:
            action = network.evaluate(torch.tensor(observation, device=device))
            observation, reward, terminated, truncated, _ = human_env.step(action)
            if terminated or truncated:
                break

    human_env.close()