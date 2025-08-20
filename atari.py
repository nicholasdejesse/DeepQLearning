import gymnasium as gym
import ale_py
from ale_py.vector_env import AtariVectorEnv
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
from collections import deque
from dqn import FrameSkipper
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2

import dqn

# Hyperparameters
FRAME_SKIPPING = 4       # Select actions every k frames instead of every frame
NUM_CONTEXT_FRAMES = 4
ACTION_SPACE = 4         # TODO: Change this to read from the environment instead

class ConvNet(nn.Module):
    def __init__(self, num_frames, actions):
        super().__init__()
        self.conv1 = nn.Conv2d(num_frames, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 81, 256)
        self.fc2 = nn.Linear(256, actions)
        # self.conv_stack = nn.Sequential(
        #     nn.Conv2d(num_frames, 16, kernel_size=8, stride=4),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(81, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, actions)
        # )
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32 * 81)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    gym.register_envs(ale_py)

    # Preprocessing steps to do on observations
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        # v2.Grayscale(),
        # v2.Resize((110, 84)),
        # lambda img: v2.functional.crop(img, 16, 0, 84, 84) # Crop down to 84 by 84 playing area
    ])

    # Test transforms
    # test_env = gym.make("ALE/Breakout-v5")
    # observation, _ = test_env.reset()
    # print(observation.shape)
    # plt.imshow(observation)
    # plt.show()
    # observation = transforms(observation)
    # observation = np.moveaxis(observation.numpy() * 255, 0, -1)
    # plt.imshow(observation)
    # plt.show()

    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"

    parser = argparse.ArgumentParser(
        prog="DQN Atari",
        description="A deep q learning implementation for solving some of the Atari environments in Farama's Gymnasium"
    )
    parser.add_argument("environment", help="The name of the environment to use.")
    parser.add_argument("--train", nargs=2, help="Flag to train the model. Specify the number of episodes to train for and the filename of the model. Will render a graph of rewards after training completes.")
    parser.add_argument("--load", help="Loads the model at the given filepath and renders the environment to use it on for 30 episodes.")
    args = parser.parse_args()

    if args.train:
        print("Beginning training.")
        start = time.time()
        envs = AtariVectorEnv(
            game=args.environment,
            num_envs=8,

            frameskip=4,
            stack_num=NUM_CONTEXT_FRAMES,
            img_height=84,
            img_width=84,
            grayscale=True,
            autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP
        )
        network = dqn.DeepQNetwork(envs, device, ConvNet, 4, ACTION_SPACE, vectorized=True)
        network.frame_skipping = 4
        network.transforms = transforms

        network.train_vector(int(args.train[0]))

        envs.close()
        torch.save(network.q_net.state_dict(), args.train[1])
        end = time.time()
        print(f"Training complete after {int((end - start) // 60)} mins {round((end - start) % 60, 2)} secs.")
        print(f"Trained for {network.frames_trained} frames.")

        plt.plot(network.rewards)
        plt.title("Rewards over episodes")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.show()

    if args.load:
        human_env = gym.make(args.environment, render_mode="human")
        human_env = GrayscaleObservation(ResizeObservation(human_env, (84, 84)))

        network = dqn.DeepQNetwork(human_env, device, ConvNet, 4, human_env.action_space.n, 4)
        # network.frame_skipping = 4
        network.q_net.eval()
        network.q_net.load_state_dict(torch.load(args.load, weights_only=True))
        context = deque(maxlen=NUM_CONTEXT_FRAMES)
        # frame_skipper = dqn.FrameSkipper(FRAME_SKIPPING)
        for _ in range(30):
            count = 0
            observation, _ = human_env.reset()
            observation = transforms(observation)
            context.clear()
            # frame_skipper.reset()
            for _ in range(NUM_CONTEXT_FRAMES):
                context.append(transforms(observation))     # Fill queue with initial observation

            terminated, truncated = False, False
            while True:
                action = network.evaluate(torch.stack(tuple(context)).squeeze().to(device=device))
                print(action)
                # action = frame_skipper.get_action(action)
                observation, reward, terminated, truncated, _ = human_env.step(action)
                observation = transforms(observation)
                context.append(transforms(observation))
                if terminated or truncated:
                    break

        human_env.close()