import gymnasium as gym
import ale_py
from ale_py.vector_env import AtariVectorEnv
import matplotlib.pyplot as plt
import argparse
import time
from gymnasium.wrappers import FrameStackObservation, AtariPreprocessing, RecordVideo

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2

import dqn

# Hyperparameters
NUM_STACK_FRAMES = 4
MEMORY_CAPACITY = 1_000_000     # Max number of experiences to store
MIN_MEMORY_TO_TRAIN = 50_000    # Minimum required experiences before sampling and training from memory
TARGET_NET_UPDATE = 1_000

EPS_START = 1
EPS_END = 0.1
EPS_FRAME_TO_END = 1_000_000

class ConvNet(nn.Module):
    def __init__(self, num_frames, actions):
        super().__init__()
        self.conv1 = nn.Conv2d(num_frames, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(32 * 81, 256)
        self.fc2 = nn.Linear(256, actions)
    
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
        v2.ToDtype(torch.uint8, scale=True),
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
    parser.add_argument("--load", help="Loads the model at the given filepath and renders the environment to use it on for 10 episodes.")
    parser.add_argument("--record", help="Loads the model at the given filepath and records a video of one episode.")
    args = parser.parse_args()

    if args.train:
        print("Beginning training.")
        start = time.time()
        envs = AtariVectorEnv(
            game=args.environment,
            num_envs=8,

            frameskip=4,
            stack_num=NUM_STACK_FRAMES,
            img_height=84,
            img_width=84,
            grayscale=True,
            autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP
        )
        network = dqn.DeepQNetwork(
            env=envs,
            device=device,
            network=ConvNet,
            net_input=NUM_STACK_FRAMES,
            net_output=envs.single_action_space.n,
            vectorized=True,

            memory_capacity = 1_000_000,
            min_memory_to_train = 50_000,
            target_net_update = 1_000,

            eps_start = 1,
            eps_end = 0.1,
            eps_frame_to_end = 1_000_000,
        )
        
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

    elif args.load or args.record:
        if args.load:
            num_episodes = 10
            env = FrameStackObservation(
                AtariPreprocessing(
                    gym.make(args.environment, render_mode="human", frameskip=1),
                ),
                stack_size=NUM_STACK_FRAMES,
                padding_type="zero"
            )
        else:
            num_episodes = 1
            env = RecordVideo(
                FrameStackObservation(
                    AtariPreprocessing(
                        gym.make(args.environment, render_mode="rgb_array", frameskip=1),
                    ),
                    stack_size=NUM_STACK_FRAMES,
                    padding_type="zero"
                ),
                video_folder="videos",
                name_prefix="video-"
            )

        network = dqn.DeepQNetwork(
            env=env,
            device=device,
            network=ConvNet,
            net_input=NUM_STACK_FRAMES,
            net_output=env.single_action_space.n,
            vectorized=True,

            memory_capacity = 1_000_000,
            min_memory_to_train = 50_000,
            target_net_update = 10_000,

            eps_start = 1,
            eps_end = 0.1,
            eps_frame_to_end = 200_000,
        )
        network.q_net.eval()
        network.q_net.load_state_dict(torch.load(args.load if args.load is not None else args.record, weights_only=True))

        for e in range(num_episodes):
            total_episode_reward = 0
            observation, _ = env.reset()
            for _ in observation:
                for stack in observation:
                    for img in stack:
                        img = transforms(img)
            observation = torch.tensor(observation, dtype=torch.float)
            terminated, truncated = False, False
            while True:
                action = network.evaluate(observation.to(device=device))
                observation, reward, terminated, truncated, _ = env.step(action)
                total_episode_reward += reward
                for _ in observation:
                    for stack in observation:
                        for img in stack:
                            img = transforms(img)
                observation = torch.tensor(observation, dtype=torch.float)
                            
                if terminated or truncated:
                    print(f"Episode {e} finished with total reward of {total_episode_reward}")
                    break
        env.close()