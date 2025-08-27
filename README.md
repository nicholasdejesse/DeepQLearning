# Deep Q-Learning
## Overview
This project uses Deep Q-Learning to solve some of the RL environments with a discrete action space in Farama's Gymnasium. This includes some classic control problems, like Cart Pole and Mountain Car, as well as some Atari games in the Arcade Learning Environment (ALE), like Breakout.

## Quickstart
### Prerequisites
1. Install requirements via pip.
    ```bash
    pip install -r requirements.txt
    ```

### Classic Control Environments

1. To train a model for a classic control environment, run ```classic_control.py``` and use the ```--train``` flag. Specify both the number of episodes to train for and the name of the output file.

    ```bash
    python classic_control.py --train EPISODES FILE_PATH.pt
    ```
2. Then, to visualize the model's performance, use the ```--load``` flag, specifying the path to the model you trained.

    ```bash
    python classic_control.py --load FILE_PATH.pt
    ```

    This will create a visualization of the model in the environment for 10 episodes, which resets after each episode terminates or is truncated.

### Atari Environments