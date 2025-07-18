# Balancing a Pole
A cart agent learns to balance a pole through reinforcement learning.

## Installation
Fork this repository and create a venv. Once in the venv, use the command: 

    pip install -r /path/to/requirements.txt 

to install all the necessary libraries.  
This project was created using python 3.11, so make sure the venv is using the correct python version.

## Usage
**trainPoleBalance.py:** is used to train the agent. Run this file to train an agent.  
    At the end of the training, you will be prompted in the command line to either save  
    or discard the training.  
**testPoleBalance.py:** is used to test the trained agent. This file runs the pygame visualization.  
**dqn.py:** contains the Deep Q-Network (DQN) that the agent uses during training and testing.  
**dqn_cartpole.pth** contains the saved model parameters. Populates when saving a trained agent and is loaded when testing a saved agent.

## Screenshots
**Training Example**  
![Alt text](pole-balance-rl\screenshots\training-example.jpg?raw=true "Training-Example")

**Test Example**  
![Alt text](pole-balance-rl\screenshots\test-example.jpg?raw=true "Test-Example")

![Alt text](pole-balance-rl\screenshots\test-example-pygame.jpg?raw=true "Test-Example-pygame")