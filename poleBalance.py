import gym
import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_layer = nn.Linear(input_dim, 128)
        self.act1 = nn.ReLU()
        self.output_layer = nn.Linear(128, output_dim)

    def forward(self, x):
        return self.output_layer(self.act1(self.input_layer(x)))

# Epsilon-greedy policy
#  A random action is chosen epsilon % of the time
#  E.g. if epsilon = 0.2, a random action is chosen
#       20% of the time.
def get_action(state, epsilon, q_network, action_space):
    if random.random() < epsilon:
        return action_space.sample()
    state = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        q_values = q_network(state)
    return torch.argmax(q_values).item()

env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset()
print("Initial Observation: ", observation)

# Run an epsisode
done = False
total_reward = 0

while not done:
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward

    print(f"Observation: {observation}\nReward: {reward}\nTerminated: {terminated}\nTruncated: {truncated}\nInfo: {info}")

    done = terminated or truncated

env.close()
print("Total reward collected = ", total_reward)