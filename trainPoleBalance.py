import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from dqn import DQN

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

# Set Random Seeds
SEED = 24
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Initialize the Environment
env = gym.make("CartPole-v1")
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

# Networks and Optimizer
q_network = DQN(input_dim, output_dim)
target_network = DQN(input_dim, output_dim)
target_network.load_state_dict(q_network.state_dict())

optimizer = optim.Adam(q_network.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Hyperparameters
num_episodes = 2000
batch_size = 64
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
target_update_freq = 10

# Replay Buffer
memory = deque(maxlen=10000)

# Early Stopping
reward_history = []
REWARD_THRESHOLD = 475
PATIENCE = 100

for episode in range(num_episodes):
    state, _ = env.reset(seed=SEED)
    done = False
    total_reward = 0

    while not done:
        action = get_action(state, epsilon, q_network, env.action_space)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        # Early Stopping
        reward_history.append(total_reward)
        if len(reward_history) >= PATIENCE:
            average_reward = np.mean(reward_history[-PATIENCE:])
            if average_reward >= REWARD_THRESHOLD:
                print(f"Early Stopping Triggered: Average Reward is {average_reward}")
                break


        # Train the Network
        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)

            # Current Q Values
            q_values = q_network(states).gather(1, actions).squeeze()

            # Target Q Values
            with torch.no_grad():
                max_next_q = target_network(next_states).max(1)[0]
                targets = rewards + gamma * max_next_q * (1-dones)

            loss = loss_fn(q_values, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Decay Epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Update target netwrok
    if episode % target_update_freq == 0:
        target_network.load_state_dict(q_network.state_dict())

    print(f"Episode {episode}: {total_reward=} | {epsilon=:.3f}")

env.close()

model_save_path = "dqn_cartpole.pth"
try:
    if input("Enter 0 to discard this training.\nEnter 1 to save this training.\n") == 1:
        torch.save(q_network.state_dict(), model_save_path)
        print(f"Training saved into {model_save_path}")
except Exception as e:
    print("Please enter 0 or 1 next time. Thank you.")