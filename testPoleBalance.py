import gym
import torch
from dqn import DQN

# We don't want randomness when 
# evaluating, so epsilon = 0
def get_action(state, model):
    state = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state)
    return torch.argmax(q_values).item()

# Test Agent
def test_agent(env, model, episodes=5):
    test_rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = get_action(state, model)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        test_rewards.append(total_reward)
        print(f"Test Episode Reward: {total_reward}")

    average_test_rewards = sum(test_rewards) / len(test_rewards)
    print(f"Average reward throughout the test was {average_test_rewards}")

# Load Environment
env = gym.make("CartPole-v1", render_mode="human")
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

# Recreate Model
model_save_path = "dqn_cartpole.pth"
model = DQN(input_dim, output_dim)
model.load_state_dict(torch.load(model_save_path))
model.eval()

test_agent(env, model)

env.close()
