import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# ========================
# 超参数设置
# ========================
HYPERPARAMS = {
    "env_name": "CartPole-v1",      # 环境名称
    "episodes": 500,               # 训练轮数
    "max_timesteps": 200,          # 每轮最大时间步
    "batch_size": 64,              # 批量大小
    "learning_rate": 1e-3,         # 学习率
    "discount": 0.99,              # 折扣因子
    "epsilon_start": 1.0,          # 初始探索率
    "epsilon_end": 0.01,           # 最小探索率
    "epsilon_decay": 0.995,        # 探索率衰减系数
    "buffer_size": 100000,         # 经验回放缓存大小
    "target_update_freq": 10,      # 目标网络更新频率
}

# ========================
# Q-Network 定义
# ========================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ========================
# 经验回放缓冲区
# ========================
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float),
            torch.tensor(actions, dtype=torch.long).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float).unsqueeze(1),
            torch.tensor(next_states, dtype=torch.float),
            torch.tensor(dones, dtype=torch.float).unsqueeze(1),
        )

    def size(self):
        return len(self.buffer)

# ========================
# DQN Agent 定义
# ========================
class DQNAgent:
    def __init__(self, state_dim, action_dim, hyperparams):
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_q_network = QNetwork(state_dim, action_dim)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=hyperparams["learning_rate"])

        self.replay_buffer = ReplayBuffer(hyperparams["buffer_size"])
        self.discount = hyperparams["discount"]
        self.epsilon = hyperparams["epsilon_start"]
        self.epsilon_end = hyperparams["epsilon_end"]
        self.epsilon_decay = hyperparams["epsilon_decay"]
        self.target_update_freq = hyperparams["target_update_freq"]
        self.action_dim = action_dim
        self.total_steps = 0

    def select_action(self, state, train=True):
        if train and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            with torch.no_grad():
                return torch.argmax(self.q_network(state)).item()

    def train(self, batch_size):
        if self.replay_buffer.size() < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Q值更新
        current_q_values = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            max_next_q_values = self.target_q_network(next_states).max(dim=1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * self.discount * max_next_q_values

        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 衰减 epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # 更新目标网络
        if self.total_steps % self.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        self.total_steps += 1

# ========================
# 训练和推理函数
# ========================
def train_dqn(hyperparams):
    env = gym.make(hyperparams["env_name"])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim, hyperparams)
    rewards = []

    for episode in range(hyperparams["episodes"]):
        state = env.reset()
        episode_reward = 0

        for t in range(hyperparams["max_timesteps"]):
            action = agent.select_action(state, train=True)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.add(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            agent.train(hyperparams["batch_size"])

            if done:
                break

        rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Epsilon = {agent.epsilon:.2f}")

    return agent

def test_dqn(agent, hyperparams):
    env = gym.make(hyperparams["env_name"])
    for episode in range(hyperparams["test_episodes"]):
        state = env.reset()
        episode_reward = 0

        for t in range(hyperparams["max_timesteps"]):
            action = agent.select_action(state, train=False)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_reward += reward

            if done:
                break

        print(f"Test Episode {episode + 1}: Reward = {episode_reward:.2f}")

# ========================
# 主函数调用
# ========================
if __name__ == "__main__":
    dqn_agent = train_dqn(HYPERPARAMS)
    test_dqn(dqn_agent, HYPERPARAMS)
