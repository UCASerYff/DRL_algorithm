import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# ========================
# 超参数设置
# ========================
HYPERPARAMS = {
    # 环境参数
    "env_name": "Pendulum-v1",       # 环境名称
    "episodes": 300,                # 训练轮数
    "max_timesteps": 200,           # 每轮最大时间步数
    "test_episodes": 5,             # 测试轮数

    # 网络与优化参数
    "actor_lr": 3e-4,               # Actor 学习率
    "critic_lr": 3e-4,              # Critic 学习率
    "alpha_lr": 3e-4,               # 温度参数学习率
    "batch_size": 64,               # 批量大小
    "discount": 0.99,               # 折扣因子
    "tau": 0.005,                   # 目标网络软更新系数

    # 经验回放
    "buffer_size": 1000000,         # 回放缓存大小
}

# ========================
# Actor 网络（使用高斯策略）
# ========================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)  # 限制标准差的范围
        return mean, log_std

    def sample_action(self, state):
        mean, log_std = self(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # 重新参数化采样
        action = torch.tanh(x_t) * self.max_action
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=1, keepdim=True)

# ========================
# Critic 网络
# ========================
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 网络
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        # Q2 网络
        self.fc4 = nn.Linear(state_dim + action_dim, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)

        # Q1
        q1 = torch.relu(self.fc1(sa))
        q1 = torch.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        # Q2
        q2 = torch.relu(self.fc4(sa))
        q2 = torch.relu(self.fc5(q2))
        q2 = self.fc6(q2)

        return q1, q2

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
            torch.tensor(actions, dtype=torch.float),
            torch.tensor(rewards, dtype=torch.float).unsqueeze(1),
            torch.tensor(next_states, dtype=torch.float),
            torch.tensor(dones, dtype=torch.float).unsqueeze(1),
        )

    def size(self):
        return len(self.buffer)

# ========================
# SAC Agent 定义
# ========================
class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, hyperparams):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=hyperparams["actor_lr"])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=hyperparams["critic_lr"])
        self.log_alpha = torch.tensor(0.0, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=hyperparams["alpha_lr"])

        self.replay_buffer = ReplayBuffer(hyperparams["buffer_size"])
        self.max_action = max_action
        self.discount = hyperparams["discount"]
        self.tau = hyperparams["tau"]

        self.target_entropy = -action_dim  # 默认目标熵值

    def select_action(self, state, deterministic=False):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        if deterministic:
            mean, _ = self.actor(state)
            action = torch.tanh(mean) * self.max_action
        else:
            action, _ = self.actor.sample_action(state)
        return action.detach().numpy()[0]

    def train(self, batch_size):
        if self.replay_buffer.size() < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # 更新 Critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample_action(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.discount * (torch.min(target_q1, target_q2) - torch.exp(self.log_alpha) * next_log_probs)

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新 Actor
        actions, log_probs = self.actor.sample_action(states)
        q1, q2 = self.critic(states, actions)
        actor_loss = (torch.exp(self.log_alpha) * log_probs - torch.min(q1, q2)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新 Alpha（温度参数）
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # 软更新目标 Critic 网络
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# ========================
# 训练和推理函数
# ========================
def train_sac(hyperparams):
    env = gym.make(hyperparams["env_name"])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = SACAgent(state_dim, action_dim, max_action, hyperparams)

    for episode in range(hyperparams["episodes"]):
        state = env.reset()
        episode_reward = 0

        for t in range(hyperparams["max_timesteps"]):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.add(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            agent.train(hyperparams["batch_size"])

            if done:
                break

        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

    return agent

def test_sac(agent, hyperparams):
    env = gym.make(hyperparams["env_name"])
    for episode in range(hyperparams["test_episodes"]):
        state = env.reset()
        episode_reward = 0

        for t in range(hyperparams["max_timesteps"]):
            action = agent.select_action(state, deterministic=True)
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
    sac_agent = train_sac(HYPERPARAMS)
    test_sac(sac_agent, HYPERPARAMS)
