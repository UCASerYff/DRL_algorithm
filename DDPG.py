import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# ========================
# 超参数设置
# ========================
HYPERPARAMS = {
    # 环境参数
    "env_name": "Pendulum-v1",       # 环境名称
    "episodes": 100,                # 训练轮数
    "test_episodes": 5,             # 测试轮数
    "max_timesteps": 200,           # 每轮最大时间步长

    # 网络与优化参数
    "actor_lr": 1e-3,               # Actor 学习率
    "critic_lr": 1e-3,              # Critic 学习率
    "batch_size": 64,               # 批量大小
    "discount": 0.99,               # 折扣因子
    "tau": 0.005,                   # 目标网络软更新系数

    # 优先经验回放
    "buffer_size": 100000,          # 回放缓存大小
    "alpha": 0.6,                   # 优先经验回放权重因子
    "beta_start": 0.4,              # 重要性采样初始值
    "beta_increment": 1e-4,         # β 的每步递增值

    # 动作噪声
    "noise_std": 0.1,               # 动作高斯噪声标准差
}

# ========================
# Actor 网络定义
# ========================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        return x * self.max_action

# ========================
# Critic 网络定义
# ========================
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)

    def forward(self, state, action):
        # 将状态和动作作为输入
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

# ========================
# 优先经验回放缓冲区
# ========================
class PrioritizedReplayBuffer:
    def __init__(self, max_size, alpha=0.6):
        self.buffer = []  # 用于存储经验
        self.priorities = np.zeros((max_size,), dtype=np.float32)  # 优先级数组
        self.max_size = max_size
        self.alpha = alpha
        self.ptr = 0  # 当前存储位置

    def add(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.max_size:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.ptr] = (state, action, reward, next_state, done)
        self.priorities[self.ptr] = max_priority
        self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size, beta=0.4):
        # 根据优先级采样
        priorities = self.priorities[:len(self.buffer)] ** self.alpha
        probabilities = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        states, actions, rewards, next_states, dones = zip(*samples)

        # 计算重要性采样权重
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        return (
            torch.tensor(states, dtype=torch.float),
            torch.tensor(actions, dtype=torch.float),
            torch.tensor(rewards, dtype=torch.float).unsqueeze(1),
            torch.tensor(next_states, dtype=torch.float),
            torch.tensor(dones, dtype=torch.float).unsqueeze(1),
            torch.tensor(weights, dtype=torch.float).unsqueeze(1),
            indices
        )

    def update_priorities(self, indices, td_errors):
        # 更新优先级
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-5

# ========================
# DDPG Agent 定义
# ========================
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, hyperparams):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=hyperparams["actor_lr"])

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=hyperparams["critic_lr"])

        self.replay_buffer = PrioritizedReplayBuffer(max_size=hyperparams["buffer_size"], alpha=hyperparams["alpha"])
        self.max_action = max_action
        self.discount = hyperparams["discount"]
        self.tau = hyperparams["tau"]

    def select_action(self, state, noise=0.0):
        # 使用 Actor 网络选择动作
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        if noise > 0:
            action += np.random.normal(0, noise, size=action.shape)
        return np.clip(action, -self.max_action, self.max_action)

    def train(self, batch_size, beta):
        if len(self.replay_buffer.buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones, weights, indices = self.replay_buffer.sample(batch_size, beta)

        # Critic 损失
        with torch.no_grad():
            target_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, target_actions)
            target_q = rewards + (1 - dones) * self.discount * target_q

        current_q = self.critic(states, actions)
        td_errors = target_q - current_q
        critic_loss = (weights * td_errors.pow(2)).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新优先级
        self.replay_buffer.update_priorities(indices, td_errors.detach().numpy())

        # Actor 损失
        actor_loss = -(weights * self.critic(states, self.actor(states))).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新目标网络
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic.state_dict(), filename + "_critic.pth")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor.pth"))
        self.critic.load_state_dict(torch.load(filename + "_critic.pth"))

# ========================
# 训练和推理函数
# ========================
def train_ddpg(hyperparams):
    env = gym.make(hyperparams["env_name"])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = DDPGAgent(state_dim, action_dim, max_action, hyperparams)
    beta = hyperparams["beta_start"]

    for episode in range(hyperparams["episodes"]):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, noise=hyperparams["noise_std"])
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            agent.train(hyperparams["batch_size"], beta)

        beta = min(1.0, beta + hyperparams["beta_increment"])
        print(f"Episode {episode + 1}, Reward: {episode_reward}")

    return agent


def infer_ddpg(agent, hyperparams):
    env = gym.make(hyperparams["env_name"])
    for episode in range(hyperparams["test_episodes"]):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, noise=0.0)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            env.render()

        print(f"Test Episode {episode + 1}, Reward: {episode_reward}")
    env.close()

# ========================
# 主函数
# ========================
if __name__ == "__main__":
    trained_agent = train_ddpg(HYPERPARAMS)
    infer_ddpg(trained_agent, HYPERPARAMS)
