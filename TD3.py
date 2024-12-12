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
    "env_name": "Pendulum-v1",  # 环境名称
    "episodes": 100,  # 训练轮数
    "test_episodes": 5,  # 测试轮数
    "max_timesteps": 200,  # 每轮最大时间步长

    # 网络与优化参数
    "actor_lr": 1e-3,  # Actor 学习率
    "critic_lr": 1e-3,  # Critic 学习率
    "batch_size": 64,  # 批量大小
    "discount": 0.99,  # 折扣因子
    "tau": 0.005,  # 目标网络软更新系数
    "policy_noise": 0.2,  # 目标动作噪声
    "noise_clip": 0.5,  # 动作噪声裁剪范围
    "policy_update_freq": 2,  # 策略更新频率

    # 优先经验回放
    "buffer_size": 100000,  # 回放缓存大小

    # 动作噪声
    "exploration_noise": 0.1,  # 训练期间的高斯噪声
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
        # Q1 网络
        self.layer1 = nn.Linear(state_dim + action_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)

        # Q2 网络
        self.layer4 = nn.Linear(state_dim + action_dim, 256)
        self.layer5 = nn.Linear(256, 256)
        self.layer6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)

        # Q1 计算
        q1 = torch.relu(self.layer1(sa))
        q1 = torch.relu(self.layer2(q1))
        q1 = self.layer3(q1)

        # Q2 计算
        q2 = torch.relu(self.layer4(sa))
        q2 = torch.relu(self.layer5(q2))
        q2 = self.layer6(q2)

        return q1, q2

    def q1(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = torch.relu(self.layer1(sa))
        q1 = torch.relu(self.layer2(q1))
        return self.layer3(q1)


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
# TD3 Agent 定义
# ========================
class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action, hyperparams):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=hyperparams["actor_lr"])

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=hyperparams["critic_lr"])

        self.replay_buffer = ReplayBuffer(max_size=hyperparams["buffer_size"])
        self.max_action = max_action
        self.discount = hyperparams["discount"]
        self.tau = hyperparams["tau"]
        self.policy_noise = hyperparams["policy_noise"]
        self.noise_clip = hyperparams["noise_clip"]
        self.policy_update_freq = hyperparams["policy_update_freq"]

        self.total_it = 0

    def select_action(self, state, noise=0.0):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        if noise > 0:
            action += np.random.normal(0, noise, size=action.shape)
        return np.clip(action, -self.max_action, self.max_action)

    def train(self, batch_size):
        if self.replay_buffer.size() < batch_size:
            return

        self.total_it += 1

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        with torch.no_grad():
            # 目标动作添加噪声
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)

            # 计算目标 Q 值
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.discount * target_q

        # 更新 Critic 网络
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 延迟更新 Actor 网络
        if self.total_it % self.policy_update_freq == 0:
            actor_loss = -self.critic.q1(states, self.actor(states)).mean()

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
def train_td3(hyperparams):
    env = gym.make(hyperparams["env_name"])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = TD3Agent(state_dim, action_dim, max_action, hyperparams)

    for episode in range(hyperparams["episodes"]):
        state = env.reset()
        episode_reward = 0

        for t in range(hyperparams["max_timesteps"]):
            action = agent.select_action(state, noise=hyperparams["exploration_noise"])
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.add(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            agent.train(hyperparams["batch_size"])

            if done:
                break

        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

    return agent


def test_td3(agent, hyperparams):
    env = gym.make(hyperparams["env_name"])
    for episode in range(hyperparams["test_episodes"]):
        state = env.reset()
        episode_reward = 0

        for t in range(hyperparams["max_timesteps"]):
            action = agent.select_action(state)
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
    td3_agent = train_td3(HYPERPARAMS)
    test_td3(td3_agent, HYPERPARAMS)
