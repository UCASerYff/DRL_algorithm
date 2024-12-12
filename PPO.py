import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ========================
# 超参数设置
# ========================
HYPERPARAMS = {
    # 环境参数
    "env_name": "CartPole-v1",      # 环境名称
    "episodes": 1000,               # 训练轮数
    "max_timesteps": 200,           # 每轮最大时间步
    "test_episodes": 10,            # 测试轮数

    # PPO 参数
    "actor_lr": 3e-4,               # Actor 学习率
    "critic_lr": 1e-3,              # Critic 学习率
    "gamma": 0.99,                  # 折扣因子
    "gae_lambda": 0.95,             # GAE λ
    "clip_eps": 0.2,                # PPO clip 参数
    "k_epochs": 4,                  # 更新次数
    "batch_size": 64,               # 批量大小

    # 经验回放
    "buffer_size": 2048,            # 回放缓存大小
}

# ========================
# Actor-Critic 网络
# ========================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Actor 网络
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        # Critic 网络
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value

    def get_action(self, state):
        action_probs, _ = self.forward(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def evaluate_actions(self, states, actions):
        action_probs, state_values = self.forward(states)
        dist = torch.distributions.Categorical(action_probs)
        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return action_log_probs, state_values, entropy

# ========================
# PPO Agent
# ========================
class PPOAgent:
    def __init__(self, state_dim, action_dim, hyperparams):
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer_actor = optim.Adam(self.policy.actor.parameters(), lr=hyperparams["actor_lr"])
        self.optimizer_critic = optim.Adam(self.policy.critic.parameters(), lr=hyperparams["critic_lr"])

        self.gamma = hyperparams["gamma"]
        self.gae_lambda = hyperparams["gae_lambda"]
        self.clip_eps = hyperparams["clip_eps"]
        self.k_epochs = hyperparams["k_epochs"]
        self.batch_size = hyperparams["batch_size"]

        # 缓存
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def store_transition(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear_buffer(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def compute_advantages(self):
        advantages = []
        returns = []
        discounted_return = 0
        for reward, done, value in zip(reversed(self.rewards), reversed(self.dones), reversed(self.values)):
            if done:
                discounted_return = 0
            discounted_return = reward + self.gamma * discounted_return
            returns.insert(0, discounted_return)

        returns = torch.tensor(returns, dtype=torch.float)
        values = torch.tensor(self.values, dtype=torch.float)
        advantages = returns - values
        return advantages, returns

    def update(self):
        # 将缓冲区数据转化为张量
        states = torch.tensor(self.states, dtype=torch.float)
        actions = torch.tensor(self.actions, dtype=torch.long)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float)

        # 计算优势和回报
        advantages, returns = self.compute_advantages()

        for _ in range(self.k_epochs):
            # 获取新的 log 概率和状态值
            new_log_probs, state_values, entropy = self.policy.evaluate_actions(states, actions)

            # 计算比率
            ratio = torch.exp(new_log_probs - old_log_probs)

            # 计算 PPO 损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(state_values.squeeze(), returns)

            # 总损失
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

            # 更新策略网络
            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            loss.backward()
            self.optimizer_actor.step()
            self.optimizer_critic.step()

        self.clear_buffer()

# ========================
# 训练和推理函数
# ========================
def train_ppo(hyperparams):
    env = gym.make(hyperparams["env_name"])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(state_dim, action_dim, hyperparams)

    for episode in range(hyperparams["episodes"]):
        state = env.reset()
        episode_reward = 0

        for t in range(hyperparams["max_timesteps"]):
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            action, log_prob = agent.policy.get_action(state_tensor)
            value = agent.policy.critic(state_tensor).item()

            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, log_prob, reward, done, value)

            state = next_state
            episode_reward += reward

            if done:
                break

        # 更新策略
        agent.update()
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

    return agent

def test_ppo(agent, hyperparams):
    env = gym.make(hyperparams["env_name"])
    for episode in range(hyperparams["test_episodes"]):
        state = env.reset()
        episode_reward = 0

        for t in range(hyperparams["max_timesteps"]):
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            action, _ = agent.policy.get_action(state_tensor)
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
    ppo_agent = train_ppo(HYPERPARAMS)
    test_ppo(ppo_agent, HYPERPARAMS)
