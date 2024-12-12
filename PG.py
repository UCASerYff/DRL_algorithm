import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ========================
# 超参数设置
# ========================
HYPERPARAMS = {
    # 环境参数
    "env_name": "CartPole-v1",      # 环境名称
    "episodes": 500,                # 训练轮数
    "max_timesteps": 200,           # 每轮最大时间步数
    "test_episodes": 10,            # 测试轮数

    # Policy Gradient 参数
    "gamma": 0.99,                  # 折扣因子
    "lr": 1e-3,                     # 学习率
}

# ========================
# 策略网络
# ========================
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.model(state)

    def get_action(self, state):
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

# ========================
# PG Agent
# ========================
class PGAgent:
    def __init__(self, state_dim, action_dim, hyperparams):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=hyperparams["lr"])
        self.gamma = hyperparams["gamma"]

    def compute_returns(self, rewards):
        """计算折扣回报"""
        returns = []
        discounted_sum = 0
        for reward in reversed(rewards):
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        return torch.tensor(returns, dtype=torch.float32)

    def update(self, log_probs, returns):
        """更新策略网络"""
        log_probs = torch.stack(log_probs)
        loss = -torch.sum(log_probs * returns)  # 损失函数为加权的负对数概率
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# ========================
# 训练函数
# ========================
def train_pg(hyperparams):
    env = gym.make(hyperparams["env_name"])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PGAgent(state_dim, action_dim, hyperparams)

    for episode in range(hyperparams["episodes"]):
        state = env.reset()
        log_probs, rewards = [], []
        episode_reward = 0

        for t in range(hyperparams["max_timesteps"]):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            # 与环境交互
            action, log_prob = agent.policy.get_action(state_tensor)
            next_state, reward, done, _ = env.step(action)

            # 记录数据
            log_probs.append(log_prob)
            rewards.append(reward)
            episode_reward += reward
            state = next_state

            if done:
                break

        # 计算折扣回报
        returns = agent.compute_returns(rewards)
        # 标准化回报（提升训练稳定性）
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 更新策略网络
        agent.update(log_probs, returns)

        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

    return agent

# ========================
# 测试函数
# ========================
def test_pg(agent, hyperparams):
    env = gym.make(hyperparams["env_name"])
    for episode in range(hyperparams["test_episodes"]):
        state = env.reset()
        episode_reward = 0

        for t in range(hyperparams["max_timesteps"]):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
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
    pg_agent = train_pg(HYPERPARAMS)
    test_pg(pg_agent, HYPERPARAMS)
