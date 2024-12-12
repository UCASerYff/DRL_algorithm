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

    # Actor-Critic 参数
    "gamma": 0.99,                  # 折扣因子
    "lr_actor": 1e-3,               # Actor 学习率
    "lr_critic": 1e-3,              # Critic 学习率
    "entropy_beta": 0.01,           # 熵奖励系数
}

# ========================
# Actor 网络
# ========================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
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
        return action.item(), dist.log_prob(action), dist.entropy()

# ========================
# Critic 网络
# ========================
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        return self.model(state)

# ========================
# Actor-Critic Agent
# ========================
class ACAgent:
    def __init__(self, state_dim, action_dim, hyperparams):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)

        self.gamma = hyperparams["gamma"]
        self.entropy_beta = hyperparams["entropy_beta"]

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=hyperparams["lr_actor"])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=hyperparams["lr_critic"])

    def update(self, states, actions, rewards, dones, log_probs, entropies):
        # 计算状态值
        states_tensor = torch.tensor(states, dtype=torch.float32)
        values = self.critic(states_tensor).squeeze()

        # 计算回报
        returns = []
        g = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            g = reward + self.gamma * g * (1 - done)
            returns.insert(0, g)
        returns = torch.tensor(returns, dtype=torch.float32)

        # 计算优势
        advantages = returns - values.detach()

        # Actor 损失 (策略梯度 + 熵正则化)
        log_probs_tensor = torch.stack(log_probs)
        entropy_tensor = torch.stack(entropies)
        actor_loss = -torch.mean(log_probs_tensor * advantages + self.entropy_beta * entropy_tensor)

        # Critic 损失 (MSE 损失)
        critic_loss = nn.MSELoss()(values, returns)

        # 更新 Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新 Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

# ========================
# 训练函数
# ========================
def train_ac(hyperparams):
    env = gym.make(hyperparams["env_name"])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = ACAgent(state_dim, action_dim, hyperparams)

    for episode in range(hyperparams["episodes"]):
        state = env.reset()
        states, actions, rewards, dones, log_probs, entropies = [], [], [], [], [], []
        episode_reward = 0

        for t in range(hyperparams["max_timesteps"]):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            # 与环境交互
            action, log_prob, entropy = agent.actor.get_action(state_tensor)
            next_state, reward, done, _ = env.step(action)

            # 记录数据
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob)
            entropies.append(entropy)

            state = next_state
            episode_reward += reward

            if done:
                break

        # 更新 Actor 和 Critic
        agent.update(states, actions, rewards, dones, log_probs, entropies)

        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

    return agent

# ========================
# 测试函数
# ========================
def test_ac(agent, hyperparams):
    env = gym.make(hyperparams["env_name"])
    for episode in range(hyperparams["test_episodes"]):
        state = env.reset()
        episode_reward = 0

        for t in range(hyperparams["max_timesteps"]):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action, _, _ = agent.actor.get_action(state_tensor)
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
    ac_agent = train_ac(HYPERPARAMS)
    test_ac(ac_agent, HYPERPARAMS)
