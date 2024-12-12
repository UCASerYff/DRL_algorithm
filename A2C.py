import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ========================
# 超参数设置
# ========================
HYPERPARAMS = {
    # 环境参数
    "env_name": "CartPole-v1",      # 环境名称
    "episodes": 1000,               # 训练轮数
    "max_timesteps": 200,           # 每轮最大时间步数
    "test_episodes": 10,            # 测试轮数

    # A2C 参数
    "actor_lr": 1e-4,               # Actor 学习率
    "critic_lr": 5e-4,              # Critic 学习率
    "gamma": 0.99,                  # 折扣因子
    "gae_lambda": 1.0,              # GAE λ
    "update_interval": 5,           # 策略更新间隔
}

# ========================
# Actor-Critic 网络
# ========================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # 公共特征层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        # Actor 分支
        self.actor = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        # Critic 分支
        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        shared_features = self.shared(state)
        action_probs = self.actor(shared_features)
        state_value = self.critic(shared_features)
        return action_probs, state_value

    def get_action(self, state):
        action_probs, _ = self.forward(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

# ========================
# A2C Agent
# ========================
class A2CAgent:
    def __init__(self, state_dim, action_dim, hyperparams):
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer_actor = optim.Adam(self.policy.actor.parameters(), lr=hyperparams["actor_lr"])
        self.optimizer_critic = optim.Adam(self.policy.critic.parameters(), lr=hyperparams["critic_lr"])

        self.gamma = hyperparams["gamma"]
        self.gae_lambda = hyperparams["gae_lambda"]
        self.update_interval = hyperparams["update_interval"]

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
        old_log_probs = torch.stack(self.log_probs)

        # 计算优势和回报
        advantages, returns = self.compute_advantages()

        # 获取新的 log 概率和状态值
        new_log_probs, state_values, entropy = self.evaluate_actions(states, actions)

        # Actor 损失
        actor_loss = -(advantages * new_log_probs).mean()
        # Critic 损失
        critic_loss = nn.MSELoss()(state_values.squeeze(), returns)
        # 总损失
        total_loss = actor_loss + 0.5 * critic_loss

        # 更新策略网络
        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        total_loss.backward()
        self.optimizer_actor.step()
        self.optimizer_critic.step()

        self.clear_buffer()

    def evaluate_actions(self, states, actions):
        action_probs, state_values = self.policy(states)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, state_values, entropy

# ========================
# 训练和推理函数
# ========================
def train_a2c(hyperparams):
    env = gym.make(hyperparams["env_name"])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = A2CAgent(state_dim, action_dim, hyperparams)

    for episode in range(hyperparams["episodes"]):
        state = env.reset()
        episode_reward = 0

        for t in range(hyperparams["max_timesteps"]):
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            action, log_prob = agent.policy.get_action(state_tensor)
            value = agent.policy(state_tensor)[1].item()

            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, log_prob, reward, done, value)

            state = next_state
            episode_reward += reward

            if done or len(agent.states) >= hyperparams["update_interval"]:
                agent.update()

            if done:
                break

        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

    return agent

def test_a2c(agent, hyperparams):
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
    a2c_agent = train_a2c(HYPERPARAMS)
    test_a2c(a2c_agent, HYPERPARAMS)