import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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

    # TRPO 参数
    "gamma": 0.99,                  # 折扣因子
    "gae_lambda": 0.97,             # GAE λ
    "max_kl": 0.01,                 # 最大 KL 散度约束
    "cg_iters": 10,                 # 共轭梯度最大迭代次数
    "cg_damping": 1e-2,             # 共轭梯度阻尼因子
    "line_search_steps": 10,        # 线性搜索步数
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
# TRPO Agent
# ========================
class TRPOAgent:
    def __init__(self, state_dim, action_dim, hyperparams):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)

        self.gamma = hyperparams["gamma"]
        self.gae_lambda = hyperparams["gae_lambda"]
        self.max_kl = hyperparams["max_kl"]
        self.cg_iters = hyperparams["cg_iters"]
        self.cg_damping = hyperparams["cg_damping"]
        self.line_search_steps = hyperparams["line_search_steps"]

    def compute_advantages(self, rewards, values, dones):
        advantages = []
        discounted_return = 0
        for reward, done, value in zip(reversed(rewards), reversed(dones), reversed(values)):
            if done:
                discounted_return = 0
            discounted_return = reward + self.gamma * discounted_return
            advantages.insert(0, discounted_return - value)
        return advantages

    def compute_loss_and_kl(self, states, actions, advantages, old_log_probs):
        # 获取新的 log 概率
        action_probs = self.actor(states)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        # 计算损失
        ratio = torch.exp(log_probs - old_log_probs)
        surrogate_loss = (ratio * advantages).mean()
        # 计算 KL 散度
        kl_divergence = torch.mean(torch.sum(old_log_probs.exp() * (old_log_probs - log_probs), dim=1))
        return surrogate_loss, kl_divergence

    def conjugate_gradient(self, fisher_vector_product, b, cg_iters):
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        r_dot_old = torch.dot(r, r)

        for _ in range(cg_iters):
            Ap = fisher_vector_product(p)
            alpha = r_dot_old / (torch.dot(p, Ap) + 1e-10)
            x += alpha * p
            r -= alpha * Ap
            r_dot_new = torch.dot(r, r)

            if torch.sqrt(r_dot_new) < 1e-10:
                break

            beta = r_dot_new / r_dot_old
            p = r + beta * p
            r_dot_old = r_dot_new

        return x

    def fisher_vector_product(self, states, actions, kl_grad):
        action_probs = self.actor(states)
        dist = Categorical(action_probs)

        log_probs = dist.log_prob(actions)
        kl = torch.sum(dist.log_prob(actions) * (dist.probs.detach() - dist.probs), dim=1).mean()

        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        flat_kl_grad = torch.cat([grad.view(-1) for grad in kl_grad])

        kl_hessian = torch.autograd.grad(flat_kl_grad @ kl_grad, self.actor.parameters())
        return kl_hessian


# ========================
# 训练函数
# ========================
def train_trpo(hyperparams):
    env = gym.make(hyperparams["env_name"])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = TRPOAgent(state_dim, action_dim, hyperparams)

    for episode in range(hyperparams["episodes"]):
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
        state = env.reset()
        episode_reward = 0

        for t in range(hyperparams["max_timesteps"]):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action, log_prob, _ = agent.actor.get_action(state_tensor)
            value = agent.critic(state_tensor).item()

            next_state, reward, done, _ = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob)
            values.append(value)

            state = next_state
            episode_reward += reward

            if done:
                break

        # 计算优势和回报
        advantages = agent.compute_advantages(rewards, values, dones)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.int64)
        old_log_probs = torch.stack(log_probs).detach()

        # 策略梯度计算
        surrogate_loss, kl_divergence = agent.compute_loss_and_kl(states_tensor, actions_tensor, advantages, old_log_probs)

        def fisher_vector_product(p):
            return agent.fisher_vector_product(states_tensor, actions_tensor, p) + hyperparams["cg_damping"] * p

        # 共轭梯度计算搜索方向
        loss_grad = torch.autograd.grad(surrogate_loss, agent.actor.parameters())
        flat_loss_grad = torch.cat([grad.view(-1) for grad in loss_grad]).detach()
        step_dir = agent.conjugate_gradient(fisher_vector_product, flat_loss_grad, hyperparams["cg_iters"])

        # 更新步长的计算
        shs = 0.5 * (step_dir @ fisher_vector_product(step_dir))
        step_size = torch.sqrt(2 * hyperparams["max_kl"] / (shs + 1e-10))

        # 线性搜索
        step_success = False
        for _ in range(hyperparams["line_search_steps"]):
            for param, direction in zip(agent.actor.parameters(), step_dir):
                param.data += step_size * direction

            # 检查 KL 散度约束
            _, new_kl = agent.compute_loss_and_kl(states_tensor, actions_tensor, advantages, old_log_probs)
            if new_kl <= hyperparams["max_kl"]:
                step_success = True
                break

            # 回退更新
            for param, direction in zip(agent.actor.parameters(), step_dir):
                param.data -= step_size * direction
            step_size *= 0.5

        if not step_success:
            print(f"Episode {episode + 1}: Line search failed.")

        # Critic 更新
        returns = torch.tensor(rewards, dtype=torch.float32).detach()
        critic_loss = nn.MSELoss()(agent.critic(states_tensor).squeeze(), returns)
        critic_optimizer = optim.Adam(agent.critic.parameters(), lr=1e-3)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, KL = {new_kl:.4f}")

    return agent


def test_trpo(agent, hyperparams):
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
    trpo_agent = train_trpo(HYPERPARAMS)
    test_trpo(trpo_agent, HYPERPARAMS)
