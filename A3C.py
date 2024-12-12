import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np

# ========================
# 超参数设置
# ========================
HYPERPARAMS = {
    # 环境参数
    "env_name": "CartPole-v1",      # 环境名称
    "max_timesteps": 200,           # 每轮最大时间步数
    "test_episodes": 10,            # 测试轮数

    # A3C 参数
    "actor_lr": 1e-4,               # Actor 学习率
    "critic_lr": 5e-4,              # Critic 学习率
    "gamma": 0.99,                  # 折扣因子
    "gae_lambda": 1.0,              # GAE λ

    # 并行设置
    "num_processes": 4,             # 并行进程数
    "update_interval": 5,           # 全局模型更新间隔
    "max_steps": 1000000,           # 最大训练步数
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
# Worker 线程
# ========================
class Worker(mp.Process):
    def __init__(self, global_model, optimizer, rank, hyperparams):
        super(Worker, self).__init__()
        self.global_model = global_model
        self.optimizer = optimizer
        self.rank = rank
        self.hyperparams = hyperparams
        self.env = gym.make(hyperparams["env_name"])
        self.local_model = ActorCritic(self.env.observation_space.shape[0], self.env.action_space.n)

    def compute_advantages(self, rewards, values, next_value, dones):
        advantages = []
        discounted_return = next_value
        for reward, value, done in zip(reversed(rewards), reversed(values), reversed(dones)):
            if done:
                discounted_return = 0
            discounted_return = reward + self.hyperparams["gamma"] * discounted_return
            advantages.insert(0, discounted_return - value)
        return advantages

    def run(self):
        state = self.env.reset()
        local_steps = 0
        while local_steps < self.hyperparams["max_steps"]:
            states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
            for _ in range(self.hyperparams["update_interval"]):
                state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
                action, log_prob = self.local_model.get_action(state_tensor)
                value = self.local_model.forward(state_tensor)[1]

                next_state, reward, done, _ = self.env.step(action)
                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                dones.append(done)
                values.append(value.item())

                state = next_state if not done else self.env.reset()
                local_steps += 1

                if done:
                    break

            # 计算优势和回报
            next_value = 0 if done else self.local_model.forward(torch.tensor(state, dtype=torch.float).unsqueeze(0))[1].item()
            advantages = self.compute_advantages(rewards, values, next_value, dones)
            returns = [adv + val for adv, val in zip(advantages, values)]

            # 转换为张量
            states = torch.tensor(states, dtype=torch.float)
            actions = torch.tensor(actions, dtype=torch.long)
            log_probs = torch.stack(log_probs)
            advantages = torch.tensor(advantages, dtype=torch.float)
            returns = torch.tensor(returns, dtype=torch.float)

            # 更新本地模型
            new_log_probs, state_values, _ = self.evaluate_actions(states, actions)
            actor_loss = -(advantages * new_log_probs).mean()
            critic_loss = nn.MSELoss()(state_values.squeeze(), returns)
            total_loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            for local_param, global_param in zip(self.local_model.parameters(), self.global_model.parameters()):
                global_param.grad = local_param.grad
            self.optimizer.step()

            # 同步本地模型和全局模型
            self.local_model.load_state_dict(self.global_model.state_dict())

    def evaluate_actions(self, states, actions):
        action_probs, state_values = self.local_model.forward(states)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, state_values, entropy

# ========================
# 测试函数
# ========================
def test_a3c(global_model, hyperparams):
    env = gym.make(hyperparams["env_name"])
    for episode in range(hyperparams["test_episodes"]):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            action_probs, _ = global_model(state_tensor)
            action = torch.argmax(action_probs, dim=1).item()
            state, reward, done, _ = env.step(action)
            episode_reward += reward

        print(f"Test Episode {episode + 1}: Reward = {episode_reward:.2f}")

# ========================
# 主训练函数
# ========================
def train_a3c(hyperparams):
    env = gym.make(hyperparams["env_name"])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    global_model = ActorCritic(state_dim, action_dim)
    global_model.share_memory()  # 共享内存
    optimizer = optim.Adam(global_model.parameters(), lr=hyperparams["actor_lr"])

    processes = []
    for rank in range(hyperparams["num_processes"]):
        worker = Worker(global_model, optimizer, rank, hyperparams)
        worker.start()
        processes.append(worker)

    for p in processes:
        p.join()

    return global_model

# ========================
# 主函数调用
# ========================
if __name__ == "__main__":
    global_a3c_model = train_a3c(HYPERPARAMS)
    test_a3c(global_a3c_model, HYPERPARAMS)
