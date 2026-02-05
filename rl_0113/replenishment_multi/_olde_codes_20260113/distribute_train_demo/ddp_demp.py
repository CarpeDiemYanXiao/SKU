import os
import random
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from collections import deque
import argparse


# 设置随机种子以确保可重复性
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 定义简单的DQN网络
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.FloatTensor(np.array(state)),
            torch.LongTensor(np.array(action)),
            torch.FloatTensor(np.array(reward)),
            torch.FloatTensor(np.array(next_state)),
            torch.FloatTensor(np.array(done)),
        )

    def __len__(self):
        return len(self.buffer)


# epsilon贪心策略选择动作
def select_action(state, policy_net, epsilon, device, action_dim):
    if random.random() > epsilon:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            return policy_net(state).max(1)[1].item()
    else:
        return random.randrange(action_dim)


# 训练函数
def train(local_rank, args):
    # 计算全局rank
    global_rank = args.node_rank * args.gpus + local_rank

    # 设置分布式环境
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)

    # 初始化进程组
    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        rank=global_rank,
        world_size=args.world_size,
    )

    # 设置设备
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(local_rank)

    # 环境设置
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 初始化模型
    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    policy_net = DDP(policy_net, device_ids=[local_rank] if torch.cuda.is_available() else None)
    target_net.load_state_dict(policy_net.module.state_dict())
    target_net.eval()

    # 优化器
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)

    # 经验回放缓冲区
    replay_buffer = ReplayBuffer(10000)

    # 训练参数
    batch_size = 128
    gamma = 0.99
    epsilon_start = 0.9
    epsilon_end = 0.05
    epsilon_decay = 1000
    target_update = 10
    num_episodes = 500

    # 训练循环
    total_steps = 0

    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        episode_reward = 0

        while not done:
            # 计算当前的epsilon
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1.0 * total_steps / epsilon_decay)

            # 选择动作
            action = select_action(state, policy_net.module, epsilon, device, action_dim)

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 存储转移
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            total_steps += 1

            # 当缓冲区足够大时开始训练
            if len(replay_buffer) > batch_size:
                # 从回放缓冲区采样
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                # 将数据移至设备
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                dones = dones.to(device)

                # 计算当前Q值
                current_q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                # 计算下一个状态的最大Q值
                next_q_values = target_net(next_states).max(1)[0]
                target_q_values = rewards + gamma * next_q_values * (1 - dones)

                # 计算损失并优化
                loss = F.smooth_l1_loss(current_q_values, target_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 每隔一定间隔更新目标网络
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.module.state_dict())

        # 只在全局主进程中打印
        if global_rank == 0:
            print(f"Episode {episode}, Reward: {episode_reward}, Epsilon: {epsilon:.2f}")

    # 清理
    dist.destroy_process_group()
    env.close()


# 分布式训练无经验回放版本
def train_without_replay(local_rank, args):
    # 初始化分布式环境（与原来相同）
    global_rank = args.node_rank * args.gpus + local_rank
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        rank=global_rank,
        world_size=args.world_size,
    )

    # 设备设置（与原来相同）
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(local_rank)

    # 模型初始化（与原来相同）
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy_net = DQN(state_dim, action_dim).to(device)
    policy_net = DDP(policy_net, device_ids=[local_rank] if torch.cuda.is_available() else None)

    # 优化器（与原来相同）
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)

    # 训练循环（无经验回放）
    for episode in range(500):
        state = env.reset()[0]
        done = False
        episode_reward = 0

        while not done:
            # 选择动作
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = policy_net(state_tensor).max(1)[1].item()

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)

            # 直接计算TD误差并优化（在线学习）
            q_value = policy_net(state_tensor).gather(1, torch.tensor([[action]], device=device))
            next_q_value = policy_net(next_state_tensor).max(1)[0].unsqueeze(1)
            expected_q = reward + (0.99 * next_q_value * (1 - done))

            # 计算损失并更新
            loss = F.mse_loss(q_value, expected_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            episode_reward += reward

        # 同步打印
        if global_rank == 0:
            print(f"Episode {episode}, Reward: {episode_reward}")

    # 清理（与原来相同）
    dist.destroy_process_group()
    env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-node DDP Reinforcement Learning")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs per node")
    parser.add_argument("--node_rank", type=int, default=0, help="Ranking of the current node")
    parser.add_argument("--master_addr", type=str, default="localhost", help="Master node address")
    parser.add_argument("--master_port", type=int, default=12355, help="Master port")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    args.world_size = args.nodes * args.gpus

    # 设置种子
    set_seed(args.seed)

    print(f"Node {args.node_rank}/{args.nodes}, starting training with {args.gpus} GPUs per node")

    # 启动多进程训练
    if torch.cuda.is_available():
        import torch.multiprocessing as mp

        mp.spawn(train, args=(args,), nprocs=args.gpus)
    else:
        train(0, args)
