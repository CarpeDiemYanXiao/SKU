import numpy as np
import torch

def gae_advantages(rewards, values, dones, config):
    """
    使用 GAE 计算优势函数
    Args:
        rewards: 每一步的即时奖励 (list or array)
        values: 每一步的状态值函数估计 (list or array)
        gamma: 折扣因子
        lam: GAE 的衰减系数
    Returns:
        advantages: 每一步的优势函数估计
    """
    if not hasattr(config, "gamma"):
        gamma = 0.99
    else:
        gamma = config.gamma
    if not hasattr(config, "lam"):
        lam = 0.95
    else:
        lam = config.lam
    # gamma = config.get("gamma",0.99)
    # lam = config.get("lam", 0.95)
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    discounted_reward = 0  # 初始化 GAE
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (values[t + 1] if (t < len(rewards) - 1) else 0) - values[t]
        if dones[t]:
            discounted_reward = 0
        returns[t] = rewards[t] + (gamma * discounted_reward)
        discounted_reward = delta + gamma * lam * discounted_reward
        advantages[t] = discounted_reward
    return advantages,returns


def base_advantages(rewards, values, dones, config):
    # 送入函数之前必须保证rewards，values和dones在同一个device上
    # gamma = config.get("gamma",0.99)
    if not hasattr(config, "gamma"):
        gamma = 0.99
    else:
        gamma = config.gamma
    returns = torch.zeros_like(rewards)
    discounted_reward = 0
    # for reward, done in zip(reversed(rewards), reversed(dones)):
    for t in reversed(range(len(rewards))):
        if dones[t]:
            discounted_reward = 0
        discounted_reward = rewards[t] + (gamma * discounted_reward)
        returns[t] = discounted_reward
    # 先用ndarray算好，直接把list强转tensor会慢
    advantages = returns - values
    return advantages, returns

