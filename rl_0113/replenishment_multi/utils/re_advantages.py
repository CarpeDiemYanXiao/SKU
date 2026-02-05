import torch
import numpy as np

def gae_advantages_torch(td_delta, gamma, lmbda):
    """纯 PyTorch 实现，避免 CPU-GPU 数据搬运"""
    T = td_delta.shape[0]
    advantages = torch.zeros_like(td_delta)
    
    gae = 0
    for t in reversed(range(T)):
        gae = td_delta[t] + gamma * lmbda * gae
        advantages[t] = gae
    
    return advantages

def gae_advantages_numpy(td_delta, gamma, lmbda):
    """
    计算 GAE 优势函数
    A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
    
    Args:
        td_delta: TD 误差序列，shape (T, 1) 或 (T,)
        gamma: 折扣因子
        lmbda: GAE 的 λ 参数，控制偏差-方差权衡
    
    Returns:
        advantages: 优势估计，shape 同 td_delta
    """
    td_delta = td_delta.detach().cpu().numpy()  # 转 numpy 方便操作
    advantages = np.zeros_like(td_delta)
    
    gae = 0
    # 从后往前累积计算
    for t in reversed(range(len(td_delta))):
        gae = td_delta[t] + gamma * lmbda * gae
        advantages[t] = gae
    
    return torch.tensor(advantages, dtype=torch.float, device=td_delta.device if hasattr(td_delta, 'device') else 'cpu')