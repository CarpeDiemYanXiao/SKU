"""
PPO Agent 实现（增强版）
参考 rl_0113/replenishment_abo 的优化技巧
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from .networks import create_networks


def gae_advantages_torch(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GAE (Generalized Advantage Estimation) 的高效 PyTorch 实现
    
    Args:
        rewards: (N,) 奖励序列
        values: (N,) 价值估计序列
        dones: (N,) 终止标志序列
        gamma: 折扣因子
        gae_lambda: GAE λ 参数
    
    Returns:
        advantages: (N,) 优势估计
        returns: (N,) 回报估计
    """
    n = len(rewards)
    advantages = torch.zeros(n, device=rewards.device)
    returns = torch.zeros(n, device=rewards.device)
    
    gae = 0.0
    next_value = 0.0
    
    for t in reversed(range(n)):
        if t == n - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]
    
    return advantages, returns


class PPOAgent:
    """
    PPO (Proximal Policy Optimization) Agent（增强版）
    
    新增功能:
    - 优势值裁剪（防止极端值）
    - Log ratio 裁剪（防止数值溢出）
    - 训练诊断日志
    - 分布式训练支持
    - 完整的 checkpoint 保存/加载
    """
    
    def __init__(self, config: dict, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        
        # PPO 超参数
        ppo_cfg = config["ppo"]
        self.gamma = ppo_cfg["gamma"]
        self.gae_lambda = ppo_cfg["gae_lambda"]
        self.clip_epsilon = ppo_cfg["clip_epsilon"]
        self.entropy_coef = ppo_cfg["entropy_coef"]
        self.value_loss_coef = ppo_cfg["value_loss_coef"]
        self.max_grad_norm = ppo_cfg["max_grad_norm"]
        self.k_epochs = ppo_cfg["k_epochs"]
        self.batch_size = ppo_cfg["batch_size"]
        self.mini_batch_size = ppo_cfg.get("mini_batch_size", 512)
        self.normalize_advantages = ppo_cfg["normalize_advantages"]
        
        # 新增：高级训练选项
        self.advantage_clip = ppo_cfg.get("advantage_clip", 5.0)  # 优势值裁剪范围
        self.log_ratio_clip = ppo_cfg.get("log_ratio_clip", 20.0)  # log ratio 裁剪
        self.diagnose_interval = ppo_cfg.get("diagnose_interval", 100)  # 诊断日志间隔
        
        # 动作配置
        action_cfg = config["action"]
        self.action_type = action_cfg["type"]
        self.action_mode = action_cfg.get("action_mode", "multiplier")
        
        if self.action_type == "discrete":
            if self.action_mode == "stock_days":
                # 库存天数模式（支持float步长）
                stock_days_range = action_cfg.get("stock_days_range", [0, 10])
                step = action_cfg.get("stock_days_step", 1)
                self.action_list = np.arange(
                    stock_days_range[0],
                    stock_days_range[1] + step * 0.5,
                    step
                ).round(2).tolist()
            else:
                # 乘数模式
                step = action_cfg["multiplier_step"]
                self.action_list = np.arange(
                    action_cfg["multiplier_range"][0],
                    action_cfg["multiplier_range"][1] + step,
                    step
                ).round(2).tolist()
        else:
            self.action_list = None
        
        # 创建网络
        self.policy_net, self.value_net = create_networks(config)
        self.policy_net.to(self.device)
        self.value_net.to(self.device)
        
        # 优化器
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=ppo_cfg["lr_actor"],
            weight_decay=ppo_cfg.get("weight_decay", 0.0)
        )
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(),
            lr=ppo_cfg["lr_critic"],
            weight_decay=ppo_cfg.get("weight_decay", 0.0)
        )
        
        # 学习率调度器（可选）
        self.use_lr_scheduler = ppo_cfg.get("use_lr_scheduler", False)
        if self.use_lr_scheduler:
            self.policy_scheduler = optim.lr_scheduler.StepLR(
                self.policy_optimizer, 
                step_size=ppo_cfg.get("lr_decay_step", 100),
                gamma=ppo_cfg.get("lr_decay_gamma", 0.9)
            )
            self.value_scheduler = optim.lr_scheduler.StepLR(
                self.value_optimizer,
                step_size=ppo_cfg.get("lr_decay_step", 100),
                gamma=ppo_cfg.get("lr_decay_gamma", 0.9)
            )
        
        # 统计
        self.global_step = 0
        self.update_step = 0  # 网络更新步数
        
        # 分布式训练状态
        self.distributed = False
        self.rank = 0
        self.world_size = 1
    
    def setup_distributed(self, rank: int, world_size: int):
        """设置分布式训练"""
        self.distributed = True
        self.rank = rank
        self.world_size = world_size
        
        # 缩放学习率
        import math
        lr_scale = math.sqrt(world_size)
        for param_group in self.policy_optimizer.param_groups:
            param_group['lr'] *= lr_scale
        for param_group in self.value_optimizer.param_groups:
            param_group['lr'] *= lr_scale
        
        # 使用 DDP 包装网络
        from torch.nn.parallel import DistributedDataParallel as DDP
        if torch.cuda.is_available() and self.device.type != "cpu":
            self.policy_net = DDP(self.policy_net, device_ids=[self.device])
            self.value_net = DDP(self.value_net, device_ids=[self.device])
        else:
            self.policy_net = DDP(self.policy_net)
            self.value_net = DDP(self.value_net)
    
    def select_action(
        self, 
        state: np.ndarray, 
        deterministic: bool = False
    ) -> Tuple[int, float]:
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob = self.policy_net.get_action(state_tensor, deterministic)
        
        return action.cpu().item(), log_prob.cpu().item()
    
    def select_actions_batch(
        self,
        states: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """批量选择动作"""
        states_tensor = torch.FloatTensor(states).to(self.device)
        
        with torch.no_grad():
            actions, log_probs = self.policy_net.get_action(states_tensor, deterministic)
        
        return actions.cpu().numpy(), log_probs.cpu().numpy()
    
    def get_value(self, state: np.ndarray) -> float:
        """获取状态价值"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            value = self.value_net(state_tensor)
        
        return value.cpu().item()
    
    def get_values_batch(self, states: np.ndarray) -> np.ndarray:
        """批量获取状态价值"""
        states_tensor = torch.FloatTensor(states).to(self.device)
        
        with torch.no_grad():
            values = self.value_net(states_tensor)
        
        return values.cpu().numpy().flatten()
    
    def update(self, rollout_buffer: Dict, writer=None) -> Dict[str, float]:
        """
        PPO 更新（增强版）
        
        新增功能:
        - 优势值裁剪
        - Log ratio 裁剪防止数值溢出
        - 诊断日志
        - TensorBoard 记录
        """
        states = torch.FloatTensor(rollout_buffer["states"]).to(self.device)
        if self.action_type == "discrete":
            actions = torch.LongTensor(rollout_buffer["actions"]).to(self.device)
        else:
            actions = torch.FloatTensor(rollout_buffer["actions"]).to(self.device)
        rewards = torch.FloatTensor(rollout_buffer["rewards"]).to(self.device)
        dones = torch.FloatTensor(rollout_buffer["dones"]).to(self.device)
        old_log_probs = torch.FloatTensor(rollout_buffer["old_log_probs"]).to(self.device)
        values = torch.FloatTensor(rollout_buffer["values"]).to(self.device)
        
        # 诊断：检查输入数据
        if self._should_diagnose():
            self._diagnose_inputs(states, actions, rewards, values)
        
        # 计算 GAE（支持预计算的 per-SKU GAE）
        if "advantages" in rollout_buffer and "returns" in rollout_buffer:
            advantages = torch.FloatTensor(rollout_buffer["advantages"]).to(self.device)
            returns = torch.FloatTensor(rollout_buffer["returns"]).to(self.device)
        else:
            advantages, returns = gae_advantages_torch(
                rewards, values, dones, self.gamma, self.gae_lambda
            )
        
        # 归一化 advantages
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 裁剪 advantages（防止极端值）
        if self.advantage_clip > 0:
            advantages = torch.clamp(advantages, -self.advantage_clip, self.advantage_clip)
        
        # 训练统计
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        n_updates = 0
        
        # 多轮更新
        n_samples = len(states)
        indices = np.arange(n_samples)
        
        for epoch in range(self.k_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, n_samples, self.mini_batch_size):
                end = min(start + self.mini_batch_size, n_samples)
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 计算新的 log_prob 和 entropy
                new_log_probs, entropy = self.policy_net.evaluate_action(
                    batch_states, batch_actions
                )
                
                # 计算 ratio（带裁剪，防止数值溢出）
                log_ratio = new_log_probs - batch_old_log_probs
                log_ratio = torch.clamp(log_ratio, -self.log_ratio_clip, self.log_ratio_clip)
                ratio = torch.exp(log_ratio)
                
                # 计算近似 KL 散度
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_fraction = (torch.abs(ratio - 1.0) > self.clip_epsilon).float().mean()
                
                # PPO clip loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                new_values = self.value_net(batch_states).squeeze(-1)
                value_loss = nn.functional.mse_loss(new_values, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # 总损失
                loss = (
                    policy_loss 
                    + self.value_loss_coef * value_loss 
                    + self.entropy_coef * entropy_loss
                )
                
                # 反向传播
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                
                self.policy_optimizer.step()
                self.value_optimizer.step()
                
                # 统计累积
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_approx_kl += approx_kl.item()
                total_clip_fraction += clip_fraction.item()
                n_updates += 1
                
                self.update_step += 1
                
                # TensorBoard 记录（每个 mini-batch）
                if writer is not None and self.rank == 0:
                    writer.add_scalar('loss/policy', policy_loss.item(), self.update_step)
                    writer.add_scalar('loss/value', value_loss.item(), self.update_step)
                    writer.add_scalar('train/entropy', entropy.mean().item(), self.update_step)
                    writer.add_scalar('train/approx_kl', approx_kl.item(), self.update_step)
                    writer.add_scalar('train/clip_fraction', clip_fraction.item(), self.update_step)
        
        # 学习率调度
        if self.use_lr_scheduler:
            self.policy_scheduler.step()
            self.value_scheduler.step()
        
        self.global_step += 1
        
        # 计算解释方差
        with torch.no_grad():
            explained_var = 1 - (returns - self.value_net(states).squeeze(-1)).var() / (returns.var() + 1e-8)
        
        # 诊断日志
        if self._should_diagnose() and self.rank == 0:
            self._print_diagnostics(
                advantages, returns, total_policy_loss / n_updates,
                total_value_loss / n_updates, total_entropy / n_updates
            )
        
        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "approx_kl": total_approx_kl / n_updates,
            "clip_fraction": total_clip_fraction / n_updates,
            "explained_variance": explained_var.item(),
        }
    
    def _should_diagnose(self) -> bool:
        return False
    
    def _diagnose_inputs(self, states, actions, rewards, values):
        """诊断输入数据"""
        if torch.isnan(states).any():
            print("⚠️ NaN detected in states!")
        if torch.isnan(rewards).any():
            print("⚠️ NaN detected in rewards!")
        if torch.isinf(rewards).any():
            print(f"⚠️ Inf in rewards: max={rewards.max():.2f}, min={rewards.min():.2f}")
    
    def _print_diagnostics(self, advantages, returns, policy_loss, value_loss, entropy):
        """打印诊断信息"""
        print(f"\n[Diagnostics Step {self.global_step}]")
        print(f"  Advantages - mean: {advantages.mean():.4f}, std: {advantages.std():.4f}, "
              f"max: {advantages.max():.4f}, min: {advantages.min():.4f}")
        print(f"  Returns    - mean: {returns.mean():.2f}, std: {returns.std():.2f}")
        print(f"  Losses     - policy: {policy_loss:.6f}, value: {value_loss:.4f}")
        print(f"  Entropy    - {entropy:.4f}")
    
    def save(self, path: str, extra_info: dict = None):
        """保存完整 checkpoint"""
        checkpoint = {
            # 模型参数
            "policy_net": self._get_state_dict(self.policy_net),
            "value_net": self._get_state_dict(self.value_net),
            
            # 优化器状态
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            
            # 训练状态
            "global_step": self.global_step,
            "update_step": self.update_step,
            
            # 配置
            "config": self.config,
        }
        
        if extra_info:
            checkpoint.update(extra_info)
        
        torch.save(checkpoint, path)
    
    def _get_state_dict(self, model):
        """获取模型状态字典（处理 DDP 包装）"""
        if hasattr(model, 'module'):
            return model.module.state_dict()
        return model.state_dict()
    
    def load(self, path: str, mode: str = "full"):
        """
        加载 checkpoint
        
        Args:
            path: checkpoint 路径
            mode: 加载模式
                - "full": 完全恢复（包括优化器状态）
                - "weights": 只加载网络权重
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # 加载网络权重
        if hasattr(self.policy_net, 'module'):
            self.policy_net.module.load_state_dict(checkpoint["policy_net"])
            self.value_net.module.load_state_dict(checkpoint["value_net"])
        else:
            self.policy_net.load_state_dict(checkpoint["policy_net"])
            self.value_net.load_state_dict(checkpoint["value_net"])
        
        if mode == "full":
            # 恢复优化器状态
            if "policy_optimizer" in checkpoint:
                self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
            if "value_optimizer" in checkpoint:
                self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])
            
            # 恢复训练状态
            if "global_step" in checkpoint:
                self.global_step = checkpoint["global_step"]
            if "update_step" in checkpoint:
                self.update_step = checkpoint["update_step"]
        
        print(f"[Agent] Checkpoint loaded from {path} (mode={mode})")
        return checkpoint
    
    def get_action_meaning(self, action: int) -> float:
        """获取动作对应的 multiplier 值"""
        if self.action_type == "discrete":
            return self.action_list[action]
        else:
            return float(action)


class RolloutBuffer:
    """
    Rollout Buffer
    存储一个 episode 的数据
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def get(self) -> Dict:
        return {
            "states": np.array(self.states),
            "actions": np.array(self.actions),
            "rewards": np.array(self.rewards),
            "dones": np.array(self.dones),
            "old_log_probs": np.array(self.log_probs),
            "values": np.array(self.values),
        }
    
    def __len__(self):
        return len(self.states)
