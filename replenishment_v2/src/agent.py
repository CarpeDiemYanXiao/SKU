"""
PPO Agent 实现
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from .networks import create_networks


class PPOAgent:
    """
    PPO (Proximal Policy Optimization) Agent
    
    支持离散和连续动作空间
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
        self.mini_batch_size = ppo_cfg.get("mini_batch_size", 512)  # NPU优化：使用mini-batch
        self.normalize_advantages = ppo_cfg["normalize_advantages"]
        
        # 动作配置
        action_cfg = config["action"]
        self.action_type = action_cfg["type"]
        if self.action_type == "discrete":
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
            weight_decay=ppo_cfg["weight_decay"]
        )
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(),
            lr=ppo_cfg["lr_critic"],
            weight_decay=ppo_cfg["weight_decay"]
        )
        
        # 统计
        self.global_step = 0
    
    def select_action(
        self, 
        state: np.ndarray, 
        deterministic: bool = False
    ) -> Tuple[int, float]:
        """
        选择动作
        
        Args:
            state: 状态向量
            deterministic: 是否确定性策略
            
        Returns:
            action: 动作 (离散为索引，连续为值)
            log_prob: 对数概率
        """
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
    
    def update(self, rollout_buffer: Dict) -> Dict[str, float]:
        """
        PPO 更新
        
        Args:
            rollout_buffer: 包含 rollout 数据的字典
                - states: (N, state_dim)
                - actions: (N,)
                - rewards: (N,)
                - dones: (N,)
                - old_log_probs: (N,)
                - values: (N,)
                
        Returns:
            训练统计信息
        """
        states = torch.FloatTensor(rollout_buffer["states"]).to(self.device)
        actions = torch.LongTensor(rollout_buffer["actions"]).to(self.device) if self.action_type == "discrete" else torch.FloatTensor(rollout_buffer["actions"]).to(self.device)
        rewards = torch.FloatTensor(rollout_buffer["rewards"]).to(self.device)
        dones = torch.FloatTensor(rollout_buffer["dones"]).to(self.device)
        old_log_probs = torch.FloatTensor(rollout_buffer["old_log_probs"]).to(self.device)
        values = torch.FloatTensor(rollout_buffer["values"]).to(self.device)
        
        # 计算 GAE
        advantages, returns = self._compute_gae(rewards, values, dones)
        
        # 归一化 advantages
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 训练统计
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0
        
        # 多轮更新
        n_samples = len(states)
        indices = np.arange(n_samples)
        
        for _ in range(self.k_epochs):
            np.random.shuffle(indices)
            
            # 使用 mini_batch_size 进行更新
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
                
                # 计算 ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
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
                
                # 更新 policy
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                
                self.policy_optimizer.step()
                self.value_optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
        
        self.global_step += 1
        
        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }
    
    def _compute_gae(
        self, 
        rewards: torch.Tensor, 
        values: torch.Tensor, 
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算 GAE (Generalized Advantage Estimation)
        """
        n = len(rewards)
        advantages = torch.zeros(n, device=self.device)
        returns = torch.zeros(n, device=self.device)
        
        gae = 0.0
        next_value = 0.0
        
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "value_net": self.value_net.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "global_step": self.global_step,
            "config": self.config,
        }, path)
        print(f"[Agent] Model saved to {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.value_net.load_state_dict(checkpoint["value_net"])
        if "policy_optimizer" in checkpoint:
            self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        if "value_optimizer" in checkpoint:
            self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])
        if "global_step" in checkpoint:
            self.global_step = checkpoint["global_step"]
        print(f"[Agent] Model loaded from {path}")
    
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
