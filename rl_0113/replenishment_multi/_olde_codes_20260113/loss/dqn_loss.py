import torch.nn as nn
import torch
from torch.distributions import Categorical
class Surrogate_loss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.eps_clip = config.eps_clip
        if not hasattr(config, "loss_entropy"):
            config.loss_entropy = 0.01
        self.loss_entropy = config.loss_entropy
    def forward(self, actions, logprobs, probs, advantages):
        m = Categorical(probs)
        new_logprobs = m.log_prob(actions)
        entropy = m.entropy().mean()
        ratios = torch.exp(new_logprobs - logprobs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        loss = -torch.min(surr1, surr2).mean() - self.loss_entropy * entropy
        return loss

class Mse_loss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss = nn.MSELoss()
    def forward(self, values, returns):
        # 确保 values 和 returns 的形状一致，避免广播警告
        # 如果 values 是标量（0维），将其转换为与 returns 相同的形状
        if values.dim() == 0:
            values = values.unsqueeze(0)
        # 如果 returns 是标量（0维），将其转换为与 values 相同的形状
        if returns.dim() == 0:
            returns = returns.unsqueeze(0)
        # 如果形状仍然不匹配，确保它们有相同的形状
        if values.shape != returns.shape:
            # 如果元素数量相同，reshape 使其形状一致
            if values.numel() == returns.numel():
                values = values.view_as(returns)
        return self.loss(values, returns)

class dqn_loss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.objective_loss = Surrogate_loss(self.config)
        self.critic_loss = Mse_loss(self.config)
        if not hasattr(config, "share_optimizer"):
            config.share_optimizer = False
        self.share_optimizer = config.share_optimizer
    def forward(self, actions, logprobs, probs, advantages, values, returns):
        objective_loss = self.objective_loss(actions, logprobs, probs, advantages)
        critic_loss = self.critic_loss(values, returns)
        # if self.share_optimizer:
        #     objective_loss = objective_loss+critic_loss
        #     return objective_loss, None
        # else:
        return objective_loss, critic_loss