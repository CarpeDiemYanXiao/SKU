"""
神经网络模块
实现 Policy Network 和 Value Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, dim: int, use_layer_norm: bool = True, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.layer_norm = nn.LayerNorm(dim) if use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.layer_norm(x + residual)
        return F.relu(x)


class FeatureExtractor(nn.Module):
    """
    特征提取器
    支持残差连接和层归一化
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        use_layer_norm: bool = True,
        use_residual: bool = True,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1]
        self.use_residual = use_residual
        
        # 激活函数
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.1)
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()
        
        # 输入层归一化
        self.input_norm = nn.LayerNorm(input_dim) if use_layer_norm else nn.Identity()
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.layers = nn.Sequential(*layers)
        
        # 残差块 (如果启用)
        if use_residual and len(hidden_dims) >= 2:
            self.residual_block = ResidualBlock(
                hidden_dims[-1], 
                use_layer_norm=use_layer_norm,
                dropout=dropout
            )
        else:
            self.residual_block = None
        
        self._init_weights()
    
    def _init_weights(self):
        """正交初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.layers(x)
        if self.residual_block is not None:
            x = self.residual_block(x)
        return x


class PolicyNetworkDiscrete(nn.Module):
    """
    离散动作策略网络
    输出动作概率分布
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256, 128],
        use_layer_norm: bool = True,
        use_residual: bool = True,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 特征提取器
        self.feature_extractor = FeatureExtractor(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            use_layer_norm=use_layer_norm,
            use_residual=use_residual,
            dropout=dropout,
            activation=activation,
        )
        
        # 策略头
        self.policy_head = nn.Linear(hidden_dims[-1], action_dim)
        
        # 初始化策略头 (小权重)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: (batch_size, state_dim)
            
        Returns:
            probs: (batch_size, action_dim) 动作概率
        """
        features = self.feature_extractor(state)
        logits = self.policy_head(features)
        probs = F.softmax(logits, dim=-1)
        return probs
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样动作
        
        Args:
            state: (batch_size, state_dim)
            deterministic: 是否使用确定性策略
            
        Returns:
            action: (batch_size,) 动作索引
            log_prob: (batch_size,) 对数概率
        """
        probs = self.forward(state)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        
        log_prob = torch.log(probs.gather(1, action.unsqueeze(-1)) + 1e-8).squeeze(-1)
        return action, log_prob
    
    def evaluate_action(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        评估给定动作
        
        Returns:
            log_prob: 对数概率
            entropy: 熵
        """
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return log_prob, entropy


class PolicyNetworkContinuous(nn.Module):
    """
    连续动作策略网络
    输出高斯分布的均值和标准差
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        hidden_dims: List[int] = [256, 256, 128],
        action_min: float = 0.8,
        action_max: float = 3.0,
        std_min: float = 0.1,
        std_max: float = 0.5,
        use_layer_norm: bool = True,
        use_residual: bool = True,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_min = action_min
        self.action_max = action_max
        self.std_min = std_min
        self.std_max = std_max
        
        # 特征提取器
        self.feature_extractor = FeatureExtractor(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            use_layer_norm=use_layer_norm,
            use_residual=use_residual,
            dropout=dropout,
            activation=activation,
        )
        
        # 均值头
        self.mu_head = nn.Linear(hidden_dims[-1], action_dim)
        
        # 标准差头
        self.std_head = nn.Linear(hidden_dims[-1], action_dim)
        
        # 初始化
        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.orthogonal_(self.std_head.weight, gain=0.01)
        nn.init.zeros_(self.std_head.bias)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns:
            mu: 均值 (映射到 action 范围)
            std: 标准差
        """
        features = self.feature_extractor(state)
        
        # 均值: sigmoid -> [action_min, action_max]
        mu_normalized = torch.sigmoid(self.mu_head(features))
        mu = mu_normalized * (self.action_max - self.action_min) + self.action_min
        
        # 标准差: sigmoid -> [std_min, std_max]
        std_normalized = torch.sigmoid(self.std_head(features))
        std = std_normalized * (self.std_max - self.std_min) + self.std_min
        
        return mu, std
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样动作"""
        mu, std = self.forward(state)
        
        if deterministic:
            action = mu
        else:
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            action = torch.clamp(action, self.action_min, self.action_max)
        
        # 计算 log_prob
        dist = torch.distributions.Normal(mu, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action.squeeze(-1), log_prob
    
    def evaluate_action(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """评估给定动作"""
        mu, std = self.forward(state)
        dist = torch.distributions.Normal(mu, std)
        
        if action.dim() == 1:
            action = action.unsqueeze(-1)
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy


class ValueNetwork(nn.Module):
    """
    价值网络
    估计状态价值 V(s)
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = [256, 256, 128],
        use_layer_norm: bool = True,
        use_residual: bool = True,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        
        self.state_dim = state_dim
        
        # 特征提取器
        self.feature_extractor = FeatureExtractor(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            use_layer_norm=use_layer_norm,
            use_residual=use_residual,
            dropout=dropout,
            activation=activation,
        )
        
        # 价值头
        self.value_head = nn.Linear(hidden_dims[-1], 1)
        
        # 初始化
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Returns:
            value: (batch_size, 1) 状态价值
        """
        features = self.feature_extractor(state)
        value = self.value_head(features)
        return value


def create_networks(config: dict) -> Tuple[nn.Module, nn.Module]:
    """
    根据配置创建网络
    
    Returns:
        policy_network, value_network
    """
    net_cfg = config["network"]
    action_cfg = config["action"]
    env_cfg = config["env"]
    
    # 计算 state_dim
    n_dynamic = len(env_cfg["state_features"]["dynamic"])
    n_static = len(env_cfg["state_features"]["static"])
    state_dim = n_dynamic + n_static
    
    hidden_dims = net_cfg.get("hidden_dims", [256, 256, 128])
    use_layer_norm = net_cfg.get("use_layer_norm", True)
    use_residual = net_cfg.get("use_residual", True)
    dropout = net_cfg.get("dropout", 0.0)
    activation = net_cfg.get("activation", "relu")
    
    # 创建策略网络
    if action_cfg["type"] == "discrete":
        multiplier_range = action_cfg["multiplier_range"]
        multiplier_step = action_cfg["multiplier_step"]
        action_list = np.arange(
            multiplier_range[0], 
            multiplier_range[1] + multiplier_step, 
            multiplier_step
        ).round(2).tolist()
        action_dim = len(action_list)
        
        policy_network = PolicyNetworkDiscrete(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            use_layer_norm=use_layer_norm,
            use_residual=use_residual,
            dropout=dropout,
            activation=activation,
        )
    else:
        policy_network = PolicyNetworkContinuous(
            state_dim=state_dim,
            action_dim=1,
            hidden_dims=hidden_dims,
            action_min=action_cfg["multiplier_range"][0],
            action_max=action_cfg["multiplier_range"][1],
            use_layer_norm=use_layer_norm,
            use_residual=use_residual,
            dropout=dropout,
            activation=activation,
        )
    
    # 创建价值网络
    value_network = ValueNetwork(
        state_dim=state_dim,
        hidden_dims=hidden_dims,
        use_layer_norm=use_layer_norm,
        use_residual=use_residual,
        dropout=dropout,
        activation=activation,
    )
    
    return policy_network, value_network
