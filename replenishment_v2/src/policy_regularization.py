"""
策略正则化模块 (Policy Regularization)
参考论文: DeepStock: Reinforcement Learning with Policy Regularizations for Inventory Management

核心思想:
1. Base Stock 正则化: a_t = max(π(x_t) - y_t, 0)
   - 网络输出目标库存水平，补货量 = 目标库存 - 当前库存
   
2. Coefficients 正则化: a_t = max(π(x_t)^T · φ(x_t), 0)
   - 网络输出特征系数，补货量 = 系数 × 特征向量
   
3. 组合形式: a_t = max(π(x_t)^T · φ̃(x_t, y_t), 0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class BaseStockNetwork(nn.Module):
    """
    Base Stock 正则化网络
    
    输出目标库存水平 S*，补货量 = max(S* - 当前库存, 0)
    
    优势:
    - 直接学习最优库存水平，降低策略搜索空间
    - 不同库存状态下的最优动作相近，Q目标学习更稳定
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        use_layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        
        # 特征提取层
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # 输出目标库存水平 (标量)
        self.target_head = nn.Linear(hidden_dims[-1], 1)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # 输出层小初始化
        nn.init.orthogonal_(self.target_head.weight, gain=0.01)
        nn.init.zeros_(self.target_head.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        输出目标库存水平
        
        Args:
            state: (batch, state_dim) 状态向量
            
        Returns:
            target_stock: (batch, 1) 目标库存水平 (非负)
        """
        features = self.feature_extractor(state)
        target_stock = F.softplus(self.target_head(features))  # 保证非负
        return target_stock


class CoefficientsNetwork(nn.Module):
    """
    Coefficients 正则化网络
    
    输出特征系数，补货量 = max(系数^T · 特征向量, 0)
    
    特征向量 φ(x_t) 包含:
    - 历史需求特征 (7天/14天平均)
    - 预测需求特征 (leadtime天预测)
    - 常数偏置项
    
    优势:
    - 补货量与关键特征呈正相关，符合业务直觉
    - 提升策略可解释性和泛化能力
    """
    
    def __init__(
        self,
        state_dim: int,
        n_features: int = 5,  # 论文中使用5个特征
        hidden_dims: List[int] = [256, 128, 64],
        use_layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.n_features = n_features
        
        # 特征提取层
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # 输出系数向量 (n_features 维)
        self.coeff_head = nn.Linear(hidden_dims[-1], n_features)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # 系数初始化为小正值
        nn.init.orthogonal_(self.coeff_head.weight, gain=0.01)
        nn.init.constant_(self.coeff_head.bias, 0.1)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        输出特征系数
        
        Args:
            state: (batch, state_dim)
            
        Returns:
            coefficients: (batch, n_features) 系数向量 (非负)
        """
        features = self.feature_extractor(state)
        coefficients = F.softplus(self.coeff_head(features))  # 保证非负
        return coefficients


class BaseStockCoeffNetwork(nn.Module):
    """
    Base Stock + Coefficients 组合正则化网络
    
    补货量 = max(π(x_t)^T · φ̃(x_t, y_t), 0)
    
    其中 φ̃ 包含:
    - 需求相关特征
    - 库存相关特征 (含Base Stock项 -y_t)
    - 常数偏置
    """
    
    def __init__(
        self,
        state_dim: int,
        n_demand_features: int = 4,  # 需求特征数
        hidden_dims: List[int] = [256, 128, 64],
        use_layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        # 总特征数: 需求特征 + 库存特征(-y_t) + 偏置
        self.n_features = n_demand_features + 2
        
        # 特征提取层
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # 输出系数向量
        self.coeff_head = nn.Linear(hidden_dims[-1], self.n_features)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # 系数初始化
        nn.init.orthogonal_(self.coeff_head.weight, gain=0.01)
        # 初始偏置：需求系数为正，库存系数为负（实现Base Stock效果）
        bias_init = torch.tensor([0.2, 0.2, 0.1, 0.1, -0.5, 0.5])[:self.n_features]
        self.coeff_head.bias.data = bias_init
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        输出特征系数
        """
        features = self.feature_extractor(state)
        coefficients = self.coeff_head(features)
        return coefficients


class PolicyRegularizedAgent:
    """
    策略正则化 Agent
    
    支持三种模式:
    1. base_stock: 网络输出目标库存，补货量 = max(目标 - 当前库存, 0)
    2. coefficients: 网络输出系数，补货量 = max(系数 · 特征, 0)
    3. base_coeff: 组合模式
    """
    
    def __init__(
        self,
        config: dict,
        device: str = "cpu",
        regularization_type: str = "base_coeff",
    ):
        self.config = config
        self.device = torch.device(device)
        self.reg_type = regularization_type
        
        # 网络配置
        net_cfg = config.get("network", {})
        env_cfg = config["env"]
        
        n_dynamic = len(env_cfg["state_features"]["dynamic"])
        n_static = len(env_cfg["state_features"]["static"])
        self.state_dim = n_dynamic + n_static
        
        hidden_dims = net_cfg.get("hidden_dims", [256, 128, 64])
        use_layer_norm = net_cfg.get("use_layer_norm", True)
        dropout = net_cfg.get("dropout", 0.0)
        
        # 创建网络
        if regularization_type == "base_stock":
            self.policy_net = BaseStockNetwork(
                state_dim=self.state_dim,
                hidden_dims=hidden_dims,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            )
        elif regularization_type == "coefficients":
            self.policy_net = CoefficientsNetwork(
                state_dim=self.state_dim,
                n_features=5,
                hidden_dims=hidden_dims,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            )
        else:  # base_coeff
            self.policy_net = BaseStockCoeffNetwork(
                state_dim=self.state_dim,
                n_demand_features=4,
                hidden_dims=hidden_dims,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            )
        
        self.policy_net.to(self.device)
        
        # Value网络用于PPO
        from .networks import ValueNetwork
        self.value_net = ValueNetwork(
            state_dim=self.state_dim,
            hidden_dims=hidden_dims,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )
        self.value_net.to(self.device)
    
    def compute_replenish_qty(
        self,
        state: torch.Tensor,
        current_stock: torch.Tensor,
        demand_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        根据策略正则化类型计算补货量
        
        Args:
            state: (batch, state_dim) 完整状态
            current_stock: (batch, 1) 当前库存+在途
            demand_features: (batch, n_features) 需求相关特征
            
        Returns:
            replenish_qty: (batch, 1) 补货量
        """
        if self.reg_type == "base_stock":
            # Base Stock: a = max(π(x) - y, 0)
            target_stock = self.policy_net(state)
            replenish_qty = F.relu(target_stock - current_stock)
            
        elif self.reg_type == "coefficients":
            # Coefficients: a = max(π(x)^T · φ(x), 0)
            coefficients = self.policy_net(state)  # (batch, n_features)
            replenish_qty = F.relu((coefficients * demand_features).sum(dim=-1, keepdim=True))
            
        else:  # base_coeff
            # 组合: a = max(π(x)^T · φ̃(x, y), 0)
            coefficients = self.policy_net(state)  # (batch, n_features)
            # φ̃ = [demand_features, -current_stock, 1]
            extended_features = torch.cat([
                demand_features,
                -current_stock,  # Base Stock项
                torch.ones_like(current_stock),  # 偏置
            ], dim=-1)
            replenish_qty = F.relu((coefficients * extended_features).sum(dim=-1, keepdim=True))
        
        return replenish_qty


def extract_demand_features(
    pred_y: float,
    avg_sales_7d: float,
    avg_sales_14d: float,
    predicts: List[float],
    leadtime: int,
) -> np.ndarray:
    """
    提取需求相关特征向量 φ(x)
    
    论文中使用4个需求特征 + 1个常数偏置:
    - 近期历史需求 (7天平均)
    - 远期历史需求 (14天平均)
    - 近期预测需求 (LT天)
    - 远期预测需求 (LT+7天平均)
    """
    # 7天平均销量
    feat_hist_short = avg_sales_7d if avg_sales_7d > 0 else 0.1
    
    # 14天平均销量
    feat_hist_long = avg_sales_14d if avg_sales_14d > 0 else feat_hist_short
    
    # LT天预测
    feat_pred_lt = pred_y if pred_y > 0 else feat_hist_short * 0.5
    
    # LT后的平均预测
    if len(predicts) > leadtime:
        future_preds = predicts[leadtime:]
        feat_pred_future = np.mean([p if p > 0 else feat_hist_short * 0.3 for p in future_preds])
    else:
        feat_pred_future = feat_hist_short * 0.5
    
    return np.array([feat_hist_short, feat_hist_long, feat_pred_lt, feat_pred_future], dtype=np.float32)
