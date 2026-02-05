import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# class PolicyNetwork_continue_action(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim=64, action_min = -1, action_max = 1):
#         """
#         初始化策略网络

#         参数:
#             state_dim (int): 状态空间维度
#             action_dim (int): 动作空间维度
#             hidden_dim (int): 隐藏层维度，默认为64
#         """
#         super(PolicyNetwork_continue_action, self).__init__()
#         # print(state_dim)
#         # 先试一下只有一层网络
#         self.fc1 = nn.Linear(state_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
#         self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
#         self.action_min, self.action_max = action_min,action_max

#     def forward(self, x):
#         """
#         前向传播函数

#         参数:
#             x (torch.Tensor): 输入状态张量

#         返回:
#             mu: 均值
#             std: 标准差
#         """
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         # x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
#         # 计算均值时映射到动作的取值范围啊
#         mu = (self.action_max - self.action_min) / 2 * torch.tanh(self.fc_mu(x)) + (self.action_min + self.action_max) / 2
#         std = F.softplus(self.fc_std(x)) + 1e-6  # 添加一个小的正数偏移量
#         std = torch.clamp(std, min=0.01, max=1)  # 限制std的最大值,最大值先草率设为1，避免高方差
#         # 去除多余的维度，否则只输入一个state时，mu和std都是一维张量比如tensor([0.3,])，将其变为tensor(0.3)
#         # if mu.shape[-1] == 1:
#         #     mu = mu.squeeze(-1)
#         #     std = std.squeeze(-1)
            
#         return mu, std

class PolicyNetwork_continue_action(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, 
                 action_min=0, action_max=1000,  # 补货量通常非负
                 std_min=0.3, std_max=None):      # 可调的标准差范围
        super().__init__()
        
        # 更深的网络 + LayerNorm处理不同量纲
        self.ln_input = nn.LayerNorm(state_dim)
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)  # 额外一层
        
        self.fc_mu = nn.Linear(hidden_dim // 2, action_dim)
        self.fc_std = nn.Linear(hidden_dim // 2, action_dim)
        
        self.action_min = action_min
        self.action_max = action_max
        self.std_min = std_min
        # 标准差上限与动作范围成比例
        self.std_max = std_max if std_max else (action_max - action_min) * 0.1
        # self.std_max = 5
        
        self._init_weights()
    
    def _init_weights(self):
        """正交初始化，有助于RL训练稳定性"""
        for m in [self.fc1, self.fc2, self.fc3]:
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.zeros_(m.bias)
        # 输出层用更小的初始化
        nn.init.orthogonal_(self.fc_mu.weight, gain=0.01)
        nn.init.orthogonal_(self.fc_std.weight, gain=0.01)
    
    def forward(self, x):
        # 输入归一化
        x = self.ln_input(x)
        
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.fc3(x))
        
        # 均值映射到[action_min, action_max]
        mu_normalized = torch.sigmoid(self.fc_mu(x))  # [0,1]
        mu = mu_normalized * (self.action_max - self.action_min) + self.action_min
        
        # 标准差：与动作范围成比例，支持足够探索
        std_normalized = torch.sigmoid(self.fc_std(x))  # [0,1]
        std = std_normalized * (self.std_max - self.std_min) + self.std_min
        
        return mu, std

