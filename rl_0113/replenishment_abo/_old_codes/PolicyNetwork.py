import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """
        初始化策略网络
        
        参数:
            state_dim (int): 状态空间维度
            action_dim (int): 动作空间维度
            hidden_dim (int): 隐藏层维度，默认为64
        """
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x (torch.Tensor): 输入状态张量
            
        返回:
            tuple: (action_probs, state_values)
                - action_probs: 动作概率分布
                - state_values: 状态值估计
        """
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        action_probs = torch.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)
        return action_probs, state_values