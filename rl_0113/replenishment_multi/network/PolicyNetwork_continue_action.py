import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork_continue_action(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, action_min = -1, action_max = 1):
        """
        初始化策略网络

        参数:
            state_dim (int): 状态空间维度
            action_dim (int): 动作空间维度
            hidden_dim (int): 隐藏层维度，默认为64
        """
        super(PolicyNetwork_continue_action, self).__init__()
        # print(state_dim)
        # 先试一下只有一层网络
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.action_min, self.action_max = action_min,action_max

    def forward(self, x):
        """
        前向传播函数

        参数:
            x (torch.Tensor): 输入状态张量

        返回:
            mu: 均值
            std: 标准差
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        # 计算均值时映射到动作的取值范围
        mu = (self.action_max - self.action_min) / 2 * torch.tanh(self.fc_mu(x)) + (self.action_min + self.action_max) / 2
        std = F.softplus(self.fc_std(x)) + 1e-6  # 添加一个小的正数偏移量
        std = torch.clamp(std, min=1e-6, max=1)  # 限制std的最大值,最大值先草率设为1，避免高方差
        # 去除多余的维度，否则只输入一个state时，mu和std都是一维张量比如tensor([0.3,])，将其变为tensor(0.3)
        if mu.shape[-1] == 1:
            mu = mu.squeeze(-1)
            std = std.squeeze(-1)
            
        return mu, std
