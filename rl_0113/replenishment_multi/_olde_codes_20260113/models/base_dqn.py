import torch.nn as nn
import torch
from torch.distributions import Categorical
import sys
import numpy as np
sys.path.append('../')
from network import networks_dict
# from network.PolicyNetwork import PolicyNetwork
# from network.ValueNetwork import ValueNetwork

class Base_dqnConfig:
    def __init__(self):
        self.share_optimizer = False
        self.share_features_extractor = False
        self.hidden_dim = 64

class Base_dqn(nn.Module): # ??
    def __init__(self, state_dim, action_dim, config):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        QNetwork = networks_dict["QNetwork"]
        # PolicyNetwork = networks_dict["PolicyNetwork"]
        # ValueNetwork = networks_dict["ValueNetwork"]

        # 定义本次用到的优化器类型
        # self.share_optimizer = config.get("share_optimizer", False)
        self.share_optimizer = config.share_optimizer
        # self.share_features_extractor = config.get("share_features_extractor", False)
        self.share_features_extractor = config.share_features_extractor
        # self.hidden_dim = config.get("hidden_dim", 64)
        self.hidden_dim = config.hidden_dim
        # 定义一个q网络
        self.q_net = QNetwork(state_dim = state_dim, action_dim = action_dim, hidden_dim = self.hidden_dim)
        self.target_q_net = QNetwork(state_dim = state_dim, action_dim = action_dim, hidden_dim = self.hidden_dim)
        # self.policy_model = PolicyNetwork(state_dim = state_dim, action_dim = action_dim, hidden_dim = self.hidden_dim)
        # # 定义一个value模型
        # self.value_model = ValueNetwork(state_dim = state_dim, hidden_dim = self.hidden_dim)
    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            action = self.q_net(state).argmax().item()
        return action
        
    # def do_policy(self,x):
    #     probs = self.policy_model(x)
    #     return probs
    # def do_value(self,x):
    #     values = self.value_model(x)
    #     return values
    # def forward(self, x):
    #     q_values = self.dqn_model(x)

    #     logprob, action = self.select_action(probs)
    #     return probs, values, logprob, action
