import torch.nn as nn
import torch
from torch.distributions import Categorical
import sys
sys.path.append('../')
from network import networks_dict
# from network.PolicyNetwork import PolicyNetwork
# from network.ValueNetwork import ValueNetwork

class Base_ppoConfig:
    def __init__(self):
        self.share_optimizer = False
        self.share_features_extractor = False
        self.hidden_dim = 64

class Base_ppo(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super().__init__()
        self.config = config
        PolicyNetwork = networks_dict["PolicyNetwork"]
        ValueNetwork = networks_dict["ValueNetwork"]
        # 定义本次用到的优化器类型
        # self.share_optimizer = config.get("share_optimizer", False)
        self.share_optimizer = config.share_optimizer
        # self.share_features_extractor = config.get("share_features_extractor", False)
        self.share_features_extractor = config.share_features_extractor
        # self.hidden_dim = config.get("hidden_dim", 64)
        self.hidden_dim = config.hidden_dim
        # 定义一个policy模型
        self.policy_model = PolicyNetwork(state_dim = state_dim, action_dim = action_dim, hidden_dim = self.hidden_dim)
        # 定义一个value模型
        self.value_model = ValueNetwork(state_dim = state_dim, hidden_dim = self.hidden_dim)
    def select_action(self, probs):
        m = Categorical(probs)
        action = m.sample()
        logprob = m.log_prob(action)
        return logprob, action
    def do_policy(self,x):
        probs = self.policy_model(x)
        return probs
    def do_value(self,x):
        values = self.value_model(x)
        return values
    def forward(self, x):
        probs = self.policy_model(x)
        values = self.value_model(x)
        logprob, action = self.select_action(probs)
        return probs, values, logprob, action 

class ppo_continue_action(nn.Module): # 这里是个智能体基类，但没有包含update
    def __init__(self, state_dim, action_dim, config):
        super().__init__()
        self.config = config
        
        PolicyNetwork = networks_dict["PolicyNetwork_continue_action"]

        ValueNetwork = networks_dict["ValueNetwork"]
        # 定义本次用到的优化器类型
        # self.share_optimizer = config.get("share_optimizer", False)
        self.share_optimizer = config.share_optimizer
        # self.share_features_extractor = config.get("share_features_extractor", False)
        self.share_features_extractor = config.share_features_extractor
        # self.hidden_dim = config.get("hidden_dim", 64)
        self.hidden_dim = config.hidden_dim
        # 定义一个policy网络,这里没有to device，而是外层调用时直接把整个模型to device了
        self.policy_model = PolicyNetwork(state_dim = state_dim, action_dim = action_dim, hidden_dim = self.hidden_dim, action_min = config.action_limit[0], action_max = config.action_limit[1])
        # 定义一个value网络
        self.value_model = ValueNetwork(state_dim = state_dim, hidden_dim = self.hidden_dim)
    def select_action(self, mu,std):
        action_dist = torch.distributions.Normal(mu, std)
        action = action_dist.sample()
        action = torch.clamp(action, self.config.action_limit[0], self.config.action_limit[1])
        logprob = action_dist.log_prob(action)
        return logprob, action
    def do_policy(self,x):
        probs = self.policy_model(x)
        return probs
    def do_value(self,x):
        values = self.value_model(x)
        return values
    def forward(self, x):
        mu,std = self.policy_model(x)
        values = self.value_model(x)
        logprob, action = self.select_action(mu,std)
        return mu,std, values, logprob, action  # 
