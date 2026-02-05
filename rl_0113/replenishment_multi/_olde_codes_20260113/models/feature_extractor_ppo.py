import torch.nn as nn
from torch.distributions import Categorical
import sys
import copy
sys.path.append('../')
from network import networks_dict
# from network.PolicyNetwork import PolicyNetwork
# from network.ValueNetwork import ValueNetwork

class feature_extractor_ppoConfig:
    def __init__(self):
        self.share_optimizer = False
        self.share_features_extractor = False
        self.hidden_dim = 64

class feature_extractor_ppo(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super().__init__()
        self.config = config
        PolicyNetwork = networks_dict["PolicyNetwork"]
        ValueNetwork = networks_dict["ValueNetwork"]
        extractor = networks_dict["Base_flatten_feature_extractor"]
        
        # 定义本次用到的优化器类型
        # self.share_optimizer = config.get("share_optimizer", False)
        self.share_optimizer = config.share_optimizer
        # self.share_features_extractor = config.get("share_features_extractor", False)
        self.share_features_extractor = config.share_features_extractor
        # self.hidden_dim = config.get("hidden_dim", 64)
        self.hidden_dim = config.hidden_dim
        
        # 定义feature_extractor
        self.features_extractor = extractor(state_dim = state_dim, hidden_dim = self.hidden_dim)
        if not self.share_features_extractor:
            self.v_features_extractor = extractor(state_dim = state_dim, hidden_dim = self.hidden_dim)
        # else:
        #     self.v_features_extractor = self.features_extractor

        # 定义一个policy模型
        self.policy_model = PolicyNetwork(state_dim = self.hidden_dim, action_dim = action_dim, hidden_dim = self.hidden_dim)
        # 定义一个value模型
        self.value_model = ValueNetwork(state_dim = self.hidden_dim, hidden_dim = self.hidden_dim)



    def select_action(self, probs):
        m = Categorical(probs)
        action = m.sample()
        logprob = m.log_prob(action)
        return logprob, action
    def do_policy(self,x):
        p_feature = self.features_extractor(x)
        probs = self.policy_model(p_feature)
        return probs
    def do_value(self,x):
        if self.share_features_extractor:
            v_feature = self.features_extractor(x)
        else:
            v_feature = self.v_features_extractor(x)
        values = self.value_model(v_feature)
        return values
    def forward(self, x):
        if self.share_features_extractor:
            p_feature = self.features_extractor(x)
            v_feature = p_feature
        else:
            p_feature = self.features_extractor(x)
            v_feature = self.v_features_extractor(x)
        probs = self.policy_model(p_feature)
        values = self.value_model(v_feature)
        logprob, action = self.select_action(probs)
        return probs, values, logprob, action
