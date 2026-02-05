import torch
import torch.nn as nn
from network.PolicyNetwork import PolicyNetwork
from network.ValueNetwork import ValueNetwork
from torch.distributions import Categorical
import torch.optim as optim
class Trainer():
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.share_optimizer = config.share_optimizer
        self.policy_optim_name = self.config.get("optim_name", "Adam")
        self.value_optim_name = self.config.get("optim_name", "Adam")
        self.adjusted_lr = config.lr
    def init_optimizer(self):
        if self.share_optimizer:
            self.policy_optimizer = eval(f"optim.{self.policy_optim_name}")(self.model.parameters())
            self.value_optimizer = None
        else:
            self.policy_optimizer = eval(f"optim.{self.policy_optim_name}")(self.model.policy_model.parameters())
            self.value_optimizer = eval(f"optim.{self.value_optim_name}")(self.model.value_model.parameters())
        return self.policy_optimizer, self.value_optimizer
    def train_step(self,objective_loss, critic_loss):
        if self.share_optimizer:
            objective_loss = objective_loss+critic_loss
        else:
            self.value_optimizer.zero_grad()
            critic_loss.backward()
            self.value_optimizer.step()
        self.policy_optimizer.zero_grad()
        objective_loss.backward()
        self.policy_optimizer.step()
    
class Base_ppo(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super().__init__()
        self.config = config
        # 定义本次用到的优化器类型
        self.share_optimizer = config.get("share_optimizer", False)
        self.share_features_extractor = config.get("share_features_extractor", False)
        self.hidden_dim = config.get("hidden_dim", 64)
        # 定义一个policy模型
        self.policy_model = PolicyNetwork(state_dim = state_dim, action_dim = action_dim, hidden_dim = self.hidden_dim)
        # 定义一个value模型
        self.value_model = ValueNetwork(state_dim = state_dim, hidden_dim = self.hidden_dim)
    def select_action(self, probs):
        m = Categorical(probs)
        action = m.sample()
        logprob = m.log_prob(action)
        return logprob, action
    def train_step(self, objective_loss, critic_loss):
        self.policy_optimizer.zero_grad()
        objective_loss.backward()
        self.policy_optimizer
        if critic_loss and self.value_optimizer:
            self.value_optimizer.zero_grad()
            critic_loss.backward()
            self.value_optimizer.step()
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
