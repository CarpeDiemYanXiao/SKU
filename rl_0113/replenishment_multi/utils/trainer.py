import torch.optim as optim
import torch
import math
class Trainer():
    def __init__(self, config, model):
        self.model = model
        self.share_optimizer = config.share_optimizer
        self.policy_optim_name = config.optim_name
        self.value_optim_name = config.optim_name
        self.adjusted_lr = config.lr * math.sqrt(config.world_size) if config.distributed else config.lr
    def init_optimizer(self):
        if self.share_optimizer:
            self.policy_optimizer = eval(f"optim.{self.policy_optim_name}")(self.model.parameters())
            self.value_optimizer = None
        else:
            self.policy_optimizer = eval(f"optim.{self.policy_optim_name}")(self.model.policy_model.parameters(), lr=self.adjusted_lr)
            self.value_optimizer = eval(f"optim.{self.value_optim_name}")(self.model.value_model.parameters(), lr=self.adjusted_lr)
        return self.policy_optimizer, self.value_optimizer
    def train_step(self,objective_loss, critic_loss):
        if self.share_optimizer:
            objective_loss = objective_loss+critic_loss
        else:
        # todo: 这里是临时代码，需要进一步验证
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()

            objective_loss.backward()
            critic_loss.backward()
            # 梯度裁剪
            # torch.nn.utils.clip_grad_norm_(self.model.policy_model.parameters(), max_norm=1.0)
            # torch.nn.utils.clip_grad_norm_(self.model.value_model.parameters(), max_norm=1.0)

            self.policy_optimizer.step()
            self.value_optimizer.step()
    