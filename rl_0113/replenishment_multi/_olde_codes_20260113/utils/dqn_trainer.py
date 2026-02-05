import torch.optim as optim
import math
class Trainer():
    def __init__(self, config, model):
        self.model = model
        self.share_optimizer = config.share_optimizer
        self.optim_name = config.optim_name
        
        self.adjusted_lr = config.lr * math.sqrt(config.world_size) if config.distributed else config.lr
    def init_optimizer(self):
        
        self.optimizer = eval(f"optim.{self.optim_name}")(self.model.q_net.parameters(), lr=self.adjusted_lr) # 只有q_net参与更新

        return self.optimizer
    
    def train_step(self,objective_loss, critic_loss):
        if self.share_optimizer:
            objective_loss = objective_loss+critic_loss
        else:
        # todo: 这里是临时代码，需要进一步验证
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            objective_loss.backward()
            critic_loss.backward()
            self.policy_optimizer.step()
            self.value_optimizer.step()
    