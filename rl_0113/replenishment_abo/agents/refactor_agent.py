
import torch.utils.data
import torch.optim as optim
import torch
import torch.nn.functional as F
from network.PolicyNetwork import PolicyNetwork
from network.ValueNetwork import ValueNetwork
from network.PolicyNetwork_continue_action import PolicyNetwork_continue_action
import numpy as np
import logging
import math
from utils.log import logging_once
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler
from utils.re_advantages import gae_advantages_torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist





class Agent(torch.nn.Module):
    def __init__(self, config):
        super(Agent, self).__init__() # 或者super().__init__()，py3风格
        self.config = config
        self.device = config.device
        self.state_dim = config.state_dim
        self.action_dim = 1
        self.hidden_dim = config.hidden_dim
        self.global_update_step = 0
        # self.actor_lr = config.actor_lr
        # self.critic_lr = config.critic_lr
        self.adjusted_lr = config.lr * math.sqrt(config.world_size) if config.distributed else config.lr # 缩放学习率
        # self.adjusted_lr = config.lr

        # self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=100, gamma=0.1)
        # self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=100, gamma=0.1)

        self.seed = config.seed if hasattr(config, "seed") else 42

        # 注意，model to gpu ，cpu不留备份，后续网络计算时，输入tensor要to(device)
        self.actor = PolicyNetwork_continue_action(state_dim=self.state_dim, action_dim=self.action_dim, hidden_dim=self.hidden_dim, action_min=config.action_limit[0], action_max=config.action_limit[1]).to(self.device)
        self.critic = ValueNetwork(state_dim=self.state_dim, hidden_dim=self.hidden_dim).to(self.device)
        if config.distributed:
            if torch.cuda.is_available() and self.device != "cpu":
                self.actor = DDP(self.actor, device_ids=[self.device], output_device=self.device)
                self.critic = DDP(self.critic, device_ids=[self.device], output_device=self.device)
            else:
                self.actor = DDP(self.actor)
                self.critic = DDP(self.critic)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.adjusted_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.adjusted_lr)
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        
        self.eps = config.eps_clip
        # self.lmbda = config.lmbda
        self.lmbda = 0.95 # 目前config里没有lmbda参数，先固定为0.95
        self.k_epochs = config.k_epochs


        self.distributed = config.distributed
        self.rank = config.rank
        self.world_size = config.world_size
        self.is_master = self.rank == 0
    
    


    
    def take_action(self, state):
        # state = torch.from_numpy(np.array(state)).reshape(1, -1).float().to(self.device) # (1, state_dim)
        # probs = self.actor(state)
        # action_dist = torch.distributions.Categorical(probs)
        # action = action_dist.sample()
        # return action.item()



        state = torch.from_numpy(np.array(state)).reshape(1, -1).float().to(self.device) # (1, state_dim)
        mu, std = self.actor(state)
        action_dist = torch.distributions.Normal(mu, std)
        action = action_dist.sample()
        action = torch.clamp(action, self.config.action_limit[0], self.config.action_limit[1])
        # logprob = action_dist.log_prob(action)
        return action.item()

    
    def update(self, transition_dict, writer):
        """
        transition_dict:
            - states: (batch_size, state_dim)
            - actions: (batch_size, action_dim)
            - rewards: (batch_size, 1)
            - next_states: (batch_size, state_dim)
            - dones: (batch_size, 1)
        """
        sku_id_ls = list(transition_dict.keys())
        
        # 先收集所有 rewards 计算全局 mean/std（直接用原始数据，不需要转 tensor）
        # all_rewards_raw = []
        # for sku_id in sku_id_ls:
        #     all_rewards_raw.extend(transition_dict[sku_id]["rewards"])
        # reward_mean = np.mean(all_rewards_raw)
        # reward_std = np.std(all_rewards_raw) + 1e-8
        
        # 一个循环完成：数据转换、归一化、计算 td_target 和 advantage
        states_ls, actions_ls, td_target_ls, advantages_ls, old_log_probs_ls = [], [], [], [], []
        
        for sku_id in sku_id_ls:
            states = torch.from_numpy(np.array(transition_dict[sku_id]["states"])).float()
            actions = torch.from_numpy(np.array(transition_dict[sku_id]["actions"])).float().reshape(-1, 1)
            rewards = torch.from_numpy(np.array(transition_dict[sku_id]["rewards"])).float().reshape(-1, 1)
            next_states = torch.from_numpy(np.array(transition_dict[sku_id]["next_states"])).float()
            dones = torch.from_numpy(np.array(transition_dict[sku_id]["dones"])).float().reshape(-1, 1)
            
            # ✅ 诊断1：检查输入数据
            if torch.isnan(states).any():
                print(f"⚠️ NaN in states for {sku_id}")
            if torch.isnan(actions).any():
                print(f"⚠️ NaN in actions for {sku_id}")
            if torch.isnan(rewards).any():
                print(f"⚠️ NaN in rewards for {sku_id}")
            if torch.isinf(rewards).any():
                print(f"⚠️ Inf in rewards for {sku_id}, max={rewards.max()}, min={rewards.min()}")
            
            # trick 1: 标准化reward
            # reward batch 归一化（不是网络内部的batch norm的概念）
            # rewards_normed = (rewards - reward_mean) / reward_std

            # 计算 td_target 和 advantage 时需要 GPU
            states_gpu = states.to(self.device)
            next_states_gpu = next_states.to(self.device)
            rewards_gpu = rewards.to(self.device)
            dones_gpu = dones.to(self.device)
            actions_gpu = actions.to(self.device)

            with torch.no_grad():
                td_target = rewards_gpu + self.gamma * self.critic(next_states_gpu) * (1 - dones_gpu)
                td_delta = td_target - self.critic(states_gpu)
                advantages = gae_advantages_torch(td_delta, self.gamma, self.lmbda)
                mu, std = self.actor(states_gpu)
                action_dist = torch.distributions.Normal(mu, std)
                old_log_probs = action_dist.log_prob(actions_gpu)
                # old_log_probs = torch.log(self.actor(states_gpu).gather(1, actions_gpu)+1e-8)
                
                # ✅ 诊断2：检查 critic 输出
                critic_out = self.critic(states_gpu)
                if torch.isnan(critic_out).any():
                    print(f"⚠️ NaN in critic output")
                
                # ✅ 诊断3：检查 actor 输出
                mu, std = self.actor(states_gpu)
                if torch.isnan(mu).any():
                    print(f"⚠️ NaN in actor mu, states range: [{states_gpu.min():.2f}, {states_gpu.max():.2f}]")
                if torch.isnan(std).any():
                    print(f"⚠️ NaN in actor std")

            # 存回 CPU 节省显存
            states_ls.append(states)
            actions_ls.append(actions)
            td_target_ls.append(td_target.cpu())
            advantages_ls.append(advantages.cpu())
            old_log_probs_ls.append(old_log_probs.cpu())

        # cat 没问题
        states_all = torch.cat(states_ls)
        actions_all = torch.cat(actions_ls)
        td_target_all = torch.cat(td_target_ls)
        advantages_all = torch.cat(advantages_ls)
        old_log_probs_all = torch.cat(old_log_probs_ls)

        # trick 2: 标准化优势值
        advantages_all = (advantages_all - advantages_all.mean()) / (advantages_all.std() + 1e-8)

        # trick 3: 裁剪优势值，避免出现极端值
        advantages_all = torch.clamp(advantages_all, -5, 5)

        dataset = torch.utils.data.TensorDataset(
            states_all, actions_all, td_target_all, advantages_all, old_log_probs_all
        )
        
        sample_ratio = 1 # 使用全部数据，不采样
        num_samples = int(len(dataset) * sample_ratio)
        sampler = RandomSampler(dataset, replacement=False, num_samples=num_samples)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)

        for epoch in range(self.k_epochs):
            for batch in dataloader:
                states, actions, td_targets, advantages, old_log_probs = batch
                
                # ✅ 检查网络权重
                for name, param in self.actor.named_parameters():
                    if torch.isnan(param).any():
                        print(f"⚠️ NaN in actor weight: {name}")
                for name, param in self.critic.named_parameters():
                    if torch.isnan(param).any():
                        print(f"⚠️ NaN in critic weight: {name}")


                # ✅ 统一在这里 to device
                states = states.to(self.device)
                actions = actions.to(self.device)
                td_targets = td_targets.to(self.device)
                advantages = advantages.to(self.device)
                # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # ❗️
                old_log_probs = old_log_probs.to(self.device)

                mean,std = self.actor(states)
                action_dist = torch.distributions.Normal(mean, std)
                # ratio裁剪，避免出现inf，导致nan
                log_probs = action_dist.log_prob(actions)
                log_ratio = torch.clamp(log_probs - old_log_probs, -20, 20)
                ratio = torch.exp(log_ratio)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages
                policy_loss = torch.mean(-torch.min(surr1, surr2)) # 策略损失
                # entropy = -torch.sum(action_dist * torch.log(action_dist + 1e-8), dim=1).mean()
                entropy = action_dist.entropy().mean()
                actor_loss = policy_loss # 熵系数一般 0.01~0.1
                value_pred = self.critic(states) # 带参数的，不detach
                critic_loss = F.mse_loss(value_pred, td_targets)

                # 聚合（所有进程都要调用）
                value_loss_avg = self.reduce_mean(critic_loss.clone()).item()
                policy_loss_avg = self.reduce_mean(actor_loss.clone()).item()
                exp_var_tensor = 1 - (td_targets - value_pred.detach()).var() / (td_targets.var()+ 1e-8) # 计算解释方差
                exp_var_avg = self.reduce_mean(exp_var_tensor).item()
                
                # 只在 rank 0 记录
                if self.rank == 0:
                    writer.add_scalar('loss/value', value_loss_avg, self.global_update_step)
                    writer.add_scalar('loss/policy', policy_loss_avg, self.global_update_step)
                    writer.add_scalar('value/explained_var', exp_var_avg, self.global_update_step)
                    writer.add_scalar('entropy', entropy.item(), self.global_update_step)
                self.global_update_step += 1

                # 加这段代码到训练循环
                if self.global_update_step % 100 == 0 and self.rank == 0:
                    with torch.no_grad():
                        print(f"Step {self.global_update_step}")
                        print(f"  Mean (mu) - avg: {mean.mean().item():.4f}, std: {mean.std().item():.4f}")
                        print(f"  Std (sigma) - avg: {std.mean().item():.4f}, min: {std.min().item():.4f}, max: {std.max().item():.4f}")
                        print(f"  Entropy: {entropy.item():.4f}")
                        print(f"td_target mean: {td_targets.mean().item():.2f}, std: {td_targets.std().item():.2f}")
                        print(f"value_pred mean: {value_pred.mean().item():.2f}, std: {value_pred.std().item():.2f}")
                        print(f"advantage mean: {advantages.mean().item():.6f}")
                        print(f"advantage std: {advantages.std().item():.6f}")
                        print(f"advantage max: {advantages.max().item():.6f}")
                        print(f"advantage min: {advantages.min().item():.6f}")
                        print(f"ratio mean: {ratio.mean().item():.4f}")
                        print(f"policy_loss: {policy_loss.item():.6f}")
                        print(f"entropy_term: {0.001 * entropy.item():.6f}")
                        print(f"actor_loss: {actor_loss.item():.6f}")
                        

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
    
    def reduce_mean(self, tensor):
        """聚合所有进程的值，返回平均"""
        if not self.distributed:
            return tensor
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor, device=self.device)
        tensor = tensor.clone()  # 避免修改原 tensor
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= self.world_size 
        return tensor


