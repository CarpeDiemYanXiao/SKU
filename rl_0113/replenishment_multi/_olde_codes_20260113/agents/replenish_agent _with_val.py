import math
import pandas as pd
import torch.utils.data
import torch.optim as optim
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from torch.distributions import Categorical
import torch.distributions as distributions
import tqdm
from envs.replenish_env import ReplenishEnv

np.set_printoptions(suppress=True, precision=2, floatmode='fixed')  # 禁用科学计数法

# from network.base_ppo import Trainer
from models import model_dict
from loss import loss_dict
from utils.normalization import Normalization, ZFilter, RewardFilter, DefaultFilter
from utils.advantages import gae_advantages, base_advantages
from utils.trainer import Trainer
from utils.io import read_json, save_json
import os
import logging

import time
from datetime import datetime, timedelta
from utils.log import logging_once
from utils.mlflow_utils import MLflowManager
from torch.utils.tensorboard import SummaryWriter

# 添加 MLflow 导入
import mlflow
import mlflow.pytorch

###TODO： 需要解决多进程中归一化参数保存问题
# 添加分布式训练所需的库
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import socket
from torch.utils.data.distributed import DistributedSampler

"""
policynetwork
state [end_of_stock,forecast]
###state:
end_of_stock: 期末库存
transit_stock:在途库存##
forecast：leadtime当天的值
###
action:  action_ls = [0.5, 1.0, 1.5, 2.0,2.5,3]

###结束状态： 所有商品都完成仿真，

##reward_function

###coverage


"""


def flag_time(nt):
    next_time = time.time()
    print(f"used_time: {next_time - nt}")
    return next_time


class GetAction:
    def __init__(self, action_ls, action):
        self.action_ls = action_ls
        self.action = action

    def __call__(self, day_idx):
        return self.action_ls[self.action]

class Get_Continue_Action:
    def __init__(self, action):
        self.action = action

    def __call__(self, day_idx):
        return self.action


# def base_advantage(rewards, values, dones, gamma=0.9):
#     discounted_reward = 0
#     for reward, done in zip(reversed(rewards), reversed(dones)):
#         if done:
#             discounted_reward = 0
#         discounted_reward = reward + (gamma * discounted_reward)
#         returns.insert(0, discounted_reward)
#     returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
#     advantages = returns - values.detach()


class ReplenishAgent:
    def __init__(self, replenish_model, config):
        # 初始化传进来的replenish_model是none
        # self.config = config
        self.replenish_model = replenish_model
        self.task_name = config.task_name
        self.device = config.device
        self.seed = config.seed if hasattr(config, "seed") else 42
        # self.seed = config.get("seed", 42)
        # 分布式训练参数
        # self.distributed = config.get("distributed", False)
        # config["distributed"] = self.distributed
        self.distributed = config.distributed
        self.rank = config.rank
        # config.get("rank", 0)
        # config["rank"] = self.rank
        self.world_size = config.world_size
        # config["world_size"] = self.world_size
        self.is_master = self.rank == 0
        self.print_every = config.print_every
        
        # config["print_every"] = self.print_every
        # 检查device是否为数字（GPU索引）并处理
        if isinstance(self.device, int) and not torch.cuda.is_available():
            print(f"Warning: CUDA not available, switching to CPU")
            self.device = "cpu"

        ####环境设置
        self.env = ReplenishEnv(
            config.data_path,
            # rts_day=config.rts_days,
            # coverage_weight=config.coverage_weight,
            # rts_weight=config.rts_weight,
            # overnight_weight=config.overnight_weight,
            # rank=self.rank,
            # world_size=config.world_size,
            cfg=config,
        )

        ###输入的数据
        ###商品清单
        # datasets = self.env.datasets
        self.avg_item_qty_7d_map = self.env.datasets.avg_item_qty_7d_map
        self.sku_id_ls = self.env.datasets.sku_ids()
        self.predict_leadtime_day = self.env.datasets.predict_leadtime_day
        self.leadtime_map = self.env.datasets.leadtime_map
        self.initial_stock_map = self.env.datasets.get_initial_stock_map()
        self.end_date_map = self.env.datasets.get_end_date_map()
        self.total_sales = self.env.datasets.total_sales
        self.sales_map = self.env.datasets.sales_map
        self.static_state_map = self.env.datasets.get_static_state_map()
        self.dq_map = self.env.datasets.dq_map

        self.config = config
        self.use_state_norm = config.use_state_norm
        self.use_discount_reward_norm = config.use_discount_reward_norm
        self.state_scaling = config.state_scaling
        # 在config中配置，可选项为["ZFilter", "RewardFilter"]
        self.reward_scaling = config.reward_scaling
        # 在config中配置，可选项为["base_advantages", "gae_advantages"]
        self.advantages_fun = config.advantages_fun
        ###network 配置
        # self.gamma = config["gamma"]
        self.state_label = config.state_label
        self.state_dim = config.state_dim
        self.state_norm = eval(self.state_scaling)(shape=self.state_dim, config=config)
        # 定义reward_norm的类
        self.reward_norm = eval(self.reward_scaling)(shape=1, config=config)
        self.action_ls: list[float] = config.action_ls
        if config.model_name == "ppo_continue_action":
            self.action_dim = 1
        else:
            self.action_dim = len(config.action_ls)
        # self.policy = PolicyNetwork(self.state_dim, self.action_dim).to(self.device)
        # self.value = ValueNetwork(self.state_dim).to(self.device)
        torch.manual_seed(self.seed + self.rank)
        torch.cuda.manual_seed(self.seed + self.rank)
        # ppo_model = model_dict["base_ppo"]
        ppo_model = model_dict[config.model_name]
        self.actorcritic_model = ppo_model(self.state_dim, self.action_dim, self.config).to(self.device)

        self.trainer = Trainer(config, self.actorcritic_model)  # 把agent模型的update单独封装了
        loss_type = loss_dict[config.loss_name]
        self.loss_func = loss_type(config)
        # 如果使用分布式训练，将模型包装为DDP模型
        if config.distributed:
            # 检查是否可以使用CUDA
            if torch.cuda.is_available() and self.device != "cpu":
                self.actorcritic_model = DDP(
                    self.actorcritic_model, device_ids=[self.device], output_device=self.device
                )
            else:
                # CPU上使用DDP
                self.actorcritic_model = DDP(self.actorcritic_model)
        self.policy = self.actorcritic_model.module.do_policy if config.distributed else self.actorcritic_model.do_policy
        # 执行策略网络的前向传播
        self.value = self.actorcritic_model.module.do_value if config.distributed else self.actorcritic_model.do_value
        # 执行价值网络的前向传播

        # 仅在主进程中创建目录和处理文件
        # if not config.distributed or self.is_master:
        #     os.makedirs(os.path.join("output", self.task_name), exist_ok=True)
        self.policy_model_best_reward_filename = config.policy_model_best_reward_filename
        self.policy_model_filename = config.policy_model_filename
        self.value_model_filename = config.value_model_filename

        ###模型训练参数
        # adjusted_lr = config["lr"] * math.sqrt(self.world_size) if self.distributed else config["lr"]
        self.optimizer, self.value_optimizer = self.trainer.init_optimizer()  # 训练过程中没用到，update已经封装在trainer中
        # self.optimizer = optim.Adam(self.policy.parameters(), lr=adjusted_lr)
        # self.value_optimizer = optim.Adam(self.value.parameters(), lr=adjusted_lr)

        self.k_epochs = config.k_epochs  ###k_epochs
        self.eps_clip = config.eps_clip  ###eps_clip
        self.batch_size = config.batch_size  ###batch_size
        self.max_episodes = config.max_episodes
        self.start_episode = 0
        self.current_episode = 0
        if not hasattr(config, "is_checkpoint"):
            self.is_checkpoint = 0
        else:
            self.is_checkpoint = config.is_checkpoint
        if self.is_checkpoint:
            if os.path.exists(self.policy_model_filename):
                self.start_episode = self.load_checkpoint(self.policy_model_filename)
                self.current_episode = self.start_episode

        # 存储训练数据
        ###v {sku:[] for sku in self.sku_id_ls}
        self.states_map = {sku: [] for sku in self.sku_id_ls}
        self.actions_map = {sku: [] for sku in self.sku_id_ls}
        self.logprobs_map = {sku: [] for sku in self.sku_id_ls}
        self.state_values_map = {sku: [] for sku in self.sku_id_ls}
        self.rewards_map = {sku: [] for sku in self.sku_id_ls}
        self.dones_map = {sku: [] for sku in self.sku_id_ls}
        self.episode_rewards_map = {sku: [] for sku in self.sku_id_ls}
        self.episode_rewards = []
        if self.is_master:
            self.logger = SummaryWriter(log_dir=config.logdir)  # tensorboard的log加载器
        if not config.distributed or self.is_master:
            print("initialize done")
            self.total_rolling_time = max(self.env.datasets.get_end_date_map().values())
            self.episode_bar = tqdm.tqdm(range(self.total_rolling_time), desc="Rolling", leave=False)

        enable_mlflow = self.config.use_mlflow and self.is_master
        self.mlops = MLflowManager(
            config.mlflow_host,
            config.mlflow_experiment_name,
            self.task_name,
            enabled=enable_mlflow,
        )
        self.mlops.log_config(config)
        self.mlops.log_dataset(self.config.data_path)

    def verify_ddp_gradients(self, model, rank):
        """打印并验证各层梯度"""
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # 将梯度转换为CPU上的NumPy数组以便打印
                grad_data = param.grad.detach().cpu().numpy()
                grad_sum = np.sum(grad_data)
                grad_mean = np.mean(grad_data)
                print(f"Rank {rank}, Layer {name}: grad_sum={grad_sum:.6f}, grad_mean={grad_mean:.6f}")

    def get_update_action(self, sku_id, state):
        state, action, action_detach, logprob, state_value = self.select_action(state)
        self.states_map[sku_id].append(state)
        self.actions_map[sku_id].append(action_detach)
        self.logprobs_map[sku_id].append(logprob)
        self.state_values_map[sku_id].append(state_value)
        return action

    def select_action(self, state):
        """选择动作"""
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            probs = self.policy(state)  # 网络forward，前向传播
            m = Categorical(probs)
            action = m.sample()
            logprob = m.log_prob(action).detach()
            value = self.value(state)

        return state, action.item(), action.detach(), logprob, value
    
    def select_continue_action(self, state):
        """选择动作"""
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).to(self.device)

            mu,std = self.policy(state)  # 网络forward，前向传播
            action_dist = torch.distributions.Normal(mu, std)
            action = action_dist.sample()
            action = torch.clamp(action, self.config.action_limit[0], self.config.action_limit[1])
            logprob = action_dist.log_prob(action)

            value = self.value(state)

        return state, action.item(), action.detach(), logprob, value

    def select_action_deterministic(self, state):
        """确定性地选择动作，每次选择最优的动作"""
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            if self.config.model_name == "ppo_continue_action":
                mu,std = self.policy(state)
                action = torch.clamp(mu,self.config.action_limit[0],self.config.action_limit[1]).item()

            else:
                probs = self.policy(state)  # 解包返回值，只使用概率
                action = torch.argmax(probs).item()
        return action

    # def select_action_deterministic(self, state):
    #     """调用模型选择最优的action"""
    #     with torch.no_grad():
    #         state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
    #         probs = self.policy(state)  # 解包返回值，只使用概率
    #         action = torch.argmax(probs).item()
    #     return self.action_ls[int(action)]

    def plot_rewards(self):
        """绘制奖励趋势图并保存"""
        # 设置中文字体
        # plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]  # Mac系统
        # plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
        print(f"max episode_rewards = {max(self.episode_rewards)}")
        data = np.array(self.episode_rewards)
        plt.figure(figsize=(12, 6))
        plt.plot(data, linewidth=1)
        # 使用英文标题避免字体问题
        plt.title("Reward Trend for : " + self.task_name)
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.grid(True, linestyle="--", alpha=0.7)
        # 生成时间戳文件名并保存图片
        from datetime import datetime

        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.config.reward_trend_filename
        plt.savefig(filename)
        logging_once(f"reward trend is saved in {filename}", logging.WARNING)
        plt.close()
    
    def cal_stock_level(self,stock, sku, day_idx):
        avg_qty = self.avg_item_qty_7d_map[sku][day_idx] if self.avg_item_qty_7d_map[sku][day_idx] else 0.1
        
        stock_level = stock / avg_qty
        if stock_level > 5:
            return 5
        return stock_level
            
        

    ###TODO
    def gen_network_state(self, state_dict, sku, day_idx):
        """
        生成输入network的state：[end_of_stock,next_day_arrive_order,next_day_rts,forecast,leadtime]
        :return:
        """
        transit_stock = state_dict["transit_stock"] + [0] * (5 - len(state_dict["transit_stock"]))

        if self.state_label == "with_rts_split":
            new_state = [
                state_dict["end_of_stock"],
                state_dict["next_day_rts"],
                transit_stock[0],
                transit_stock[1],
                transit_stock[2],
                transit_stock[3],
                transit_stock[4],
            ] + self.static_state_map[sku][day_idx]
        elif self.state_label == "with_rts_combine":
            start_stock = max(0, state_dict["end_of_stock"] - state_dict["next_day_rts"])
            new_state = [
                    start_stock,
                    transit_stock[0],
                    transit_stock[1],
                    transit_stock[2],
                    transit_stock[3],
                    transit_stock[4],
                ] + self.static_state_map[sku][day_idx]
        elif self.state_label == "with_stock_level":
            start_stock = max(0, state_dict["end_of_stock"] - state_dict["next_day_rts"])
            # avg_qty = self.avg_item_qty_7d_map[sku][day_idx] if self.avg_item_qty_7d_map[sku][day_idx] else 0.1
            # stock_to_sales_ratio = state_dict["end_of_stock"] / avg_qty
            stock_level=self.cal_stock_level(state_dict["end_of_stock"], sku, day_idx)
            new_state = [
                    start_stock,
                    transit_stock[0],
                    transit_stock[1],
                    transit_stock[2],
                    transit_stock[3],
                    transit_stock[4],
                    stock_level  # 新增库存销量水平
                ] + self.static_state_map[sku][day_idx]
        else:
            new_state = [
                state_dict["end_of_stock"],
                transit_stock[0],
                transit_stock[1],
                transit_stock[2],
                transit_stock[3],
                transit_stock[4],
            ] + self.static_state_map[sku][day_idx]
        return new_state

    def reset_network_state(self, sku):
        """
        生成输入network的state
        :return:期末库存，预测值
        """
        
        if self.state_label == "with_rts_split":
            return [self.initial_stock_map[sku], 0, 0, 0, 0, 0, 0] + self.static_state_map[sku][0]  
        elif self.state_label == "with_rts_combine":                
            return [self.initial_stock_map[sku], 0, 0, 0, 0, 0] + self.static_state_map[sku][0]  
        elif self.state_label == "with_stock_level":
            stock_level=self.cal_stock_level(self.initial_stock_map[sku], sku, 0)
            return [self.initial_stock_map[sku], 0, 0, 0, 0, 0,stock_level] + self.static_state_map[sku][0]  
        else:
            return [self.initial_stock_map[sku], 0, 0, 0, 0, 0] + self.static_state_map[sku][0]  

    def reset_day_idx(self):
        return 0

    def gen_update_data(self):
        # if self.use_discount_reward_norm == 1:  # Trick 3:reward normalization
        #     reward_norm = Normalization(shape=1)
        #     #reward_norm.reset()
        states_ls = []
        actions_ls = []
        logprobs_ls = []
        advantages_ls = []
        returns_ls = []
        for sku in self.sku_id_ls:
            # print(len(self.states_map[sku]),self.leadtime_map[sku][0])
            states = torch.stack(self.states_map[sku]).to(self.device)
            # states:26*9,states是一个episode拼接的结果
            if self.config.model_name == "ppo_continue_action":
                actions = torch.tensor(self.actions_map[sku], dtype=torch.float32).to(self.device)
            else:
                actions = torch.tensor(self.actions_map[sku], dtype=torch.int64).to(self.device)
            
            logprobs = torch.stack(self.logprobs_map[sku]).to(self.device)
            state_values = torch.cat(self.state_values_map[sku]).squeeze().to(self.device)
            rewards = torch.tensor(self.rewards_map[sku], dtype=torch.float32).to(self.device)
            dones = torch.tensor(self.dones_map[sku], dtype=torch.float32).to(self.device)
            advantages, returns = eval(self.advantages_fun)(rewards, state_values, dones, self.config)
            states_ls.append(states)
            actions_ls.append(actions)
            logprobs_ls.append(logprobs)
            returns_ls.append(returns)
            advantages_ls.append(advantages)
        # 多卡归一化参数同步
        self.reward_norm.running_ms.set_sample_num(len(self.sku_id_ls))
        self.reward_norm.running_ms.sync_mean_std(self.distributed, rank=self.rank, world_size=self.world_size)
        states_all = torch.cat(states_ls)
        # states_all：2552*9（所有品的state拼接）
        actions_all = torch.cat(actions_ls)
        logprobs_all = torch.cat(logprobs_ls)
        returns_all = torch.cat(returns_ls)
        advantages_all = torch.cat(advantages_ls)
        ### # 标准化优势值
        advantages_all = (advantages_all - advantages_all.mean()) / (advantages_all.std() + 1e-8)

        return states_all, actions_all, logprobs_all, returns_all, advantages_all

    def select_action_batch(self, states_batch):
        """批量选择动作"""
        probs, state_values = self.policy(states_batch)
        m = Categorical(probs)
        actions = m.sample()
        logprobs = m.log_prob(actions)
        values = self.value(states_batch)

        return (states_batch, actions, actions.detach(), logprobs.detach(), values)

    def update(self):
        """更新策略网络"""
        ###生成数据
        states, actions, logprobs, returns, advantages = self.gen_update_data()

        ### PPO更新
        # 将所有经验数据合并为一个数据集，dataset只是当前rank下采样的数据？？？
        dataset = torch.utils.data.TensorDataset(states, actions, logprobs, advantages, returns)

        # 在分布式训练中使用DistributedSampler
        if self.distributed:
            sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)
        else:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # if self.is_master:
        #     total_samples = len(dataset)
        #     print(f"当前env下,数据总量❗️: {total_samples}")

        # PPO更新
        for epoch in range(self.k_epochs):
            logging_once(f"epoch={epoch}| total step {len(dataloader)}", logging.WARNING)
            # if self.is_master:
            #     with open('/home/work/apb-project/ais-deploy-demo-cache/replenishment_vb/replenishment_v16.20/train_data.txt', 'a') as file:
            #         file.write(f"epoch{epoch}❗️❗️❗️❗️❗️❗️❗️❗️:\n")
            # 在每个epoch开始前设置sampler的epoch
            if self.distributed:
                dataloader.sampler.set_epoch(epoch)  # type: ignore
            num_data = 0 # 测试
            for local_step, batch in enumerate(dataloader):
                (
                    batch_states,
                    batch_actions,
                    batch_logprobs,
                    batch_advantages,
                    batch_returns,
                ) = batch
                num_data += batch_states.shape[0]
                # if self.is_master:
                #     with open('/home/work/apb-project/ais-deploy-demo-cache/replenishment_vb/replenishment_v16.20/train_data.txt', 'a') as file:
                #         file.write(f"local_step{local_step}🫡🫡:\n")


                if self.config.model_name == "ppo_continue_action":
                    mu,std, batch_values, logprob, action = self.actorcritic_model(batch_states)
                    m = distributions.Normal(mu, std)
                    # entropy = m.entropy().mean()
                    entropy = 0

                    new_logprobs = m.log_prob(batch_actions)

                    objective_loss, critic_loss = self.loss_func(
                        entropy, new_logprobs, batch_logprobs, batch_advantages, batch_values.squeeze(), batch_returns
                    )
                    # if self.is_master and local_step == 0: # 监控数据
                    #     batch_states_np = batch_states[:10].numpy()
                    #     mu_np = mu[:10].detach().numpy()
                    #     std_np = std[:10].detach().numpy()

                    #     # batch_states_np = np.around(batch_states_np, decimals=2)
                    #     mu_np = np.around(mu_np, decimals=2)
                    #     std_np = np.around(std_np, decimals=2)
                    #     if self.use_state_norm == 1:
                    #         state_mean = self.state_norm.running_ms.mean
                    #         state_std = self.state_norm.running_ms.std
                    #         batch_states_np = batch_states_np * state_std + state_mean
                    #         batch_states_np = np.around(batch_states_np, decimals=2)
                    #     with open('/home/work/apb-project/ais-deploy-demo-cache/replenishment_vb/replenishment_v16.20/train_data.txt', 'a') as file:
                    #         file.write(f"batch_states:{batch_states_np}\n")
                    #         file.write(f"mu:{mu_np}\n")
                    #         file.write(f"std:{std_np}\n")
                    #         file.write(f"entropy:{entropy}\n")
                    #         file.write(f"objective_loss:{objective_loss}\n")
                    #         file.write(f"critic_loss:{critic_loss}\n")
                else:
                    probs, batch_values, logprob, action = self.actorcritic_model(batch_states) # probs是新的，batch_logprobs是旧的，用两个prob计算策略变化的ratio
                    m = Categorical(probs)
                    entropy = m.entropy().mean()
                    new_logprobs = m.log_prob(batch_actions)

                    objective_loss, critic_loss = self.loss_func(
                        entropy, new_logprobs, batch_logprobs, batch_advantages, batch_values.squeeze(), batch_returns
                    )
                if local_step % self.print_every == 0:
                    logging_once(
                        f"step {local_step:>2} p_loss: {objective_loss:.3f}, v_loss: {critic_loss:.3f}", logging.WARNING
                    )
                # objective_loss, critic_loss = self.loss(batch_actions, batch_logprobs, probs, batch_advantages,batch_values.squeeze(),batch_returns)
                ###更新网络
                self.trainer.train_step(objective_loss, critic_loss)
                # self.optimizer.zero_grad()
                # self.value_optimizer.zero_grad()
                # loss.backward()
                # loss_value.backward()
                # self.optimizer.step()
                # self.value_optimizer.step()

            # print(f"每epoch,用于训练的数据量❗️{num_data}")
        # 所有epoch结束后 保存一次模型，仅在主进程中进行
        if not self.distributed or self.is_master:
            # self.save_model(self.config.policy_model_filename)
            self.save_value_model(self.value_model_filename)

        # 清空缓存
        self.states_map = {sku: [] for sku in self.sku_id_ls}
        self.actions_map = {sku: [] for sku in self.sku_id_ls}
        self.logprobs_map = {sku: [] for sku in self.sku_id_ls}
        self.state_values_map = {sku: [] for sku in self.sku_id_ls}
        self.rewards_map = {sku: [] for sku in self.sku_id_ls}
        self.dones_map = {sku: [] for sku in self.sku_id_ls}
        self.episode_rewards_map = {sku: [] for sku in self.sku_id_ls}

    def train(self):
        max_reward = 0
        # Trick 2:state normalization
        # if self.use_reward_norm:  # Trick 3:reward normalization
        #     reward_norm = Normalization(shape=1)

        """模型训练"""
        for episode in range(self.start_episode, self.max_episodes):
            start_time = time.time()
            self.current_episode = episode

            # 只在主进程打印信息
            if not self.distributed or self.is_master:
                print(f"############## episode={episode + 1} #################")
                self.episode_bar.reset(self.total_rolling_time)

            ####所有商品进行初始化
            ###商品network-state初始化，日期初始化
            state_map = {sku: self.reset_network_state(sku) for sku in self.sku_id_ls}  ##idx:state
            if self.use_state_norm == 1:
                for sku in self.sku_id_ls:
                    state_map[sku] = self.state_norm(state_map[sku]) # 调用Normalization实例的默认函数来归一化
            # 设置样本长度，方便一会的mean与std同步
            self.state_norm.running_ms.set_sample_num(len(self.sku_id_ls))
            day_idx_map = {sku: self.reset_day_idx() for sku in self.sku_id_ls}
            done_map = {sku: False for sku in self.sku_id_ls}

            self.env.reset()  ###需要对所有sku进行初始化
            done_all = False
            total_reward = 0
            rts_qty = 0
            over_night = 0
            total_rep = 0
            """智能体执行，获得轨迹数据"""
            stock_lack_num = 0  # 未达到安全库存次数
            while not done_all:
                action_map = {}
                for sku in self.sku_id_ls:
                    sku_action_map = {}
                    if self.config.model_name == "ppo_continue_action":
                        state, action, action_detach, logprob, state_value = self.select_continue_action(state_map[sku]) # 智能体计算动作 
                        sku_action_map["get_action"] = Get_Continue_Action(action) # 这里要改成一个function调用，不然会报错
                    else:
                        state, action, action_detach, logprob, state_value = self.select_action(state_map[sku])  # 智能体计算动作（得到动作索引）
                        sku_action_map["get_action"] = GetAction(self.action_ls, action)  # 获得实际动作
                    
                    sku_action_map["day_idx"] = day_idx_map[sku]
                    action_map[sku] = sku_action_map
                    if not done_map[sku]:
                        self.states_map[sku].append(state)
                        self.actions_map[sku].append(action_detach)
                        self.logprobs_map[sku].append(logprob)
                        self.state_values_map[sku].append(state_value)
                
                """环境step"""
                st = time.time()
                rolling_state_map, result_map, _, _ = self.env.batch_step(action_map)  # 环境step，获得新的state相关信息和reward信息

                ###state_dict
                
                for sku in self.sku_id_ls:
                    if not done_map[sku]:
                        reward = result_map[sku]["reward"]
                        done = result_map[sku]["done"]
                        ###生成policy-network-state
                        ###更新day_idx
                        day_idx_map[sku] += 1
                        new_network_state = self.gen_network_state(rolling_state_map[sku], sku, day_idx_map[sku])  # 获取state（这里获得能直接往网络中传的state）
                        # print("origin_state: ",new_network_state)
                        if self.dq_map[sku][day_idx_map[sku]] >= 0.8 and new_network_state[0] < (self.avg_item_qty_7d_map[sku][day_idx_map[sku]] * self.config.safe_stock_standard) and day_idx_map[sku] > self.leadtime_map[sku][0]:
                            stock_lack_num += 1
                        if self.use_state_norm == 1:
                            new_network_state = self.state_norm(new_network_state)  # 动态归一化
                            # print("norm_state: ",new_network_state)
                        ###更新sku的state
                        state_map[sku] = new_network_state
                        done_map[sku] = done
                        total_reward += reward
                        # reward归一化以及mean和std的更新改为在这里进行
                        if self.use_discount_reward_norm == 1:
                            reward = self.reward_norm(reward)
                        self.rewards_map[sku].append(reward)
                        self.dones_map[sku].append(done)
                        rts_qty += rolling_state_map[sku]["estimate_rts_qty"]
                        overnight_qty = rolling_state_map[sku]["estimate_overnight"]
                        over_night += sum(overnight_qty)
                        total_rep += rolling_state_map[sku]["abo_qty"]

                if set(done_map.values()) == {True}:  ###所有商品都是done才能结束
                    done_all = True
                if not self.distributed or self.is_master:
                    self.episode_bar.update(1)

            # 记录该episode中未达到安全库存的次数
            # if not self.distributed or self.is_master:
            #     with open('/home/work/apb-project/ais-deploy-demo-cache/replenishment_vb/replenishment_v2_6.27/log_info/check_safe_stock_0701.txt', 'a') as file:
            #         file.write(f"episode:{episode},未达到安全库存次数:{stock_lack_num}\n")
            # 更新网络
            self.update()
            # 更新mean和std，这里是更新全局mean和std
            self.state_norm.running_ms.sync_mean_std(self.distributed, rank=self.rank, world_size=self.world_size)
            if self.distributed:
                # 更新mean和std
                stats_tensor = torch.tensor(
                    [total_reward, float(rts_qty), float(over_night), float(total_rep)], device=self.device
                )
                # 在所有进程上收集总和
                dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
                # 更新为全局统计值
                total_reward = stats_tensor[0].item()
                rts_qty = int(stats_tensor[1].item())
                over_night = int(stats_tensor[2].item())
                total_rep = int(stats_tensor[3].item())

            # 只在主进程记录奖励历史
            if not self.distributed or self.is_master:
                self.logger.add_scalar("trian_reward",total_reward,episode)
                self.logger.add_scalar("trian_rep",total_rep,episode)
                self.logger.add_scalar("trian_rts",rts_qty,episode)
                self.logger.add_scalar("trian_over_night",over_night,episode)
                self.episode_rewards.append(total_reward)
                # if (episode + 1) % self.config.save_every_eposide == 0:
                #     # with open("reward_data.json", "w") as file:
                #     #     json.dump(self.episode_rewards, file)
                #     # todo:暂时mask掉，后续再放开
                #     self.plot_rewards()
                end_time = time.time()  # 获取结束时间
                elapsed_time = end_time - start_time  # 计算运行时间

                metrics = {
                    "total_reward": total_reward,
                    "rts_quantity": rts_qty,
                    "overnight_quantity": over_night,
                    "total_replenishment": total_rep,
                    "episode_runtime": elapsed_time,
                    "rts": rts_qty,
                    "replenishment": total_rep,
                }
                self.mlops.log_metrics(metrics, step=episode)

                print(
                    f"episode[{episode + 1}] GLOBAL\ttotal reward: {total_reward:.2f}\trts_qty: {rts_qty}\tover_night: {over_night}\ttotal_rep: {total_rep}\truntime:  {elapsed_time:.2f} s"
                )

            # 在分布式训练中每个进程同步进度
            if self.distributed:
                dist.barrier()
            # 每一个进程都会有一个env，同步四个env之间的mean和std
            ###更新max_reward的值 &保存模型
            if not self.distributed or self.is_master:
                if episode >= self.config.episode_lower_limit and total_reward >= max_reward:
                    ## 更新max_reward的值
                    max_reward = max(max_reward, total_reward)
                    self.config.best_reward = max_reward
                    self.save_model(self.config.policy_model_filename)
                if (episode + 1) % 50 == 0 and episode >= 199:
                    policy_model_epoch = os.path.join(self.config.outs_dir, f"repl_policy_model_{episode + 1}.pth")
                    self.save_model(policy_model_epoch)
        if not self.distributed or self.is_master:
            self.logger.close()

        self.mlops.end_run()

    def cal_date(self, start_date, delta_days):
        end_date = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=delta_days)
        return end_date.strftime("%Y-%m-%d")

    def gen_actions(self, actions_map, start_date, task_name):

        key_ls = []
        res_data_ls = []

        for key, value in actions_map.items():
            key_ls.append(key)
            res_data_ls.append(value)

        df = pd.DataFrame(data={"idx": key_ls, "action": res_data_ls})

        df["action_index"] = df["action"].apply(lambda x: list(range(len(x))))

        exploded_df = df.explode(["action", "action_index"])
        exploded_df["ds"] = exploded_df.apply(lambda s: self.cal_date(start_date, s["action_index"]), axis=1)
        exploded_df["dt_version"] = task_name
        exploded_df = exploded_df[["idx", "ds", "action", "dt_version"]]
        exploded_df.columns = ["idx", "ds", "action", "dt_version"]
        return exploded_df

    def gen_simu_result(self, model_path):
        self.load_model(model_path)
        ##self.load_model_v1(model_path)
        print("print load successfully")
        actions_all = []
        actions_ls = []
        states_all = []
        ####所有商品进行初始化
        ###商品network-state初始化，日期初始化
        actions_map = {sku: [] for sku in self.sku_id_ls}
        state_map = {sku: self.reset_network_state(sku) for sku in self.sku_id_ls}  ##idx:state
        if self.use_state_norm == 1:
            for sku in self.sku_id_ls:
                state_map[sku] = self.state_norm(state_map[sku], update=False)

        states_map = {sku: [] for sku in self.sku_id_ls}
        day_idx_map = {sku: self.reset_day_idx() for sku in self.sku_id_ls}
        done_map = {sku: False for sku in self.sku_id_ls}
        self.env.reset()  ###需要对所有sku进行初始化
        done_all = False
        ###
        simu_rts_qty = 0
        simu_bind_qty = 0

        while not done_all:
            action_map = {}
            for sku in self.sku_id_ls:
                sku_action_map = {}
                states_map[sku].append(state_map[sku])
                ##print(state_map[sku])
                action = self.select_action_deterministic(state_map[sku])
                actions_map[sku].append(self.action_ls[int(action)])
                actions_ls.append(self.action_ls[int(action)])
                if self.config.model_name == "ppo_continue_action":
                    sku_action_map["get_action"] = Get_Continue_Action(action)
                else:
                    sku_action_map["get_action"] = GetAction(self.action_ls, action)
                sku_action_map["evaluate"] = True
                sku_action_map["day_idx"] = day_idx_map[sku]
                action_map[sku] = sku_action_map
                ###TODO如何标识，需要确认
            rolling_state_map, result_map, bind, rts = self.env.batch_step(action_map)
            simu_rts_qty += rts
            simu_bind_qty += bind
            ###state_dict
            for sku in self.sku_id_ls:
                ###生成policy-network-state
                ###更新day_idx
                day_idx_map[sku] += 1
                new_network_state = self.gen_network_state(rolling_state_map[sku], sku, day_idx_map[sku])
                print("origin_state: ", new_network_state)
                if self.use_state_norm == 1:
                    new_network_state = self.state_norm(new_network_state, update=False)
                ###更新sku的state
                state_map[sku] = new_network_state
                done_map[sku] = day_idx_map[sku] >= self.end_date_map[sku] - 1
                ##total_reward += reward

            if set(done_map.values()) == {True}:  ###所有商品都是done才能结束
                done_all = True
        ##self.start_date="2024-12-16"
        print(
            f"total_sales ={self.total_sales} ,simu_rts_qty= {simu_rts_qty}, simu_bind_qty={simu_bind_qty},acc_rate={simu_bind_qty/self.total_sales},rts_rate={simu_rts_qty/simu_bind_qty}"
        )
        # 将列表转换为 Pandas Series
        actions_series = pd.Series(actions_ls)
        action_dist_dict = actions_series.value_counts().to_dict()
        print(f"actions_dist: {action_dist_dict}")
        # multiplier_df = self.gen_multipliers(actions_map, start_date, self.task_name)

        actions_all.append(actions_map)
        states_all.append(states_map)
        total_sales_all = [self.total_sales]
        simu_rts_qty_all = [simu_rts_qty]
        simu_bind_qty_all = [simu_bind_qty]

        res_df = pd.DataFrame(
            data={
                "actual_qty": total_sales_all,
                "bind_qty": simu_bind_qty_all,
                "rts_qty": simu_rts_qty_all,
                "action_dist": action_dist_dict,
            }
        )
        res_df["rts_rate"] = res_df["rts_qty"] / res_df["bind_qty"]
        res_df["acc_rate"] = res_df["bind_qty"] / res_df["actual_qty"]
        res_df = res_df[["actual_qty", "bind_qty", "rts_qty", "acc_rate", "rts_rate", "action_dist"]]
        res_data = os.path.join("output", self.task_name, f"simu_res_data.csv")
        # multiplier_res_data = os.path.join("output", self.task_name, f"simu_multiplier_data.csv")
        res_df.to_csv(res_data, index=False)

        self.mlops.log_eval_metrics(
            self.task_name,
            params={"model_path": model_path, "total_sales": self.total_sales},
            metrics={
                "total_sales": self.total_sales,
                "simu_rts_qty": simu_rts_qty,
                "simu_bind_qty": simu_bind_qty,
                "acc_rate": simu_bind_qty / self.total_sales,
                "rts_rate": simu_rts_qty / simu_bind_qty,
            },
            action_dist_dict=action_dist_dict,
        )
        self.mlops.end_run()

    ###TODO:这种保存方式有问题，需要修改
    def save_model(self, path):
        """保存模型和网络配置"""
        # 只在主进程保存模型
        if self.distributed and not self.is_master:
            return
        model_state_dict = (
            self.actorcritic_model.module.policy_model.state_dict()
            if self.distributed
            else self.actorcritic_model.policy_model.state_dict()
        )
        model_info = {
            "state_dict": model_state_dict,
            "network_type": "",
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "episode": self.current_episode,
            "state_norm_mean": self.state_norm.running_ms.mean.tolist() if self.use_state_norm else [],
            "state_norm_std": self.state_norm.running_ms.std.tolist() if self.use_state_norm else [],
        }
        torch.save(model_info, path)
        logging_once(f"model path: {path}", logging.CRITICAL)

        self.mlops.log_pytorch_model(
            self.actorcritic_model.module.policy_model if self.distributed else self.actorcritic_model.policy_model,
            [0.0] * self.state_dim,
            "policy_model",
        )
        # 会将最新的归一化参数也存入json文件中
        # self.config.state_norm_mean = self.state_norm.running_ms.mean.tolist()
        # self.config.state_norm_std = self.state_norm.running_ms.std.tolist()

    def make_json_file(self, path):
        if not dist.is_initialized() or dist.get_rank() == 0:
            self.config.state_norm_mean = self.state_norm.running_ms.mean.tolist() if self.use_state_norm else []
            self.config.state_norm_std = self.state_norm.running_ms.std.tolist() if self.use_state_norm else []
            save_json(self.config.__dict__, self.config.conf_path, indent=4)

    def save_value_model(self, path):
        """保存模型和网络配置"""
        # 只在主进程保存模型
        if self.distributed and not self.is_master:
            return

        model_state_dict = (
            self.actorcritic_model.module.value_model.state_dict()
            if self.distributed
            else self.actorcritic_model.value_model.state_dict()
        )
        model_info = {
            "state_dict": model_state_dict,
            "network_type": "",
            "state_dim": self.state_dim,
        }
        torch.save(model_info, path)

    def load_model(self, path):
        """加载模型和网络配置"""
        model_info = torch.load(path, map_location=self.device, weights_only=True)
        if self.distributed:
            # 目前的self.policy是一个函数，需要换一种加载方式
            # self.policy.module.load_state_dict(model_info["state_dict"])
            self.actorcritic_model.module.policy_model.load_state_dict(model_info["state_dict"])
        else:
            self.actorcritic_model.policy_model.load_state_dict(model_info["state_dict"])
        self.actorcritic_model.to(self.device)
        if self.use_state_norm == 1:
            self.state_norm = Normalization(shape=model_info["state_dim"], config=self.config)
            self.state_norm.running_ms.mean = np.array(model_info["state_norm_mean"])
            self.state_norm.running_ms.std = np.array(model_info["state_norm_std"])

    # fix me: 加载模型和网络配置代码块，还需要加载value的权重参数
    def load_checkpoint(self, path):
        """加载模型和网络配置"""
        model_info = torch.load(path, map_location=self.device)
        # 加载模型参数
        self.policy.load_state_dict(model_info["state_dict"])
        self.policy.to(self.device)
        self.optimizer.load_state_dict(model_info["optimizer_state_dict"])
        start_episode = model_info["episode"]
        print(f"Checkpoint loaded from {path}. Resuming from episode {start_episode}")
        return start_episode

    def save_to_onnx(self, path):
        if self.distributed and not self.is_master:
            return
        # 存onnx前需要正确加载模型的权重参数
        # model = self.policy.module if self.distributed else self.policy
        model = self.actorcritic_model.module.policy_model if self.distributed else self.actorcritic_model.policy_model
        dummy_input = torch.randn(1, self.state_dim).to(self.device)
        torch.onnx.export(
            model.to("cpu"),
            dummy_input,
            path,
            input_names=["x"],
            output_names=["output"],
            dynamic_axes={"x": {0: "batch"}},  # 指定动态维度
            verbose=True,
        )

        # 使用MLflowManager记录ONNX模型
        self.mlops.log_artifact(path, "onnx_models")


if __name__ == "__main__":

    with open(
        "/home/work/apb-project/ais-deploy-demo-cache/replenishment_test/config/20250329/config_test_1k_baseline.json",
        "r",
    ) as f:
        cfg = json.load(f)
    ra = ReplenishAgent(None, cfg)
    model_path = "/home/work/apb-project/ais-deploy-demo-cache/replenishment/output/test_20250328_1k_baseline/repl_policy_model_2.pth"
    ra.train()
    # ra.load_model_v1(model_path)

    # print("state_mean", ra.state_norm.mean)
    # print("state_std", ra.state_norm.std)
    # ra.save_model("rep_model.pth")
    # ra.plot_rewards()
    # print("end")
