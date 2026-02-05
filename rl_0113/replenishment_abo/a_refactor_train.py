# 功能点：agent和env的交互

# 学习率 todo

from agents.refactor_agent import Agent
from envs.refactor_replenish_env import ReplenishEnv
import copy
import gc


from asyncio import Task
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
from utils.normalization import Normalization, ZFilter, RewardFilter, DefaultFilter
from utils.advantages import gae_advantages, base_advantages
from utils.trainer import Trainer
from utils.io import read_json, save_json
import os
import logging
import time
from datetime import datetime, timedelta
from utils.log import logging_once
from torch.utils.tensorboard import SummaryWriter

# 添加 MLflow 导入
# import mlflow
# import mlflow.pytorch
# from utils.mlflow_utils import MLflowManager

###TODO： 需要解决多进程中归一化参数保存问题
# 添加分布式训练所需的库
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import socket


# 没必要这么写，只是为了适配c++接口
class GetAction: # 连续动作
    def __init__(self, action):
        # self.action_ls = action_ls
        self.action = action
    def __call__(self, day_idx): # day_idx也完全没用到
        return self.action

class Trainer():
    def __init__(self, config):
        # # 支持单个 config 或 config 列表
        # if isinstance(config_ls, list):
        #     self.config_ls = config_ls
        #     self.config = config_ls[0]  # 用第一阶段初始化
        # else:
        #     self.config_ls = [config_ls]
        #     self.config = config_ls
        
        self.config = config
        self.stage_ls = config.curriculum_stages
        if isinstance(self.config.device, int) and not torch.cuda.is_available():
            print("Warning: CUDA not available, switching to CPU")
            self.device = "cpu"
        else:
            self.device = self.config.device
        self.seed = self.config.seed if hasattr(self.config, "seed") else 42

        self.distributed = self.config.distributed
        self.rank = self.config.rank
        self.world_size = self.config.world_size
        self.is_master = self.rank == 0
        self.max_reward = -float('inf')
        self.stage_idx = 0
        
        self.agent = Agent(self.config)
        self.env = ReplenishEnv(self.config, stage_idx=0) # 初始化为第一个stage

        self.sku_id_ls = self.env.sku_id_ls
        self.action_ls = self.config.action_ls
        
        self.print_every = self.config.print_every
        # self.max_episodes = self.config.max_episodes # todo
        self.stage_max_episodes = self.env.stage_max_episodes
        # self.start_episode = config.start_episode
        self.start_episode = 0 # 先设置为0，后面更新到config里去 todo
        self.next_episode_to_begin = self.start_episode

        if self.is_master:
            self.writer = SummaryWriter(log_dir=self.config.tb_log_path)  # tensorboard的log加载器
        if not self.config.distributed or self.is_master:
            print("initialize done")
            self.total_step_num = max(self.env.datasets.get_end_date_map().values())
            # self.episode_bar = tqdm.tqdm(range(self.total_step_num), desc="Step", leave=False)
        
        self.use_state_norm = self.config.use_state_norm
        self.use_discount_reward_norm = self.config.use_discount_reward_norm
        self.state_scaling = self.config.state_scaling
        self.state_norm = eval(self.state_scaling)(shape=self.config.state_dim, config=self.config)
        self.reward_scaling = self.config.reward_scaling
        self.reward_norm = eval(self.reward_scaling)(shape=1, config=self.config)
        self.state_norm.running_ms.set_sample_num(len(self.sku_id_ls)) # 获取商品数量，多进程同步mean和std时使用
        self.reward_norm.running_ms.set_sample_num(len(self.sku_id_ls))
        if hasattr(self.config, "checkpoint_path"): 
            if self.config.continue_mode == "resume":
                self.load_checkpoint(self.config.checkpoint_path, mode="resume")
            else:
                self.load_checkpoint(self.config.checkpoint_path, mode="pretrain")



    # def train(self):
    #     # 训练前先保存一次json文件，防止实验设置无法复现
    #     self.save_json_file(self.config.conf_path)
        
    #     max_reward = -float('inf')
    #     for episode in range(self.start_episode, self.max_episodes):
    #         # 训练信息初始化
    #         if not self.distributed or self.is_master:
    #             print(f"############## episode={episode + 1} #################")
    #             self.episode_bar.reset(self.total_step_num)
    #         start_time = time.time()
            
    #         # state初始化
    #         states_map = self.env.reset()
    #         for sku_id in self.sku_id_ls:
    #             if self.use_state_norm == 1:
    #                 states_map[sku_id] = self.state_norm(states_map[sku_id])

    #         done_map = {sku_id: False for sku_id in self.sku_id_ls}
    #         # action_map = {}
    #         # actual_idx_map = {sku_id: None for sku_id in self.sku_id_ls}
    #         sku_day_map = {sku_id: 0 for sku_id in self.sku_id_ls}  # 这里置为0，检查是否支持checkpoint，todo
    #         transition_dict = {sku_id: {"states": [], "actions": [], "rewards": [], "next_states": [], "dones": []} for sku_id in self.sku_id_ls}
            
    #         total_reward = 0
    #         # rollout过程
    #         with torch.no_grad(): # 不保存中间值，加速计算
    #             while not all(done_map.values()):
    #                 action_map = {}
    #                 actual_idx_map = {}
    #                 # 智能体选择动作
    #                 for sku_id in self.sku_id_ls:
    #                     if done_map[sku_id]: continue
    #                     state = states_map[sku_id]
    #                     sku_action_map = {}
    #                     action = self.agent.take_action(state)
    #                     actual_idx_map[sku_id] = action
    #                     ### 这部分action_map只为了匹配c++接口
    #                     sku_action_map["get_action"] = GetAction(self.action_ls, action)
    #                     sku_action_map["day_idx"] = sku_day_map[sku_id]
    #                     action_map[sku_id] = sku_action_map
    #                     sku_day_map[sku_id] += 1
                        
    #                 ### 环境step--这里是当前env的所有sku，批量step,这里已经done的sku如何step，直接跳过吗，中间值现在是计算的，浪费时间？？todo
    #                 next_states_map, reward_map, new_done_map, info_map = self.env.batch_step(action_map)
                    
    #                 # 归一化并收集transition_dict
    #                 for sku_id in self.sku_id_ls:
    #                     if done_map[sku_id]: continue
    #                     if self.use_state_norm == 1:
    #                         next_states_map[sku_id] = self.state_norm(next_states_map[sku_id]) # 归一化，并实时更新mean和std（可选参数update=True）
    #                     if self.use_discount_reward_norm == 1:
    #                         reward_map[sku_id] = self.reward_norm(reward_map[sku_id])
    #                     # 收集transition_dict
    #                     transition_dict[sku_id]["states"].append(states_map[sku_id])
    #                     transition_dict[sku_id]["actions"].append(actual_idx_map[sku_id])
    #                     transition_dict[sku_id]["rewards"].append(reward_map[sku_id])
    #                     transition_dict[sku_id]["next_states"].append(next_states_map[sku_id])
    #                     transition_dict[sku_id]["dones"].append(new_done_map[sku_id])
    #                 total_reward += info_map['total_reward_one_step']
    #                 done_map = new_done_map
    #                 states_map = next_states_map

    #                 # 每一个step更新一次进度条
    #                 if not self.distributed or self.is_master:
    #                     self.episode_bar.update(1)
    #                     self.episode_bar.refresh() # 刷新进度条，显示当前进度，防止进度条不连续
    #         rollout_runtime = time.time() - start_time
    #         # 多进程同步归一化参数
    #         self.reward_norm.running_ms.sync_mean_std(self.distributed, rank=self.rank, world_size=self.world_size)
    #         self.state_norm.running_ms.sync_mean_std(self.distributed, rank=self.rank, world_size=self.world_size)
            
    #         # agent更新网络
    #         self.agent.update(transition_dict,self.writer if self.is_master else None)

    #         # 释放内存
    #         del states_map, action_map, actual_idx_map, sku_day_map, done_map, reward_map, info_map, transition_dict

    #         # 收集总reward
    #         if self.distributed:
    #             reward_tensor = torch.tensor([total_reward], device=self.device)
    #             dist.all_reduce(reward_tensor, op=dist.ReduceOp.SUM)
    #             total_reward = reward_tensor[0].item()
            
    #         # 同步训练进度
    #         if self.distributed:
    #             dist.barrier()
            
    #         one_episode_runtime = time.time() - start_time
            
    #         # tensorboard记录
    #         if not self.distributed or self.is_master:

    #             self.writer.add_scalar("train_reward",total_reward,episode)
    #             # self.episode_rewards.append(total_reward) plt用的数据
    #             print(f"episode[{episode + 1}] GLOBAL\t total reward: {total_reward:.2f}\t rollout_runtime:{rollout_runtime:.2f} s\t one_episode_runtime:{one_episode_runtime:.2f} s")
            
    #         # 保存模型
    #         if not self.distributed or self.is_master:
    #             if episode >= self.config.episode_lower_limit and total_reward >= max_reward:
    #                 ## 更新max_reward的值
    #                 max_reward = max(max_reward, total_reward)
    #                 # self.config.best_reward = max_reward # 这里是干啥用的，貌似没用？？todo
    #                 self.save_policy_model(self.config.policy_model_filename)
    #                 self.save_value_model(self.config.value_model_filename)
    #             if (episode + 1) % 50 == 0 and episode >= 199:
    #                 policy_model_path = os.path.join(self.config.outs_dir, f"repl_policy_model_{episode + 1}.pth")
    #                 value_model_path = os.path.join(self.config.outs_dir, f"repl_value_model_{episode + 1}.pth")
    #                 self.save_policy_model(policy_model_path)
    #                 self.save_value_model(value_model_path)
        
    #         # 每个episode结束时，记录下一个episode的起始episode，暂时没用到，只为保存模型时使用
    #         self.next_episode_to_begin = episode + 1
        
    #     # 训练完全结束后保存onnx文件，防止训练过程中保存的onnx文件不完整
    #     self.save_policy_model_to_onnx(self.config.onnx_path, self.config.policy_model_filename)
    #     self.save_json_file(self.config.conf_path) # 训练完全结束后保存json文件，更新归一化相关参数

    
    
    def save_policy_model(self, path):
        """保存模型和网络配置"""
        # 只在主进程保存模型
        if self.distributed and not self.is_master:
            return
        model_state_dict = (
            self.agent.actor.module.state_dict() # ddp封装后的模型，需要用module访问
            if self.distributed
            else self.agent.state_dict()
        )
        model_info = {
            "state_dict": model_state_dict,
            "network_type": "",
            "state_dim": self.agent.state_dim,
            "action_dim": self.agent.action_dim,
            "optimizer_state_dict": self.agent.actor_optimizer.state_dict(),
            "episode": self.next_episode_to_begin,
            "state_norm_mean": self.state_norm.running_ms.mean.tolist() if self.use_state_norm else [],
            "state_norm_std": self.state_norm.running_ms.std.tolist() if self.use_state_norm else [],
        }
        torch.save(model_info, path)
        logging_once(f"model path: {path}", logging.CRITICAL)
    
    def save_value_model(self, path):
        """保存模型和网络配置"""
        # 只在主进程保存模型
        if self.distributed and not self.is_master:
            return

        model_state_dict = (
            self.agent.critic.module.state_dict() # ddp封装后的模型，需要用module访问
            if self.distributed
            else self.agent.state_dict()
        )
        model_info = {
            "state_dict": model_state_dict,
            "network_type": "",
            "state_dim": self.agent.state_dim,
            "optimizer_state_dict": self.agent.critic_optimizer.state_dict(),
        }
        torch.save(model_info, path)
    
    def save_policy_model_to_onnx(self, onnx_path, best_policy_model_path):
        if self.distributed and not self.is_master:
            return
        
        # 1. 加载最优模型
        checkpoint = torch.load(best_policy_model_path, map_location='cpu')
        
        model = self.agent.actor.module if self.distributed else self.agent.actor # 加载原始模型，因为checkpoint保存的是原始模型的state_dict
        model.load_state_dict(checkpoint["actor_state_dict"])
        
        model.cpu().eval()  # 转cpu了！，因此要在训练完全结束后再保存
        
        dummy_input = torch.randn(1, self.agent.state_dim).cpu()

        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["x"],
            output_names=["output"],
            dynamic_axes={"x": {0: "batch"}},
            verbose=True,
        )
    
    def save_json_file(self, path):
        if not self.distributed or self.is_master: # 只在主进程保存json文件
            self.config.state_norm_mean = self.state_norm.running_ms.mean.tolist() if self.use_state_norm else []
            self.config.state_norm_std = self.state_norm.running_ms.std.tolist() if self.use_state_norm else []
            save_json(self.config.__dict__, self.config.conf_path, indent=4)
    
    def save_checkpoint(self, path, episode):
        """保存完整训练状态（用于断点续训或迁移学习）"""
        if self.distributed and not self.is_master:
            return
        
        checkpoint = {
            # ========== 模型参数 ==========
            "actor_state_dict": (
                self.agent.actor.module.state_dict() if self.distributed 
                else self.agent.actor.state_dict()
            ),
            "critic_state_dict": (
                self.agent.critic.module.state_dict() if self.distributed 
                else self.agent.critic.state_dict()
            ),
            
            # ========== 优化器状态 ==========
            "actor_optimizer_state_dict": self.agent.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.agent.critic_optimizer.state_dict(),
            
            # ========== 训练进度 ==========
            "episode": episode,
            "global_update_step": self.agent.global_update_step,
            "max_reward": self.max_reward,
            "stage_idx": self.stage_idx,
            
            # ========== State 归一化参数 ==========
            "state_norm_mean": self.state_norm.running_ms.mean.tolist() if self.use_state_norm else [],
            "state_norm_std": self.state_norm.running_ms.std.tolist() if self.use_state_norm else [],
            "state_norm_n": self.state_norm.running_ms.n if self.use_state_norm else 0,
            "state_norm_S": self.state_norm.running_ms.S.tolist() if self.use_state_norm else [],
            
            # ========== Reward 归一化参数 ==========
            "reward_norm_mean": self.reward_norm.running_ms.mean.tolist() if self.use_discount_reward_norm else [],
            "reward_norm_std": self.reward_norm.running_ms.std.tolist() if self.use_discount_reward_norm else [],
            "reward_norm_n": self.reward_norm.running_ms.n if self.use_discount_reward_norm else 0,
            "reward_norm_S": self.reward_norm.running_ms.S.tolist() if self.use_discount_reward_norm else [],
            
            # ========== 配置信息（用于验证兼容性）==========
            "state_dim": self.agent.state_dim,
            "action_dim": self.agent.action_dim,
        }
        
        torch.save(checkpoint, path)
        if not self.distributed or self.is_master:
            print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path, mode="resume"):
        """
        加载 checkpoint
        
        Args:
            path: checkpoint 文件路径
            mode: 加载模式
                - "resume": 完全恢复训练（加载所有状态）
                - "pretrain": 仅加载模型权重（用于迁移学习，其他从零开始）
        
        Returns:
            max_reward: 历史最优奖励（resume 模式）或 -inf（pretrain 模式）
            stage_idx: 课程学习阶段索引（resume 模式）或 0（pretrain 模式）
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # ========== 1. 加载模型参数（两种模式都需要）==========
        if self.distributed:
            self.agent.actor.module.load_state_dict(checkpoint["actor_state_dict"])
            self.agent.critic.module.load_state_dict(checkpoint["critic_state_dict"])
        else:
            self.agent.actor.load_state_dict(checkpoint["actor_state_dict"])
            self.agent.critic.load_state_dict(checkpoint["critic_state_dict"])
        
        if mode == "resume":
            # ========== 完全恢复训练 ==========
            
            # 2. 恢复优化器状态
            self.agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
            self.agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
            
            # 3. 恢复训练进度
            self.start_episode = checkpoint["episode"]
            self.next_episode_to_begin = checkpoint["episode"]
            self.agent.global_update_step = checkpoint.get("global_update_step", 0)
            self.max_reward = checkpoint.get("max_reward", -float('inf')) # 这里每次进stage会重置为inf
            self.stage_idx = checkpoint.get("stage_idx", 0)
            
            # 4. 恢复 State 归一化参数
            if self.use_state_norm and checkpoint.get("state_norm_mean"):
                self.state_norm.running_ms.mean = np.array(checkpoint["state_norm_mean"])
                self.state_norm.running_ms.std = np.array(checkpoint["state_norm_std"])
                self.state_norm.running_ms.n = checkpoint.get("state_norm_n", 0)
                if checkpoint.get("state_norm_S"):
                    self.state_norm.running_ms.S = np.array(checkpoint["state_norm_S"])
            
            # 5. 恢复 Reward 归一化参数
            if self.use_discount_reward_norm and checkpoint.get("reward_norm_mean"):
                self.reward_norm.running_ms.mean = np.array(checkpoint["reward_norm_mean"])
                self.reward_norm.running_ms.std = np.array(checkpoint["reward_norm_std"])
                self.reward_norm.running_ms.n = checkpoint.get("reward_norm_n", 0)
                if checkpoint.get("reward_norm_S"):
                    self.reward_norm.running_ms.S = np.array(checkpoint["reward_norm_S"])
            
            if not self.distributed or self.is_master:
                print(f"[Resume] Checkpoint loaded from episode {self.start_episode}, global_step {self.agent.global_update_step}")
            
        elif mode == "pretrain":
            # ========== 仅加载模型权重，其他从零开始 ==========
            
            # 训练进度从 0 开始
            # self.start_episode = 0
            # self.next_episode_to_begin = 0
            # self.agent.global_update_step = 0
            # self.max_reward = -float('inf')，完全重新加载模型，不需要max_reward
            # self.stage_idx = 0
            
            # 归一化参数保持默认（在 __init__ 中已初始化）
            # 优化器状态保持默认（在 Agent.__init__ 中已初始化）
            
            if not self.distributed or self.is_master:
                print(f"[Pretrain] Model weights loaded, training from scratch on new data")
        
        else:
            raise ValueError(f"Unknown mode: {mode}, expected 'resume' or 'pretrain'")
        
        
    
    def train_curriculum(self):
        """课程学习主函数"""
        for idx in range(len(self.stage_ls)):
            if idx != 0:
                self.stage_idx = idx
                
                # 重新加载该阶段的数据/环境
                self.load_stage_env(self.config, stage_idx=self.stage_idx)
            
            print(f"Starting Stage {idx + 1}")
            # 训练该阶段
            self.train_stage(self.stage_max_episodes) # 更改这个字段stage_max_episodes
            
            # 阶段结束，保存 checkpoint
            model_checkpoint_path = os.path.join(self.config.outs_dir, f"checkpoint_stage{self.stage_idx + 1}_model.pth")
            self.save_checkpoint(model_checkpoint_path, episode=self.next_episode_to_begin)
        
        # 因为保存时会把model转cpu，所以要在训练完全结束后再保存
        # 训练完全结束后保存onnx文件，防止训练过程中保存的onnx文件不完整
        self.save_policy_model_to_onnx(self.config.onnx_path, self.config.model_filename)
        self.save_json_file(self.config.conf_path) # 训练完全结束后保存json文件

    def train_stage(self, stage_max_episodes):
        # 重置进度条
        if not self.distributed or self.is_master:
            if hasattr(self, 'episode_bar') and self.episode_bar is not None:
                self.episode_bar.close()
            self.episode_bar = tqdm.tqdm(
                range(self.total_step_num), 
                desc="Step", 
                leave=False,
                position=0  # 固定位置，避免多行冲突
            )
        # 训练前先保存一次json文件，防止实验设置无法复现
        self.save_json_file(self.config.conf_path)
        
        self.max_reward = -float('inf') # todo，这里加一个是否保留max_reward的参数，如果保留，则需要加载checkpoint时恢复max_reward
        begin_episode = self.next_episode_to_begin
        end_episode = begin_episode + stage_max_episodes

        for episode in range(begin_episode, end_episode):
            # 训练信息初始化
            if not self.distributed or self.is_master:
                print(f"############## episode={episode + 1} #################")
                self.episode_bar.reset(self.total_step_num)
            start_time = time.time()
            
            # state初始化
            states_map = self.env.reset()
            for sku_id in self.sku_id_ls:
                if self.use_state_norm == 1:
                    states_map[sku_id] = self.state_norm(states_map[sku_id])

            done_map = {sku_id: False for sku_id in self.sku_id_ls}
            action_map = {sku_id: None for sku_id in self.sku_id_ls}
            actual_idx_map = {sku_id: None for sku_id in self.sku_id_ls}
            sku_day_map = {sku_id: 0 for sku_id in self.sku_id_ls} 
            transition_dict = {sku_id: {"states": [], "actions": [], "rewards": [], "next_states": [], "dones": []} for sku_id in self.sku_id_ls}
            
            # 记录单回合总奖励
            total_reward = 0
            bind_reward,overnight_penalty,rts_penalty,safe_stock_penalty,stockout_penalty = 0,0,0,0,0
            # rollout过程
            with torch.no_grad(): # 不保存中间值，加速计算
                while not all(done_map.values()):
                    # 智能体选择动作
                    for sku_id in self.sku_id_ls:
                        if done_map[sku_id]: continue
                        state = states_map[sku_id]
                        sku_action_map = {}
                        action = self.agent.take_action(state)
                        actual_idx_map[sku_id] = action
                        ### 这部分action_map只为了匹配c++接口
                        sku_action_map["get_action"] = GetAction(action)
                        sku_action_map["day_idx"] = sku_day_map[sku_id]
                        sku_action_map["abo_action"] = round(action) # 改为int
                        action_map[sku_id] = sku_action_map
                        sku_day_map[sku_id] += 1
                        
                    ### 环境step--这里是当前env的所有sku，批量step
                    next_states_map, reward_map, new_done_map, info_map = self.env.batch_step(action_map)
                    
                    # 归一化
                    for sku_id in self.sku_id_ls:
                        if self.use_state_norm == 1:
                            next_states_map[sku_id] = self.state_norm(next_states_map[sku_id]) # 归一化，并实时更新mean和std（可选参数update=True）
                        if self.use_discount_reward_norm == 1:
                            reward_map[sku_id] = self.reward_norm(reward_map[sku_id])
                    # 收集transition_dict
                    for sku_id in self.sku_id_ls:
                        if done_map[sku_id]: continue
                        transition_dict[sku_id]["states"].append(states_map[sku_id])
                        transition_dict[sku_id]["actions"].append(actual_idx_map[sku_id])
                        transition_dict[sku_id]["rewards"].append(reward_map[sku_id])
                        transition_dict[sku_id]["next_states"].append(next_states_map[sku_id])
                        transition_dict[sku_id]["dones"].append(new_done_map[sku_id])
                    total_reward += info_map['total_reward_one_step']
                    bind_reward += info_map['reward_components']['bind_reward']
                    overnight_penalty += info_map['reward_components']['overnight_penalty']
                    rts_penalty += info_map['reward_components']['rts_penalty']
                    safe_stock_penalty += info_map['reward_components']['safe_stock_penalty']
                    stockout_penalty += info_map['reward_components']['stockout_penalty']

                    done_map = new_done_map
                    states_map = next_states_map

                    # 每一个step更新一次进度条
                    if not self.distributed or self.is_master:
                        self.episode_bar.update(1)
                        self.episode_bar.refresh() # 刷新进度条，显示当前进度，防止进度条不连续
            rollout_runtime = time.time() - start_time
            # 多进程同步归一化参数
            self.reward_norm.running_ms.sync_mean_std(self.distributed, rank=self.rank, world_size=self.world_size)
            self.state_norm.running_ms.sync_mean_std(self.distributed, rank=self.rank, world_size=self.world_size)
            
            # agent更新网络
            self.agent.update(transition_dict,self.writer if self.is_master else None)

            # 释放内存
            del states_map, action_map, actual_idx_map, sku_day_map, done_map, reward_map, info_map, transition_dict

            # 收集总reward
            if self.distributed:
                reward_tensor = torch.tensor([total_reward,bind_reward,overnight_penalty,rts_penalty,safe_stock_penalty,stockout_penalty], device=self.device)
                dist.all_reduce(reward_tensor, op=dist.ReduceOp.SUM)
                total_reward = reward_tensor[0].item()
                bind_reward = reward_tensor[1].item()
                overnight_penalty = reward_tensor[2].item()
                rts_penalty = reward_tensor[3].item()
                safe_stock_penalty = reward_tensor[4].item()
                stockout_penalty = reward_tensor[5].item()
            
            # 同步训练进度
            if self.distributed:
                dist.barrier()
            
            one_episode_runtime = time.time() - start_time
            
            # tensorboard记录
            if not self.distributed or self.is_master:

                self.writer.add_scalar("train_reward",total_reward,episode)
                self.writer.add_scalar("bind_reward",bind_reward,episode)
                self.writer.add_scalar("overnight_penalty",overnight_penalty,episode)
                self.writer.add_scalar("rts_penalty",rts_penalty,episode)
                self.writer.add_scalar("safe_stock_penalty",safe_stock_penalty,episode)
                self.writer.add_scalar("stockout_penalty",stockout_penalty,episode)
                # self.episode_rewards.append(total_reward) plt用的数据
                print(f"episode[{episode + 1}] GLOBAL\t total reward: {total_reward:.2f}\t rollout_runtime:{rollout_runtime:.2f} s\t one_episode_runtime:{one_episode_runtime:.2f} s")
            
            # 保存模型
            if not self.distributed or self.is_master:
                if episode >= self.config.episode_lower_limit and total_reward >= self.max_reward:
                    ## 更新max_reward的值
                    self.max_reward = max(self.max_reward, total_reward)
                    self.save_checkpoint(self.config.model_filename, episode)

                    # self.best_reward = max_reward # 这里是干啥用的，貌似没用？？todo
                    # self.save_policy_model(self.config.policy_model_filename)
                    # self.save_value_model(self.config.value_model_filename)
                if (episode + 1) % 50 == 0 and episode >= 199:

                    replenish_model_path = os.path.join(self.config.outs_dir, f"repl_model_{episode + 1}.pth")
                    self.save_checkpoint(replenish_model_path, episode=episode)

                    
                    # policy_model_path = os.path.join(self.config.outs_dir, f"repl_policy_model_{episode + 1}.pth")
                    # value_model_path = os.path.join(self.config.outs_dir, f"repl_value_model_{episode + 1}.pth")
                    
                    # self.save_policy_model(policy_model_path)
                    # self.save_value_model(value_model_path)
            
            # 每个episode结束时，记录下一个episode
            self.next_episode_to_begin = episode + 1   
        
        # 训练完全结束后保存onnx文件，防止训练过程中保存的onnx文件不完整
        # self.save_policy_model_to_onnx(self.config.onnx_path, self.config.model_filename)
        # self.save_json_file(self.config.conf_path) # 训练完全结束后保存json文件
    
    def load_stage_env(self, config,stage_idx):
        """加载当前阶段的数据"""
        # 这里要重新加载数据，是否可优化，todo（例如不改数据，只改配置，但最标准的格式还是改数据）
        # 1. 显式删除旧环境
        # if hasattr(self, 'env') and self.env is not None:
        #     del self.env
        #     gc.collect()  # 强制垃圾回收
        self.env = None
        self.env = ReplenishEnv(config, stage_idx=stage_idx)
        self.sku_id_ls = self.env.sku_id_ls
        self.stage_max_episodes = self.env.stage_max_episodes

        
        # 更新归一化器的 sample_num
        self.state_norm.running_ms.set_sample_num(len(self.sku_id_ls))
        self.reward_norm.running_ms.set_sample_num(len(self.sku_id_ls))
        
        # 更新 step 数
        if not self.config.distributed or self.is_master:
            self.total_step_num = max(self.env.datasets.get_end_date_map().values())
            # self.episode_bar = tqdm.tqdm(range(self.total_step_num), desc="Step", leave=False)
