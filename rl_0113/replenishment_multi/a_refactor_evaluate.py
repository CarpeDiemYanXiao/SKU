# 功能点：模型评估

from agents.refactor_agent import Agent
from envs.refactor_replenish_env import ReplenishEnv
from network.PolicyNetwork import PolicyNetwork

import argparse
import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from task import task_dict
from utils.io import read_json
from utils.helper import get_conf_path, create_path_with_suffix
from utils.normalization import Normalization, ZFilter, RewardFilter, DefaultFilter
from atp_sim_sdk.roller import Roller


class GetAction:
    """适配 C++ 接口的动作包装类"""
    def __init__(self, action_ls, action):
        self.action_ls = action_ls
        self.action = action

    def __call__(self, day_idx):
        return self.action_ls[self.action]


class Evaluator:
    def __init__(self, config):
        self.config = config
        self.device = config.device if hasattr(config, 'device') else 'cpu'
        
        # 初始化环境（使用最后一个stage）
        self.env = ReplenishEnv(config, stage_idx=len(config.curriculum_stages) - 1)
        self.sku_id_ls = self.env.sku_id_ls
        self.action_ls = config.action_ls
        self.end_date_map = self.env.end_date_map
        
        # 初始化 agent（仅加载 actor 网络用于推理）
        self.state_dim = config.state_dim
        self.action_dim = len(config.action_ls)
        self.hidden_dim = config.hidden_dim
        
        self.actor = PolicyNetwork(
            state_dim=self.state_dim, 
            action_dim=self.action_dim, 
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        # 加载模型
        self.load_model(config.specified_model_path)
        
        # 初始化归一化器（使用训练时保存的 mean 和 std）
        self.use_state_norm = config.use_state_norm
        self.state_scaling = config.state_scaling
        self.state_norm = eval(self.state_scaling)(shape=self.state_dim, config=config)
        
        # 如果配置中有保存的归一化参数，则加载，检查模型训练完是否保存了最终归一化参数，以及checkpoint是否保存了最终归一化参数，是否能用checkpoint来评估？todo
        if hasattr(config, 'state_norm_mean') and config.state_norm_mean:
            self.state_norm.running_ms.mean = np.array(config.state_norm_mean)
            self.state_norm.running_ms.std = np.array(config.state_norm_std)
            print(f"Loaded state normalization: mean={self.state_norm.running_ms.mean[:3]}..., std={self.state_norm.running_ms.std[:3]}...")
        
        print(f"Evaluator initialized with {len(self.sku_id_ls)} SKUs")

    def load_model(self, model_path):
        """加载训练好的模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor.eval()
        print(f"Model loaded from: {model_path}")
    
    def take_action_deterministic(self, state):
        """确定性策略：选择概率最大的动作"""
        state = torch.from_numpy(np.array(state)).reshape(1, -1).float().to(self.device)
        with torch.no_grad():
            probs = self.actor(state)
            action = torch.argmax(probs, dim=1)
        return action.item()

    def evaluate(self):
        """执行评估"""
        start_time = time.time()
        
        # 统计变量
        actions_map = {sku: [] for sku in self.sku_id_ls}
        actions_ls = []
        head_sku_actions_ls = []
        tail_sku_actions_ls = []
        
        simu_bind_qty = 0
        simu_rts_qty = 0
        simu_rep_qty = 0
        simu_actual_qty = 0
        rolling_cnt = 0
        
        # 初始化状态
        states_map = self.env.reset()
        if self.use_state_norm == 1:
            for sku_id in self.sku_id_ls:
                states_map[sku_id] = self.state_norm(states_map[sku_id], update=False)  # 评估时不更新归一化参数
        
        done_map = {sku_id: False for sku_id in self.sku_id_ls}
        sku_day_map = {sku_id: 0 for sku_id in self.sku_id_ls}
        
        # Rollout
        with torch.no_grad():
            while not all(done_map.values()):
                rolling_cnt += 1
                action_map = {}
                
                # 选择动作
                for sku_id in self.sku_id_ls:
                    if done_map[sku_id]:
                        continue
                    
                    state = states_map[sku_id]
                    action = self.take_action_deterministic(state)
                    
                    # 记录动作
                    actions_map[sku_id].append(self.action_ls[action])
                    actions_ls.append(self.action_ls[action])
                    
                    # 区分头部/尾部品动作
                    if self.env.datasets.order_ratio_7d_map[sku_id][sku_day_map[sku_id]] >= self.config.head_sku_standard:
                        head_sku_actions_ls.append(self.action_ls[action])
                    else:
                        tail_sku_actions_ls.append(self.action_ls[action])
                    
                    # 构造 action_map（适配 C++ 接口）
                    sku_action_map = {
                        "get_action": GetAction(self.action_ls, action),
                        "day_idx": sku_day_map[sku_id],
                        "evaluate": True
                    }
                    action_map[sku_id] = sku_action_map
                    sku_day_map[sku_id] += 1
                
                # 环境 step
                next_states_map, reward_map, new_done_map, info_map = self.env.batch_step(action_map, evaluate=True)
                
                # 统计指标（需要从 skus 中获取）
                for i in range(len(self.env.skus)):
                    sku = self.env.skus[i]
                    sku_id = sku.id.decode("utf-8")
                    if not done_map[sku_id]:
                        simu_actual_qty += self.env.datasets.sales_map[sku_id][sku_day_map[sku_id] - 1]
                        simu_rep_qty += sku.abo_qty
                        simu_bind_qty += sku.lead_time_bind
                        simu_rts_qty += sku.estimate_rts_qty
                
                # 归一化 & 更新状态
                for sku_id in self.sku_id_ls:
                    if done_map[sku_id]:
                        continue
                    if self.use_state_norm == 1:
                        next_states_map[sku_id] = self.state_norm(next_states_map[sku_id], update=False)
                
                # done_map = new_done_map
                done_map = {sku_id: sku_day_map[sku_id] >= self.env.end_date_map[sku_id] - 1 for sku_id in self.sku_id_ls}
                states_map = next_states_map
                
                # 所有 SKU 完成时，写入结果
                if all(done_map.values()):
                    Roller(debug=os.getenv("DEBUG", "False").lower() == "true").add_result_to_csv(
                        simu_bind_qty, simu_rep_qty, simu_rts_qty
                    )
        
        elapsed_time = time.time() - start_time
        
        # 打印结果
        print("#" * 50)
        print("Evaluation Results:")
        print(f"  Rolling count: {rolling_cnt}")
        print(f"  Actual sales: {simu_actual_qty}")
        print(f"  Bind qty: {simu_bind_qty}")
        print(f"  RTS qty: {simu_rts_qty}")
        print(f"  Rep qty: {simu_rep_qty}")
        print(f"  Acc rate: {simu_bind_qty / (simu_actual_qty + 0.001):.4f}")
        print(f"  RTS rate: {simu_rts_qty / (simu_bind_qty + 0.001):.4f}")
        print(f"  Elapsed time: {elapsed_time:.2f}s")
        print("#" * 50)
        
        # 生成结果
        self._save_results(
            actions_ls, head_sku_actions_ls, tail_sku_actions_ls,
            simu_actual_qty, simu_bind_qty, simu_rts_qty
        )
        
        return {
            "actual_qty": simu_actual_qty,
            "bind_qty": simu_bind_qty,
            "rts_qty": simu_rts_qty,
            "acc_rate": simu_bind_qty / (simu_actual_qty + 0.001),
            "rts_rate": simu_rts_qty / (simu_bind_qty + 0.001)
        }

    def _save_results(self, actions_ls, head_sku_actions_ls, tail_sku_actions_ls,
                      actual_qty, bind_qty, rts_qty):
        """保存评估结果"""
        # 动作分布统计
        actions_series = pd.Series(actions_ls)
        action_dist_dict = actions_series.value_counts().to_dict()
        
        head_actions_series = pd.Series(head_sku_actions_ls) if head_sku_actions_ls else pd.Series([])
        head_action_dist_dict = head_actions_series.value_counts().to_dict() if len(head_actions_series) > 0 else {}
        
        tail_actions_series = pd.Series(tail_sku_actions_ls) if tail_sku_actions_ls else pd.Series([])
        tail_action_dist_dict = tail_actions_series.value_counts().to_dict() if len(tail_actions_series) > 0 else {}
        
        # 保存 CSV 结果
        res_df = pd.DataFrame(data={
            "actual_qty": [actual_qty],
            "bind_qty": [bind_qty],
            "rts_qty": [rts_qty],
            "action_dist": [action_dist_dict],
            "head_action_dist": [head_action_dist_dict],
            "tail_action_dist": [tail_action_dist_dict]
        })
        res_df["acc_rate"] = res_df["bind_qty"] / res_df["actual_qty"]
        res_df["rts_rate"] = res_df["rts_qty"] / res_df["bind_qty"] # 这个口径不对，todo
        res_df.to_csv(self.config.res_data_path, index=False)
        print(f"Results saved to: {self.config.res_data_path}")
        
        # 绘制动作分布图
        if head_sku_actions_ls and tail_sku_actions_ls:
            plt.figure(figsize=(10, 6))
            sns.kdeplot(head_sku_actions_ls, fill=True, color="skyblue", label="head_sku")
            sns.kdeplot(tail_sku_actions_ls, fill=True, color="red", label="tail_sku")
            plt.title('Density Plot of Actions')
            plt.xlabel('Action Value')
            plt.ylabel('Density')
            plt.legend()
            
            fig_path = os.path.join(self.config.outs_dir, 'action_distribution.png')
            plt.savefig(fig_path)
            plt.close()
            print(f"Action distribution plot saved to: {fig_path}")


def arg_parser():
    parser = argparse.ArgumentParser(description="Model Evaluation")
    parser.add_argument("--task_name", type=str, default="base", help="task name")
    parser.add_argument("--data_ver", type=str, required=True, help="数据版本")
    parser.add_argument("--para_ver", type=str, required=True, help="实验版本")
    parser.add_argument("--test_version", type=str, default=argparse.SUPPRESS, help="测试后缀")
    parser.add_argument("--specified_model_path", type=str, default=argparse.SUPPRESS, help="指定模型路径")
    parser.add_argument("--test_data_path", type=str, required=True, help="测试数据路径")
    args = parser.parse_args()
    args.base_dir = str(Path(__file__).parents[0])
    return args


if __name__ == "__main__":
    args = arg_parser()
    
    # 加载配置
    config = task_dict["base"]()
    config.update(read_json(get_conf_path(args)), priority="high")
    config.update(args, priority="high")
    
    # 设置结果保存路径
    config.res_data_path = os.path.join(config.outs_dir, "simu_res_data.csv")
    if hasattr(config, "test_version"):
        config.res_data_path = create_path_with_suffix(config.res_data_path, config.test_version)
    
    # 设置模型路径
    if not hasattr(config, "specified_model_path"):
        config.specified_model_path = config.model_filename
    
    # 覆盖数据路径为测试数据
    # 注意：需要修改 curriculum_stages 中的 data_path
    # 这里是不是要改成最后一个stage？因为最后一个stage才是真实的？todo
    config.curriculum_stages[-1]["data_path"] = config.test_data_path
    
    # 单进程评估
    config.distributed = False
    config.rank = 0
    config.world_size = 1
    config.device = "cpu"
    
    # 执行评估
    evaluator = Evaluator(config)
    results = evaluator.evaluate()

