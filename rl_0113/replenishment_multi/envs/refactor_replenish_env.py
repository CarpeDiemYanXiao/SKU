# 功能点：1.reset函数，2.step函数
from copy import deepcopy
import os
import time
from typing import Callable, Tuple
import gym
import os, psutil, gc

from atp_sim_sdk.entity import SkuInputInfo
from atp_sim_sdk.roller import Roller
from atp_sim_sdk import rolling
from atp_sim_sdk.sku import SKU
from utils.datasets import Datasets
from task.base import TaskConfig
from typing import cast
import numpy as np


class ReplenishEnv(gym.Env):
    end_stock: dict[int, int]
    datasets: Datasets
    state: dict
    day_idx: int

    def __init__(self,cfg: TaskConfig = TaskConfig(),stage_idx: int = 0) -> None:
        super().__init__()
        self.init_args = locals().copy()
        for k in ["self", "cfg", "__class__"]:
            self.init_args.pop(k)
        cfg.update(self.init_args, priority="low")

        # 课程学习阶段env相关参数
        self.stage_name = cfg.curriculum_stages[stage_idx]["stage_name"]
        self.state_static_columns = cfg.curriculum_stages[stage_idx]["state_static_columns"]
        self.stage_max_episodes = cfg.curriculum_stages[stage_idx]["stage_max_episodes"]
        self.data_path = cfg.curriculum_stages[stage_idx]["data_path"]
        self.rts_days = cfg.curriculum_stages[stage_idx]["rts_days"]
        self.rts_weight = cfg.curriculum_stages[stage_idx]["rts_weight"]
        self.lt_bind_weight = cfg.curriculum_stages[stage_idx]["lt_bind_weight"]
        self.until_rts_bind_weight = cfg.curriculum_stages[stage_idx]["until_rts_bind_weight"]
        self.overnight_weight = cfg.curriculum_stages[stage_idx]["overnight_weight"]
        self.stockout_weight = cfg.curriculum_stages[stage_idx]["stockout_weight"]
        self.safe_stock_weight = cfg.curriculum_stages[stage_idx]["safe_stock_weight"]
        self.safe_stock_standard = cfg.curriculum_stages[stage_idx]["safe_stock_standard"]
        self.head_sku_standard = cfg.curriculum_stages[stage_idx]["head_sku_standard"]
        self.reward_type = cfg.curriculum_stages[stage_idx]["reward_type"]


        self.datasets = Datasets(self.data_path, self.state_static_columns, cfg.rank, cfg.world_size)
        self.static_state_map = self.datasets.get_static_state_map() # 静态特征
        self.initial_stock_map = self.datasets.get_initial_stock_map() # 初始库存
        self.state_label = cfg.state_label

        # self.coverage_weight = cfg.coverage_weight            # 这里是旧版本的超参名，现在已经废弃
        self.leadtime_map = self.datasets.leadtime_map
        self.action_ls = cfg.action_ls
        self.end_stock = {} 
        self.day_idx = 0

        self.sku_id_ls = self.datasets.sku_ids()
        self.end_date_map = self.datasets.get_end_date_map()
        self.skus = rolling.init_skus(self.rts_days, self.sku_id_ls, self.datasets) # 这里面存的什么格式？？
        roller = Roller(self.rts_days, os.getenv("DEBUG", "False").lower() == "true")
        if roller.debug:
            roller.save_cur_row("debug.csv", header=True)
        if cfg.rank == 0:print(f"sku_id_ls length: {len(self.sku_id_ls)}")

    def __getstate__(self):
        state = self.__dict__.copy()
        if "pool" in state:
            del state["pool"]
        return state
    
    def reset(self) -> dict[int, list]:
        self.skus = rolling.reset_skus(self.skus) # 这里貌似不是id,而是SKU对象？？
        self.day_idx = 0
        self.end_stock = {}
        # 初始化state_map
        state_map = {}
        for sku_id in self.sku_id_ls:
            # 初始化动态特征 + 第0天的静态特征
            if self.state_label == "with_rts_split":
                state_map[sku_id] = [self.initial_stock_map[sku_id], 0, 0, 0, 0, 0, 0] + self.static_state_map[sku_id][0]  
            elif self.state_label == "with_rts_combine":                
                state_map[sku_id] = [self.initial_stock_map[sku_id], 0, 0, 0, 0, 0] + self.static_state_map[sku_id][0]  
            elif self.state_label == "with_stock_level":
                stock_level=self.cal_stock_level(self.initial_stock_map[sku_id], sku_id, 0)
                state_map[sku_id] = [self.initial_stock_map[sku_id], 0, 0, 0, 0, 0,stock_level] + self.static_state_map[sku_id][0]  
            else:
                state_map[sku_id] = [self.initial_stock_map[sku_id], 0, 0, 0, 0, 0] + self.static_state_map[sku_id][0]  
        return state_map

    def batch_step(self, action_map, evaluate: bool = False):
        # TODO @weibin
        # 这里传进来的 action_map只包含没有done的sku，检查，todo
        # sku.estimate_end_stock leadtime当天期末库存
        # sku.sales[sku.day_index+lead_time_list[sku.day_index]] 销售量 (注意check越界)
        
        next_state_map = {}
        reward_map = {}
        done_map = {}
        info_map = {}

        # 在调用 batch_step 前，确保所有 sku 都有 action，检查❗️todo
        for sku_id in self.sku_id_ls:
            if sku_id not in action_map:
                action_map[sku_id] = {"get_action": lambda x: 1, "day_idx": 0}  # dummy action

        rolling.rolling(self.skus, evaluate, action_map) # rollout,目前这里evaluate会报错，train不会

        # 初始化奖励分量汇总
        total_bind_reward = 0
        total_overnight_penalty = 0
        total_rts_penalty = 0
        total_safe_stock_penalty = 0
        total_stockout_penalty = 0

        # 这里会收集所有sku的信息，其实可能有一部份已经done了，这一步过滤在外层训练，采集transition_dict时完成，todo
        for i in range(len(self.skus)):
            sku = cast(SkuInputInfo, self.skus[i])
            sku_id = self.skus[i].id.decode("utf-8")  # 从字节串转回字符串
            if sku_id not in action_map: continue # 已经done的sku不计算
            day_index = sku.day_index

            # 中间数据
            total_day = self.datasets.end_date_map[sku.id_str]
            lead_time_list = np.ctypeslib.as_array(sku.lead_time, shape=(total_day,))
            overnight_size = sku.ending_stock_list_size  # 声明时已经保持一致
            overnight_array = np.ctypeslib.as_array(sku.overnight_list, shape=(overnight_size,))
            leadtime_sale_qty = sku.sales[day_index+lead_time_list[day_index]]  # leadtime天的销量

            # 获取下一个状态
            next_state_map[sku_id] = self.get_next_state(sku)
            # 计算奖励（返回total_reward和分量字典）
            reward, components = self.cal_reward(self.reward_type,sku,sku_id,overnight_array, leadtime_sale_qty)
            reward_map[sku_id] = reward
            
            # 累加奖励分量
            total_bind_reward += components["bind_reward"]
            total_overnight_penalty += components["overnight_penalty"]
            total_rts_penalty += components["rts_penalty"]
            total_safe_stock_penalty += components["safe_stock_penalty"]
            total_stockout_penalty += components["stockout_penalty"]
            
            # done
            done_map[sku_id] = (self.rts_days + lead_time_list[day_index] + day_index + 2) >= self.end_date_map[sku_id]
        
        # 获取其他信息
        info_map['total_reward_one_step'] = sum(reward_map.values())
        info_map['reward_components'] = {
            "bind_reward": total_bind_reward,
            "overnight_penalty": total_overnight_penalty,
            "rts_penalty": total_rts_penalty,
            "safe_stock_penalty": total_safe_stock_penalty,
            "stockout_penalty": total_stockout_penalty,
        }
        
        return next_state_map, reward_map, done_map, info_map
    

    # ======================================== 状态相关函数 ========================================
    def get_next_state(self, sku):
        sku_id = sku.id.decode("utf-8")  # 从字节串转回字符串
        day_index = sku.day_index
        total_day = self.datasets.end_date_map[sku.id_str]
        lead_time_list = np.ctypeslib.as_array(sku.lead_time, shape=(total_day,))

        orders_np = np.ctypeslib.as_array(sku.orders, shape=(sku.orders_size,))
        transit_stock = orders_np[day_index + 1 : day_index + lead_time_list[day_index] + 2].tolist()

        # 在途
        transit_stock = transit_stock + [0] * (5 - len(transit_stock))
        end_of_stock = sku.end_of_stock
        next_day_rts = sku.rts_qty

        if self.state_label == "with_rts_split":
            next_state = [
                end_of_stock,
                next_day_rts,
                transit_stock[0],
                transit_stock[1],
                transit_stock[2],
                transit_stock[3],
                transit_stock[4],
            ] + self.static_state_map[sku_id][day_index]
        elif self.state_label == "with_rts_combine":
            start_stock = max(0, end_of_stock - next_day_rts)
            next_state = [
                    start_stock,
                    transit_stock[0],
                    transit_stock[1],
                    transit_stock[2],
                    transit_stock[3],
                    transit_stock[4],
                ] + self.static_state_map[sku_id][day_index]
        elif self.state_label == "with_stock_level":
            start_stock = max(0, end_of_stock - next_day_rts)
            stock_level=self.cal_stock_level(end_of_stock, sku_id, day_index)
            next_state = [
                    start_stock,
                    transit_stock[0],
                    transit_stock[1],
                    transit_stock[2],
                    transit_stock[3],
                    transit_stock[4],
                    stock_level  # 新增库存销量水平
                ] + self.static_state_map[sku_id][day_index]
        else:
            next_state = [
                end_of_stock,
                transit_stock[0],
                transit_stock[1],
                transit_stock[2],
                transit_stock[3],
                transit_stock[4],
            ] + self.static_state_map[sku_id][day_index]
        
        return next_state
    # ======================================== 奖励相关函数 ========================================
    def cal_reward(self,reward_type,sku,sku_id, overnight_qty,leadtime_sale_qty):
        """
        返回: (total_reward, reward_components_dict)
        """
        if reward_type=="with_safe_stock":
            return self.cal_reward_safety_stock(sku,sku_id, overnight_qty,leadtime_sale_qty)
        else:
            return self.cal_reward_default(sku, overnight_qty)

    def cal_reward_default(self, sku, overnight_qty):
        bind_reward = self.lt_bind_weight * sku.lead_time_bind
        overnight_penalty = self.overnight_weight * sum(overnight_qty)
        rts_penalty = self.rts_weight * sku.estimate_rts_qty  # 到货后未来14天仍未消耗的 
        total_reward = bind_reward - overnight_penalty - rts_penalty
        
        reward_components = {
            "bind_reward": bind_reward,
            "overnight_penalty": overnight_penalty,
            "rts_penalty": rts_penalty,
            "safe_stock_penalty": 0,
            "stockout_penalty": 0,
        }
        return total_reward, reward_components
    
    def cal_reward_safety_stock(self, sku,sku_id, overnight_qty,leadtime_sale_qty):
        bind_reward = self.lt_bind_weight * sku.lead_time_bind + self.until_rts_bind_weight * (sku.abo_qty - sku.estimate_rts_qty)
        
        # 如果只考虑abo单的绑定量
        # abo_bind_qty = max(sku.abo_qty - sku.estimate_end_stock, 0)
        # bind_reward = self.lt_bind_weight * abo_bind_qty

        overnight_penalty = self.overnight_weight * sum(overnight_qty)
        rts_penalty = self.rts_weight * sku.estimate_rts_qty  # 到货后未来14天仍未消耗的

        # # 递增设置overnight_penalty
        # idx = self.leadtime_map[sku_id][sku.day_index]-1
        # if idx > 0 and overnight_qty[idx-1]!=0: print("idx前一天不为0,overnight_penalty逻辑错误")
        # if overnight_qty[idx] == 0 and sum(overnight_qty) != 0:print("idx不是第一天,overnight_penalty逻辑错误")
        # # 计算本次补货，过夜天数对应的补货量
        # overnight_penalty = self.overnight_weight[0] * sum(overnight_qty[idx:idx+3]) + self.overnight_weight[1] * sum(overnight_qty[idx+3:idx+6])+self.overnight_weight[2] * sum(overnight_qty[idx+6:idx+10])+self.overnight_weight[3] * sum(overnight_qty[idx+10:])

        # 区分头部品
        if self.datasets.order_ratio_7d_map[sku_id][sku.day_index] >= self.head_sku_standard: # 头部品
            stock_gap = max((self.datasets.avg_item_qty_7d_map[sku_id][sku.day_index] * self.safe_stock_standard - sku.estimate_end_stock),0)
            safe_stock_penalty = stock_gap * self.safe_stock_weight
            stockout_penalty = max(0,(leadtime_sale_qty - sku.lead_time_bind) * self.stockout_weight)
        else:
            safe_stock_penalty = 0
            stockout_penalty = 0
        
        total_reward = bind_reward - safe_stock_penalty - overnight_penalty - rts_penalty

        reward_components = {
            "bind_reward": bind_reward,
            "overnight_penalty": overnight_penalty,
            "rts_penalty": rts_penalty,
            "safe_stock_penalty": safe_stock_penalty,
            "stockout_penalty": stockout_penalty,
        }
        return total_reward, reward_components
    
    # ======================================== 其他函数 ========================================
    def cal_stock_level(self,stock, sku_id, day_idx):
        # 使用历史七天销量均值
        avg_qty = self.datasets.avg_item_qty_7d_map[sku_id][day_idx] if self.datasets.avg_item_qty_7d_map[sku_id][day_idx] else 0.1
        stock_level = stock / avg_qty
        # 使用当天预测值
        # temp_day_qty = self.predicts_map[sku][day_idx][0] if self.predicts_map[sku][day_idx][0] else 0.1
        # stock_level = stock / temp_day_qty
        # if stock_level > 5:
        #     return 5
        return stock_level

