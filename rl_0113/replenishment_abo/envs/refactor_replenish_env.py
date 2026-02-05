# 功能点：1.reset函数，2.step函数
from copy import deepcopy
from operator import le
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

    def __init__(self, cfg: TaskConfig = TaskConfig(), stage_idx: int = 0) -> None:
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
        self.actuals_map = self.datasets.actuals_map if hasattr(self.datasets, "actuals_map") else None

        self.static_state_map = self.datasets.get_static_state_map()  # 静态特征
        self.initial_stock_map = self.datasets.get_initial_stock_map()  # 初始库存
        self.state_label = cfg.state_label
        self.action_type = cfg.action_type if hasattr(cfg, "action_type") else "multiplier"  # abo_qty

        # self.coverage_weight = cfg.coverage_weight            # 这里是旧版本的超参名，现在已经废弃
        self.leadtime_map = self.datasets.leadtime_map
        self.action_ls = cfg.action_ls
        self.end_stock = {}
        self.day_idx = 0

        self.sku_id_ls = self.datasets.sku_ids()
        self.end_date_map = self.datasets.get_end_date_map()
        self.skus = rolling.init_skus(self.rts_days, self.sku_id_ls, self.datasets)  # 这里面存的什么格式？？
        roller = Roller(self.rts_days, os.getenv("DEBUG", "False").lower() == "true")
        self.rank = cfg.rank
        if roller.debug:
            roller.save_cur_row("debug.csv", header=True)
        if cfg.rank == 0:
            print(f"sku_id_ls length: {len(self.sku_id_ls)}")
        # self.action_type = cfg.action_type if cfg.action_type in self.action_ls else "multiplier"

    def __getstate__(self):
        state = self.__dict__.copy()
        if "pool" in state:
            del state["pool"]
        return state

    def reset(self) -> dict[int, list]:
        self.skus = rolling.reset_skus(self.skus)  # 这里貌似不是id,而是SKU对象？？
        self.day_idx = 0
        self.end_stock = {}
        # 初始化state_map
        state_map = {}
        for sku_id in self.sku_id_ls:

            actuals_list = self.datasets.actuals_map[sku_id][0]
            actuals_list = actuals_list[:self.rts_days+1]
            actuals_list = actuals_list + [0] * (6 - len(actuals_list))
            lt_sell_qty = actuals_list[self.leadtime_map[sku_id][0]]


            
            # 初始化动态特征 + 第0天的静态特征
            if self.state_label == "with_rts_split":
                state_map[sku_id] = [self.initial_stock_map[sku_id], 0, 0, 0, 0, 0, 0] + self.static_state_map[sku_id][
                    0
                ]
            elif self.state_label == "with_rts_combine":
                # 6个动态特征 + 6个rts_qty_list(初始为0) + 静态特征
                
                # state_map[sku_id] = [self.initial_stock_map[sku_id]]
                #     + [0, 0, 0, 0, 0] # 在途
                #     + actuals_list # 6
                #     + [0, 0, 0, 0, 0] # rts
                #     + [0,lt_sell_qty] # 假设lt期初为0
                #     + self.static_state_map[sku_id][0] # lt和统计特征

                state_map[sku_id] = self.static_state_map[sku_id][0] + [0,0] # 这里lt期初先草率设置为0了，todo

            elif self.state_label == "with_stock_level":
                stock_level = self.cal_stock_level(self.initial_stock_map[sku_id], sku_id, 0)
                state_map[sku_id] = [
                    self.initial_stock_map[sku_id],
                    0,0,0,0,0,
                    0,0,0,0,0,0,
                    stock_level,
                ] + self.static_state_map[sku_id][0]
            else:
                state_map[sku_id] = [self.initial_stock_map[sku_id], 0, 0, 0, 0, 0] + self.static_state_map[sku_id][0]
        return state_map

    def batch_step(self, action_map, evaluate: bool = False):

        next_state_map = {}
        reward_map = {}
        done_map = {}
        info_map = {}

        for sku_id in self.sku_id_ls:
            if sku_id not in action_map:
                action_map[sku_id] = {"get_action": lambda x: 1, "day_idx": 0}  # dummy action

        rolling.rolling(self.skus, evaluate, self.action_type, action_map)  # rollout,目前这里evaluate会报错，train不会

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
            if sku_id not in action_map:
                continue  # 已经done的sku不计算
            day_index = sku.day_index

            # 中间数据
            total_day = self.datasets.end_date_map[sku.id_str]
            lead_time_list = np.ctypeslib.as_array(sku.lead_time, shape=(total_day,))
            overnight_size = sku.ending_stock_list_size  # 声明时已经保持一致
            overnight_array = np.ctypeslib.as_array(sku.overnight_list, shape=(overnight_size,))
            leadtime_sale_qty = sku.sales[day_index + lead_time_list[day_index]]  # leadtime天的销量

            # 获取下一个状态
            next_state_map[sku_id] = self.get_next_state(sku)
            # 计算奖励（返回total_reward和分量字典）
            reward, components = self.cal_reward(self.reward_type, sku, sku_id, overnight_array, leadtime_sale_qty)
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
        info_map["total_reward_one_step"] = sum(reward_map.values())
        info_map["reward_components"] = {
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
        day_index = sku.day_index # 这里取的是补货日（比如rollout一步，这里是0）
        total_day = self.datasets.end_date_map[sku.id_str]
        lead_time_list = np.ctypeslib.as_array(sku.lead_time, shape=(total_day,))

        orders_np = np.ctypeslib.as_array(sku.orders, shape=(sku.orders_size,))
        
        # 从第二天开始，共leadtime天（这里有问题，多一天⚠️，不影响结果，todo，已修改）
        transit_stock = orders_np[day_index + 1 : day_index + lead_time_list[day_index] + 1].tolist()

        # 在途
        transit_stock = transit_stock + [0] * (5 - len(transit_stock))
        end_of_stock = sku.end_of_stock
        next_day_rts = sku.rts_qty
        # 将 ctypes 指针转换为 numpy array
        rts_qty_list = np.ctypeslib.as_array(sku.estimate_rts_qty_list, shape=(sku.orders_size,))
        
        # 临时调试：写入文件
        # 调试日志：每个episode开始时清空，之后追加
        # if self.rank == 0:
        if sku_id == "103603474660-764" and day_index%5==0:
            # 每次训练开始时重置一次，之后追加
            if not hasattr(self, '_log_reset_done'):
                self._log_reset_done = True
                write_mode = 'w'  # 首次写入，清空文件
            else:
                write_mode = 'a'  # 后续追加
            with open('/Users/weibin/Desktop/weibin_projects/replenishment_conti/log_info/rts_debug.log', write_mode) as f:
                non_zero_values = rts_qty_list[:20].tolist()
                f.write(f"[rank 0] day:{sku.day_index}, sku_id: {sku_id},leadtime: {lead_time_list[day_index]}, rts_qty_list: {non_zero_values}\n")
            with open('/Users/weibin/Desktop/weibin_projects/replenishment_conti/log_info/transit_stock_debug.log', write_mode) as f:
                f.write(f"[rank 0] day:{sku.day_index}, sku_id: {sku_id},leadtime: {lead_time_list[day_index]}, transit_stock: {transit_stock}\n")
        
        rts_qty_list = rts_qty_list[1:lead_time_list[day_index]+1].tolist()
        rts_qty_list = rts_qty_list + [0] * (5 - len(rts_qty_list)) # 从next_day开始的rts，长度固定为5
        # rts_qty_list[0] = next_day_rts  # 第一维永远是0，手动置为next_day_rts


        # 滚动计算leadtime天的期初库存
        # 公式: begin_stock = end_stock + transit_stock
        #       end_stock = max(0, begin_stock - sell_qty) - rts_qty
        leadtime = lead_time_list[day_index]
        rolling_end_stock = end_of_stock  # 当前期末库存，已扣除rts
        
        # 获取未来预测销量 (用于模拟)
        # predicts_ptr = sku.predicts
        actuals_by_next_day = self.datasets.actuals_map[sku_id][day_index+1]  # [a0, a1, a2, a3, a4, a5]
        actuals_by_next_day = actuals_by_next_day[:leadtime+1]
        actuals_by_next_day = actuals_by_next_day + [0] * (6 - len(actuals_by_next_day))
        
        for i in range(leadtime):
            # 第 i+1 天的数据
            future_day = day_index + i + 1
            
            # 次日的rts (退货量)，今天看第二天
            next_day_rts = rts_qty_list[i]
            
            # 次日到货量，即今日看第二天
            next_day_arrived = transit_stock[i]
            
            # 次日期初可用库存 = 今日期末 + 次日到货
            next_day_begin_stock = rolling_end_stock + next_day_arrived
            
            # 当天预测销量 (使用 predicts[future_day][0] 作为当天预测)
            # if future_day < total_day:
            #     sell_qty_i = predicts_ptr[future_day][0]  # 预测销量
            # else:
            #     sell_qty_i = 0
            
            next_day_sell_qty = actuals_by_next_day[i]
            # 当天期末 = 期初 - 销量 (不能为负)
            rolling_end_stock = max(0, next_day_begin_stock - next_day_sell_qty) - next_day_rts
        
        # next_state(也就是下一天),leadtime 天的期初库存,无在途，所以不加transit_stock
        next_state_leadtime_begin_stock = rolling_end_stock
        lt_sell_qty = actuals_by_next_day[leadtime]

        rts_qty_list = rts_qty_list
        
        # 调试特定SKU的训练信息
        # if sku_id == "175768176161-14461":
        #     if not hasattr(self, '_train_debug_reset'):
        #         self._train_debug_reset = True
        #         debug_mode = 'w'
        #     else:
        #         debug_mode = 'a'
        #     with open('/Users/weibin/Desktop/weibin_projects/replenishment_conti/log_info/train_sku_debug.log', debug_mode) as f:
        #         f.write(f"=== day_index={day_index}, leadtime={leadtime} ===\n")
        #         f.write(f"  end_of_stock={end_of_stock}, begin_stock={sku.begin_stock}\n")
        #         f.write(f"  bind_stock={sku.bind_stock}, rts_qty={sku.rts_qty}\n")
        #         f.write(f"  today_arrived={sku.today_arrived}, abo_qty={sku.abo_qty}\n")
        #         f.write(f"  transit_stock={transit_stock[:5]}\n")
        #         f.write(f"  rts_qty_list={rts_qty_list[:6]}\n")
        #         f.write(f"  rolling_end_stock={rolling_end_stock}\n")
        #         f.write(f"  next_state_leadtime_begin_stock={next_state_leadtime_begin_stock}\n")
        #         f.write(f"  next_day_rts={next_day_rts}\n")
        #         f.write(f"  estimate_rts_qty={sku.estimate_rts_qty}\n")
        #         f.write(f"  lead_time_bind={sku.lead_time_bind}\n")
        #         f.write(f"  estimate_end_stock={sku.estimate_end_stock}\n\n")

        sell_qty_lt = self.datasets.actuals_map[sku_id][day_index+1]

        if self.state_label == "with_rts_split":
            next_state = [
                end_of_stock + next_day_rts,
                next_day_rts,
                transit_stock[0],
                transit_stock[1],
                transit_stock[2],
                transit_stock[3],
                transit_stock[4],
            ] + self.static_state_map[sku_id][day_index]
        elif self.state_label == "with_rts_combine":
            start_stock = end_of_stock # 已经减去rts
            # next_state = [
            #     start_stock,
            #     transit_stock[0],
            #     transit_stock[1],
            #     transit_stock[2],
            #     transit_stock[3],
            #     transit_stock[4],
            # ] 
            # + actuals_by_next_day  # 6
            # + rts_qty_list  # 5
            # + [next_state_leadtime_begin_stock, max(0,lt_sell_qty-next_state_leadtime_begin_stock)] # 2 聚合值
            # + self.static_state_map[sku_id][day_index] # lt和统计特征
            
            
            
            next_state = self.static_state_map[sku_id][day_index+1] + [next_state_leadtime_begin_stock,lt_sell_qty] # 增加rts_qty_list，长度固定为6
        
        # 之前的逻辑static_state也会错位一天
        
        elif self.state_label == "with_stock_level":
            start_stock = max(0, end_of_stock - next_day_rts)
            stock_level = self.cal_stock_level(end_of_stock, sku_id, day_index)
            next_state = [
                start_stock,
                transit_stock[0],
                transit_stock[1],
                transit_stock[2],
                transit_stock[3],
                transit_stock[4],
                stock_level,  # 新增库存销量水平
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
    def cal_reward(self, reward_type, sku, sku_id, overnight_qty, leadtime_sale_qty):
        """
        返回: (total_reward, reward_components_dict)
        """
        if reward_type == "with_safe_stock":
            return self.cal_reward_safety_stock(sku, sku_id, overnight_qty, leadtime_sale_qty)
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

    def cal_reward_safety_stock(self, sku, sku_id, overnight_qty, leadtime_sale_qty):
        bind_reward = self.lt_bind_weight * sku.lead_time_bind + self.until_rts_bind_weight * (
            sku.abo_qty - sku.estimate_rts_qty
        )

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
        if self.datasets.order_ratio_7d_map[sku_id][sku.day_index] >= self.head_sku_standard:  # 头部品
            stock_gap = max(
                (
                    self.datasets.avg_item_qty_7d_map[sku_id][sku.day_index] * self.safe_stock_standard
                    - sku.estimate_end_stock
                ),
                0,
            )
            safe_stock_penalty = stock_gap * self.safe_stock_weight
            stockout_penalty = max(0, (leadtime_sale_qty - sku.lead_time_bind) * self.stockout_weight)
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
    def cal_stock_level(self, stock, sku_id, day_idx):
        # 使用历史七天销量均值
        avg_qty = (
            self.datasets.avg_item_qty_7d_map[sku_id][day_idx]
            if self.datasets.avg_item_qty_7d_map[sku_id][day_idx]
            else 0.1
        )
        stock_level = stock / avg_qty
        # 使用当天预测值
        # temp_day_qty = self.predicts_map[sku][day_idx][0] if self.predicts_map[sku][day_idx][0] else 0.1
        # stock_level = stock / temp_day_qty
        # if stock_level > 5:
        #     return 5
        return stock_level
