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
from models.state import ReplenishState
from utils.datasets import Datasets
from task.base import TaskConfig
from typing import cast
import numpy as np


class ReplenishEnv(gym.Env):
    end_stock: dict[int, int]
    datasets: Datasets
    state: dict
    day_idx: int

    def __init__(
        self,
        data_path: str = "",
        cfg: TaskConfig = TaskConfig(),
    ) -> None:
        super().__init__()
        # 疑似无用的入参
        # rts_day: int = 14,
        # rts_weight: float = 0.9,
        # coverage_weight: float = 0.1,
        # overnight_weight: float = 0,
        # rank: int = 0,
        # world_size: int = 1
        self.init_args = locals().copy()
        for k in ["self", "cfg", "__class__"]:
            self.init_args.pop(k)
        # self.init_args = {k:v for k,v in self.init_args.items if k not in {"self", "cfg","__class__"}}
        cfg.update(self.init_args, priority="low")
        # data_path = cfg.data_path
        self.datasets = Datasets(data_path, cfg.state_static_columns, cfg.rank, cfg.world_size)
        self.static_state_map = self.datasets.get_static_state_map()
        # self.rts_day = cfg.get("rts_day") or rts_day
        self.rts_days = cfg.rts_days
        self.rts_weight = cfg.rts_weight

        self.coverage_weight = cfg.coverage_weight
        # self.lt_bind_weight = cfg.lt_bind_weight
        # self.until_rts_bind_weight = cfg.until_rts_bind_weight

        self.stockout_weight = cfg.stockout_weight
        self.overnight_weight = cfg.overnight_weight
        self.safe_stock_weight = cfg.safe_stock_weight
        self.safe_stock_standard = cfg.safe_stock_standard
        self.head_sku_standard = cfg.head_sku_standard
        self.reward_type = cfg.reward_type
        self.leadtime_map = self.datasets.leadtime_map
        
        # # 先这么写方便跑仿真
        # if hasattr(cfg, "reward_type"):
        #     self.reward_type = cfg.reward_type
        # else:
        #     self.reward_type = "default"
        self.end_stock = {}
        self.day_idx = 0

        self.sku_ids = self.datasets.sku_ids()
        self.end_date_map = self.datasets.get_end_date_map()
        self.skus = rolling.init_skus(self.rts_days, self.sku_ids, self.datasets)
        roller = Roller(self.rts_days, os.getenv("DEBUG", "False").lower() == "true")
        if roller.debug:
            roller.save_cur_row("debug.csv", header=True)

    def __getstate__(self):
        state = self.__dict__.copy()
        if "pool" in state:
            del state["pool"]
        return state

    

    def cal_reward(self,reward_type,sku,sku_id, overnight_qty,leadtime_sale_qty):
        # print(f"over_night_len:{len(overnight_qty)}👍") 
        # print(overnight_qty)
        # print("RSS before reward:", psutil.Process(os.getpid()).memory_info().rss // 1024, "KB")
        if reward_type=="with_safe_stock":
            reward=self.cal_reward_safety_stock(sku,sku_id, overnight_qty,leadtime_sale_qty)
        else:
            reward=self.cal_reward_default(sku, overnight_qty)
        
        # gc.collect()
        # print("RSS after reward :", psutil.Process(os.getpid()).memory_info().rss // 1024, "KB")
        return reward

    def cal_reward_by_dq(self, sku,sku_id, overnight_qty,leadtime_sale_qty):
        ###备注：参数设置临时写死，后续测试有效果再抽象出来
        
        if self.datasets.order_ratio_14d_map[sku_id][sku.day_index] >= 0.7:
            bind_reward = 0.4 * sku.lead_time_bind
            overnight_penalty = self.overnight_weight * sum(overnight_qty[1:])
        elif self.datasets.order_ratio_14d_map[sku_id][sku.day_index] >= 0.5:
            bind_reward = 0.3 * sku.lead_time_bind
            overnight_penalty = self.overnight_weight * sum(overnight_qty[1:])
        else:
            bind_reward =0.15 * sku.lead_time_bind
            overnight_penalty = self.overnight_weight * sum(overnight_qty)
        res_penalty = self.rts_weight * sku.estimate_rts_qty  # 到货后未来14天仍未消耗的 
        return bind_reward  - overnight_penalty - res_penalty
 
    def cal_reward_default(self, sku, overnight_qty):
        bind_reward = self.coverage_weight * sku.lead_time_bind
        overnight_penalty = self.overnight_weight * sum(overnight_qty)
        res_penalty = self.rts_weight * sku.estimate_rts_qty  # 到货后未来14天仍未消耗的 
        return bind_reward - overnight_penalty - res_penalty
    
    def cal_reward_safety_stock(self, sku,sku_id, overnight_qty,leadtime_sale_qty):
        bind_reward = self.coverage_weight * sku.lead_time_bind # coverage_weight现在改为了lt_bind_weight
        # bind_reward = self.lt_bind_weight * sku.lead_time_bind + self.until_rts_bind_weight * (sku.abo_qty - sku.estimate_rts_qty)
        # abo_bind_qty = max(sku.abo_qty - sku.estimate_end_stock, 0)
        # bind_reward = self.lt_bind_weight * abo_bind_qty
        overnight_penalty = self.overnight_weight * sum(overnight_qty)
        res_penalty = self.rts_weight * sku.estimate_rts_qty  # 到货后未来14天仍未消耗的

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

        return bind_reward - safe_stock_penalty - overnight_penalty - res_penalty - stockout_penalty

    ##实际需求

    def step(self, action):
        return None, 0, False, False, {}

    def step_one_sku(self, sku: SKU, action):
        nt = time.time()
        roller = Roller(self.rts_days, os.getenv("DEBUG", "False").lower() == "true")
        get_action: Callable[[int], float] = action.get("get_action")
        day_idx = action.get("day_idx") or 0
        if not get_action:
            return day_idx, sku, 0, 0, None, 0, True

        sku.set_state(lead_time=self.datasets.sku_lead_time(sku.id, day_idx))

        rolling_day = self.rts_days + sku.lead_time + 1
        predicts = self.range_predicts(day_idx, day_idx + rolling_day, sku.id)
        sales_list = self.range_sales(day_idx, day_idx + rolling_day, sku.id)

        # 滚动一天
        action = get_action(sku.id)
        roller.rolling_one_day(sku, predicts[0], sales_list[0], action, day_idx + sku.lead_time)
        done = (self.rts_days + sku.lead_time + day_idx + 2) >= self.end_date_map[sku.id]

        if action.get("evaluate", False):
            if roller.debug:
                roller.save_cur_row("debug.csv")
            return day_idx, sku, 0, 0, roller.overnight, 0, done

        tmp_sku = deepcopy(sku)
        # 开始尝试计算未来x-1天的
        lead_time_bind = roller.rolling(
            rolling_day - 1, tmp_sku, predicts, sales_list, get_action, day_idx + sku.lead_time
        )
        tmp_sku_rts = tmp_sku.rts_qty
        # 计算reward TODO
        ##reward = self.cal_reward(sku.replenish_qty, sku.rts_qty, roller.overnight)
        #reward = self.cal_reward(lead_time_bind, sku.replenish_qty, tmp_sku.rts_qty, roller.overnight)
        reward = self.cal_reward(self.reward_type,sku,sku_id,overnight_array, leadtime_sale_qty)
            
        del tmp_sku
        if roller.debug:
            roller.save_cur_row("debug.csv")

        return day_idx, sku, lead_time_bind, tmp_sku_rts, roller.overnight, reward, done

    def batch_step(self, action_map, evaluate: bool = False):

        result_map = {}  # {sku.id:result}
        state_map = {}

        rolling.rolling(self.skus, evaluate, action_map)

        bind, rts = 0, 0
        for i in range(len(self.skus)):
            sku = cast(SkuInputInfo, self.skus[i])
            total_day = self.datasets.end_date_map[sku.id_str]
            lead_time_list = np.ctypeslib.as_array(sku.lead_time, shape=(total_day,))

            orders_np = np.ctypeslib.as_array(sku.orders, shape=(sku.orders_size,))
            transit_stock = orders_np[sku.day_index + 1 : sku.day_index + lead_time_list[sku.day_index] + 2].tolist()

            overnight_size = sku.ending_stock_list_size  # 声明时已经保持一致
            overnight_array = np.ctypeslib.as_array(sku.overnight_list, shape=(overnight_size,))

            leadtime_sale_qty = sku.sales[sku.day_index+lead_time_list[sku.day_index]]  # leadtime天的销量

            sku_id = self.skus[i].id.decode("utf-8")  # 从字节串转回字符串
            if self.datasets.dq_map[sku_id][sku.day_index] >= self.head_sku_standard: # 头部品
                safe_stock_gap = max((self.datasets.avg_item_qty_7d_map[sku_id][sku.day_index] * self.safe_stock_standard - sku.estimate_end_stock),0)
            else:
                safe_stock_gap = 0

            state_map[sku_id] = {
                "abo_qty": sku.abo_qty,
                "estimate_rts_qty": sku.estimate_rts_qty,
                "estimate_overnight": overnight_array.copy(),
                "end_of_stock": sku.end_of_stock,
                "transit_stock": transit_stock.copy(),
                "estimate_bind_qty": sku.lead_time_bind,
                "next_day_rts": sku.rts_qty,
                "safe_stock_gap": safe_stock_gap
            }

            reward = self.cal_reward(self.reward_type,sku,sku_id,overnight_array, leadtime_sale_qty)
            # TODO @weibin
            # sku.estimate_end_stock leadtime当天期末库存
            # sku.sales[sku.day_index+lead_time_list[sku.day_index]] 销售量 (注意check越界)

            # reward = self.cal_reward(sku.lead_time_bind, sku.abo_qty, sku.estimate_rts_qty, overnight_array)
            done = (self.rts_days + lead_time_list[sku.day_index] + sku.day_index + 2) >= self.end_date_map[sku_id]
            result_map[sku_id] = {
                "reward": reward,
                "done": done,
            }

            bind += sku.bind_stock
            rts += sku.rts_qty

        return state_map, result_map, bind, rts

    def reset(self, *, seed=None, options=None) -> Tuple[dict[int, ReplenishState], dict]:
        # self.skus = {id: SKU(id) for id in self.sku_ids}
        self.skus = rolling.reset_skus(self.skus)
        self.day_idx = 0
        self.end_stock = {}

        observation = {id: ReplenishState(self.datasets.predict_leadtime_day[id][0]) for id in self.sku_ids}
        return observation, {}

    def range_predicts(self, day_idx_st, day_idx_ed, sku_id):
        return self.datasets.range_prdicts(day_idx_st, day_idx_ed, sku_id)

    def load_predicts(self, day_idx, sku_id):
        return self.datasets.get_predicts(day_idx, sku_id)

    def range_sales(self, day_idx_st, day_idx_ed, sku_id):
        return self.datasets.range_sales(day_idx_st, day_idx_ed, sku_id)

    def load_sales(self, day_idx, sku_id):
        return self.datasets.get_sales(day_idx, sku_id)

    def find_first_nonzero(self,data):
        try:
            return next(i for i, x in enumerate(data) if x != 0)
        except StopIteration:
            return None  # 没有找到非0值


def get_multiplier(sku_id: int) -> float:
    return 2
