import gym
import gym.spaces
from gym import spaces
import numpy as np
import pandas as pd

###滚动功能
from atp_sim_sdk_py import entity
from atp_sim_sdk_py.rolling_env import RollingEnv

###输入数据表结构: idx,date,y_true,pred_y_ls,pred_y,leadtime,
###
###state:
"""
假设：商品之间相互独立

###输入dataframe
columns：idx，leadtime,date，actual,pred_y,pred_ls，

###state:
end_of_stock: 期末库存
transit_stock:在途库存##
forecast：leadtime当天的值
###
action:  multiplier_ls = [0.5, 1.0, 1.5, 2.0,2.5,3]

###结束状态： 所有商品都完成仿真，

##reward_function

###coverage


"""


###


class ReplenishIndependentEnv(gym.Env):
    def __init__(
        self,
        input_data: pd.DataFrame,
        roller: RollingEnv,
        rts_days=14,
        rts_penalty_ratio=10,
        bind_reward_ratio=5,
        bind_reward_within_rts_ratio=2,
    ):
        self.roller = roller
        self.rts_penalty_ratio = rts_penalty_ratio
        self.bind_reward_ratio = bind_reward_ratio
        self.bind_reward_within_rts_ratio = bind_reward_within_rts_ratio
        self.rts_days = rts_days
        ####初始化：库存，开始日期
        self.current_day = 0

        self.num_products = input_data["idx"].nunique()
        self.end_date = input_data["date"].nunique()

        ####初始化库存
        self.current_stock = np.zeros(self.num_products)

        ###multiplier系数
        self.multiplier_ls = [0.5, 1.0, 1.5, 2.0]
        action_dim = len(self.multiplier_ls)
        self.action_space = gym.spaces.Discrete(action_dim)
        self.state_dim = 2  ##state_dim
        self.observation_space = spaces.Box(low=0, high=100, shape=(self.state_dim,), dtype=np.float32)
        self.input_data = input_data
        self.gen_data()

    def gen_data(self):
        """
        key:idx,多天按照时间顺序排序
        :return:
        """

        self.input_data = self.input_data.sort_values(by=["idx", "date"])
        self.input_data["end_date"] = self.input_data["date"] - 1
        self.sales_map = self.input_data.groupby("idx")["actual"].apply(list).to_dict()
        self.predict_leadtime_day = self.input_data.groupby("idx")["pred_y"].apply(list).to_dict()
        self.leadtime_map = self.input_data.groupby("idx")["leadtime"].apply(list).to_dict()

        self.leadtime_map = self.input_data.groupby("idx")["leadtime"].apply(list).to_dict()

        self.sku_id_ls = self.input_data.groupby(["date", "idx"])["pred_y"].sum().unstack().columns.tolist()
        self.predicted_demand = self.input_data.groupby(["date", "idx"])["pred_y"].sum().unstack().fillna(0).values
        self.predicts_map = self.input_data.groupby("idx")["predicted_demand"].apply(list).to_dict()
        self.end_date_map = self.input_data.groupby("idx")["end_date"].max().to_dict()
        self.current_day_map = dict(zip(self.sku_id_ls, [0 for i in range(len(self.sku_id_ls))]))

    def _get_state(self, sku_id):
        # 状态包括当前库存和当天的预测需求量
        ###
        return np.concatenate((self.stock, self.predict_leadtime_day[sku_id][self.current_day_map[sku_id]]))

    def reset(self, sku_id):
        ###从第一天补货开始，初始库存均为0
        self.current_day_map[sku_id] = 0
        self.stock = 0
        self.roller.reset()
        _ = [sku.reset() for sku in self.skus]
        return self._get_state()

    def reward_function(self, rts_qty, bind_qty, bind_qty_within_rts):
        rts_penalty = self.rts_penalty_ratio * rts_qty
        bind_reward = self.bind_reward_ratio * (bind_qty - bind_qty_within_rts)
        bind_reward_within_rts = self.bind_reward_within_rts_ratio * bind_qty_within_rts
        # 综合奖励：
        reward = bind_reward + bind_reward_within_rts - rts_penalty
        return reward

    def gen_rolling_input(self, sku_id, action):
        """
                返回用于滚动的输入信息
                sku_stat={"leadtime":2,
        "day":0,
        "multiplier":2,
        "predicts":[1,2,3,2,1,1],
        "sales":3,
        "stock":1,
        "rts_days":14
        }
                :param action:
                :return:
        """
        res_dict = {}
        res_dict["multiplier"] = self.multiplier_ls[action]
        res_dict["rts_days"] = self.rts_days
        res_dict["current_day"] = self.current_day
        res_dict["leadtime"] = self.leadtime_map[sku_id][self.current_day]
        res_dict["sales"] = self.sales_map[sku_id][self.current_day]
        # res_dict["predicts"] = self.predicted_demand[sku_id][self.current_day]
        res_dict["predicts"] = eval(self.predicts_map[sku_id][self.current_day])

        # res_dict["stock"] = self.stock[self.sku_id_ls.index(sku_id)]

        return res_dict

    def load_data(self, csv_path=""):
        skus = [
            entity.SKU(idx, 1, 2, 3, lead_time=lead_times[0], yesterday_end_stock=0)
            for idx, lead_times in self.leadtime_map.items()
        ]
        return skus

    def sku_step(self, action, sku):
        ###准备滚动的输入数据：每个品current_day的信息（leadtime，end_of_stock，multiplier，predicts，sales）

        # 所有sku滚动完毕后的统计值
        # abo_qty, bind_qty, rts_qty = roller.summary_result()
        sku_stat = self.gen_rolling_input(sku.model_id, action)
        multiplier = sku_stat["multiplier"]
        predicts = sku_stat["predicts"]  # must lead_time<len
        sales = sku_stat["sales"]
        self.roller.rolling(sku, multiplier, predicts, sales)
        # sku当前快照
        day_index, begin_stock, predict_arrive, bind_stock, end_stock, replenish_qty = sku.snapshot_result()
        print(
            f"idx={sku.soc_id},day={day_index},begin_stock={begin_stock},bind_stock={bind_stock},end_stock={end_stock},replenish_qty={replenish_qty}"
        )

        bind_qty = sku.bind_stock
        rts_qty = sku.rts_qty
        if self.current_day_map[sku] <= self.rts_days:
            bind_qty_within_rts = bind_stock
        ###更新end_of_stock
        self.stock = end_stock
        # 更新时间步
        self.current_day_map[sku.model_id] += 1  # 时间步进增加
        # 计算奖励
        reward = self.reward_function(rts_qty, bind_qty, bind_qty_within_rts)

        # 判断是否结束
        done = self.current_day_map >= self.end_date_map

        return self._get_state(), reward, done, {}

    def step(self, action):
        ###准备滚动的输入数据：每个品current_day的信息（leadtime，end_of_stock，multiplier，predicts，sales）

        ###滚动
        new_stock = []
        rts_qty = 0
        bind_qty = 0
        bind_qty_within_rts = 0
        end_stocks = []

        for sku in self.skus:
            # print(f"idx={sku.soc_id}")

            sku_stat = self.gen_rolling_input(sku.model_id, action)
            multiplier = sku_stat["multiplier"]
            predicts = sku_stat["predicts"]  # must lead_time<len
            sales = sku_stat["sales"]

            self.roller.rolling(sku, multiplier, predicts, sales)

            # sku当前快照
            day_index, begin_stock, predict_arrive, bind_stock, end_stock, replenish_qty = sku.snapshot_result()
            print(
                f"idx={sku.soc_id},day={day_index},begin_stock={begin_stock},bind_stock={bind_stock},end_stock={end_stock},replenish_qty={replenish_qty}"
            )
            new_stock.append(
                (
                    day_index,
                    begin_stock,
                    bind_stock,
                    end_stock,
                    replenish_qty,
                    predict_arrive,
                )
            )
            end_stocks.append(end_stock)
            bind_qty += sku.bind_stock
            rts_qty += sku.rts_qty
            if self.current_day <= self.rts_days:
                bind_qty_within_rts += bind_stock
        # self.roller._snapshot_table()

        # 所有sku滚动完毕后的统计值
        # abo_qty, bind_qty, rts_qty = roller.summary_result()

        ###更新end_of_stock
        self.stock = end_stocks
        # 更新时间步
        self.current_day += 1  # 时间步进增加
        # 计算奖励
        reward = self.reward_function(rts_qty, bind_qty, bind_qty_within_rts)

        # 判断是否结束
        done = self.current_day >= self.end_date - 1

        return self._get_state(), reward, done, {}

    def render(self, mode="human"):
        pass


# if __name__ == "__main__":

#     # 准备sku
#     skus: list[entity.SKU] = agent.load_data()
#     roller.reset()
#     _ = [sku.reset() for sku in skus]

#     for _ in range(15):  # 运行超过结束日期的步数
#         action = 1
#         next_state, reward, done, _ = agent.step(roller, skus, action)
#         print("Next State:", next_state, "Reward:", reward, "Done:", done)
#         if done:
#             break
