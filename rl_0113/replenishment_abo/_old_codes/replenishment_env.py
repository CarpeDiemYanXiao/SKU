import gym
import gym.spaces
from gym import spaces
import numpy as np

###滚动功能
from atp_sim_sdk_py import rolling_env, replenish_scene, strategy, entity

###输入数据表结构: idx,date,y_true,pred_y_ls,pred_y,leadtime,
###
###state:
"""
假设：

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


class ReplenishmentEnv(gym.Env):
    def __init__(
        self, input_data, rts_days=14, rts_penalty_ratio=10, bind_reward_ratio=5, bind_reward_within_rts_ratio=2
    ):
        super(ReplenishmentEnv, self).__init__()

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
        state_dim = self.num_products * 2
        self.observation_space = spaces.Box(low=0, high=100, shape=(state_dim,), dtype=np.float32)
        self.input_data = input_data
        self.gen_data()



    def gen_data(self):
        """
        key:idx,多天按照时间顺序排序
        :return:
        """
        self.input_data = self.input_data.sort_values(by=["idx", "date"])
        self.sales_map = self.input_data.groupby("idx")["actual"].apply(list).to_dict()
        self.predict_leadtime_day = self.input_data.groupby("idx")["pred_y"].apply(list).to_dict()
        self.leadtime_map = self.input_data.groupby("idx")["leadtime"].apply(list).to_dict()
        self.sku_id_ls = self.input_data.groupby(["date", "idx"])["pred_y"].sum().unstack().columns.tolist()
        self.predicted_demand = self.input_data.groupby(["date", "idx"])["pred_y"].sum().unstack().fillna(0).values

    def _get_state(self):
        # 状态包括当前库存和当天的预测需求量
        return np.concatenate((self.stock, self.predicted_demand[self.current_day]))

    def reset(self):
        ###从第一天补货开始，初始库存均为0
        self.current_day = 0
        self.stock = np.zeros(self.num_products)
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
        res_dict["predicts"] = self.predicts_map[sku_id][self.current_day]
        res_dict["stock"] = self.stock[self.sku_id_ls.index(sku_id)]

        return res_dict

    def step(self, action):
        ###准备滚动的输入数据：每个品current_day的信息（leadtime，end_of_stock，multiplier，predicts，sales）

        ###滚动
        new_stock = []
        rts_qty = 0
        bind_qty = 0
        bind_qty_within_rts = 0

        st = strategy.StrategyB()
        scene = replenish_scene.CacheOrder(st)
        roller = rolling_env.RollingEnv(scene, rts_day=self.rts_days)

        roller.reset(sku)
        for sku_id in self.sku_id_ls:
            print(sku_id)
            sku_stat = self.gen_rolling_input(sku_id, action)

            sku = entity.SKU(10, 1, 2, 3, lead_time=sku_stat["leadtime"], yesterday_end_stock=sku_stat["stock"])
            multiplier = sku_stat["multiplier"]
            predicts = sku_stat["predicts"]  # must lead_time<len
            sales = sku_stat["sales"]

            roller.rolling(sku, multiplier, predicts, sales)

            sku.snapshot_result()

            new_stock.append(sku)
            rts_qty += roller_res.rts_qty
            bind_qty += roller_res.bind_qty
            if self.current_day <= self.rts_days:
                bind_qty_within_rts += roller_res.bind_qty

        roller.summary_result()
        ###更新end_of_stock
        self.stock = new_stock
        # 更新时间步
        self.current_day += 1  # 时间步进增加
        # 计算奖励
        reward = self.reward_function(rts_qty, bind_qty, bind_qty_within_rts)

        # 判断是否结束
        done = self.current_day >= self.end_date

        return self._get_state, reward, done, {}

    def render(self, mode="human"):
        pass


if __name__ == "__main__":

    env = ReplenishmentEnv()
    state = env.reset()
    print("Initial State:", state)

    for _ in range(4):  # 运行超过结束日期的步数
        print(_)
        action = 1
        next_state, reward, done, _ = env.step(action)
        print("Next State:", next_state, "Reward:", reward, "Done:", done)
        if done:
            break
