import pandas as pd
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from torch.distributions import Categorical
from envs.replenish_env import ReplenishEnv
from network.PolicyNetwork import PolicyNetwork
from network.ValueNetwork import ValueNetwork
from utils.datasets import Datasets
import os
import time
from datetime import datetime, timedelta
from utils.log import (
    logging_once,
    logging_setup,
    timeit,
    Timer,
    seatalk_alert,
    print_once,
)

"""
policynetwork
state [end_of_stock,forecast]
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


class GetMultiplier:
    def __init__(self, multiplier_ls, action):
        self.multiplier_ls = multiplier_ls
        self.action = action

    def __call__(self, day_idx):
        return self.multiplier_ls[self.action]
        ##return self.action


class ReplenishAgentCombine:
    def __init__(self, replenish_model, config):
        self.replenish_model = replenish_model
        self.task_name = config["task_name"]
        self.device = config["device"]

        ###输入的数据
        ###商品清单
        self.datasets = Datasets(config["data_path"])
        self.sku_id_ls = self.datasets.sku_ids()
        self.predict_leadtime_day = self.datasets.predict_leadtime_day
        self.leadtime_map = self.datasets.leadtime_map
        self.initial_stock_map = self.datasets.get_initial_stock_map()
        self.end_date_map = self.datasets.get_end_date_map()
        self.total_sales = self.datasets.total_sales
        self.sales_map = self.datasets.sales_map

        ####环境设置
        self.env = ReplenishEnv(
            config["data_path"],
            rts_day=config["rts_days"],
            pool_size=config["pool_size"],
            coverage_weight=config["coverage_weight"],
            rts_weight=config["rts_weight"],
            overnight_weight=config["overnight_weight"],
        )

        ###network 配置
        self.state_dim = config["state_dim"]
        self.multiplier_ls: list[float] = config["multiplier_ls"]
        self.action_dim = len(config["multiplier_ls"])
        self.policy = PolicyNetwork(self.state_dim, self.action_dim).to(self.device)

        os.makedirs(os.path.join("output", self.task_name), exist_ok=True)
        self.policy_model_filename = os.path.join("output", self.task_name, f"repl_policy_model.pth")

        ###模型训练参数
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config["lr"])
        self.gamma = config["gamma"]
        self.k_epochs = config["k_epochs"]  ###k_epochs
        self.eps_clip = config["eps_clip"]  ###eps_clip
        self.batch_size = config["batch_size"]  ###batch_size
        self.max_episodes = config["max_episodes"]

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
        print("initialize don")

    def get_update_action(self, sku_id, state):
        state, action, action_detach, logprob, state_value = self.select_action(state)
        self.states_map[sku_id].append(state)
        self.actions_map[sku_id].append(action_detach)
        self.logprobs_map[sku_id].append(logprob)
        self.state_values_map[sku_id].append(state_value)
        return action

    def select_action(self, state):
        """选择动作"""
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        probs, state_value = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        logprob = m.log_prob(action).detach()
        return state, action.item(), action.detach(), logprob, state_value

    def select_action_deterministic(self, state):
        """确定性地选择动作，每次选择最优的动作"""
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            probs, state_value = self.policy(state)  # 解包返回值，只使用概率
            action = torch.argmax(probs).item()
        return action

    def select_multiplier_deterministic(self, state):
        """调用模型选择最优的multiplier"""
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            probs, state_value = self.policy(state)  # 解包返回值，只使用概率
            action = torch.argmax(probs).item()
        return self.multiplier_ls[int(action)]

    def plot_rewards(self):
        """绘制奖励趋势图并保存"""
        # 设置中文字体
        # plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]  # Mac系统
        # plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
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
        filename = f"output/{self.task_name}/reward_trend.jpg"
        plt.savefig(filename)
        plt.close()

    ###TODO
    def gen_network_state(self, state_dict, sku, day_idx):
        """
        生成输入network的state：[end_of_stock,next_day_arrive_order,next_day_rts,forecast,leadtime]
        :return:
        """
        ##天数
        ###day_idx=state_dict["day_idx"]
        ##new_state=[day_idx,state_dict["end_of_stock"],state_dict["arrival_abo"],state["arrival_qty"],self.predict_leadtime_day[sku][day_idx]]

        transit_stock = state_dict["transit_stock"] + [0] * (5 - len(state_dict["transit_stock"]))
        new_state = [
            state_dict["end_of_stock"],
            ##state_dict["transit_stock"],
            state_dict["next_day_rts"],
            self.predict_leadtime_day[sku][day_idx],
            self.leadtime_map[sku][day_idx],
            transit_stock[0],
            transit_stock[1],
            transit_stock[2],
            transit_stock[3],
            transit_stock[4],
        ]
        return new_state

    def reset_network_state(self, sku):
        """
        生成输入network的state
        :return:期末库存，预测值
        """
        ##return [0, 0, self.predict_leadtime_day[sku][0], self.leadtime_map[sku][0]]
        return [
            self.initial_stock_map[sku],
            0,
            self.predict_leadtime_day[sku][0],
            self.leadtime_map[sku][0],
            0,
            0,
            0,
            0,
            0,
        ]

    def reset_day_idx(self):
        return 0

    def reset_state_dict(self):
        ###补货时间，期末库存，预测值
        return {"end_of_stock": 0, "arrival_abo": 0, "arrival_qty": 0, "day_idx": 0}

    def gen_update_data(self):

        states_ls = []
        actions_ls = []
        logprobs_ls = []
        advantages_ls = []
        returns_ls = []

        for sku in self.sku_id_ls:
            ##print(len(self.states_map[sku]))
            states = torch.stack(self.states_map[sku]).to(self.device)
            actions = torch.tensor(self.actions_map[sku], dtype=torch.int64).to(self.device)
            logprobs = torch.stack(self.logprobs_map[sku]).to(self.device)
            state_values = torch.cat(self.state_values_map[sku]).squeeze().to(self.device)
            rewards = torch.tensor(self.rewards_map[sku], dtype=torch.float32).to(self.device)
            dones = torch.tensor(self.dones_map[sku], dtype=torch.float32).to(self.device)

            returns = []
            discounted_reward = 0
            for reward, done in zip(reversed(rewards), reversed(dones)):
                if done:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                returns.insert(0, discounted_reward)
            returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
            advantages = returns - state_values.detach()

            states_ls.append(states)
            actions_ls.append(actions)
            logprobs_ls.append(logprobs)
            returns_ls.append(returns)
            advantages_ls.append(advantages)

        states_all = torch.cat(states_ls)
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

        return (states_batch, actions, actions.detach(), logprobs.detach(), state_values)

    def update(self):
        """更新策略网络"""
        ###生成数据
        states, actions, logprobs, returns, advantages = self.gen_update_data()
        # print(f"state len ={len(states)}")
        ### PPO更新
        # 将所有经验数据合并为一个数据集
        dataset = torch.utils.data.TensorDataset(states, actions, logprobs, advantages, returns)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # PPO更新
        for epoch in range(self.k_epochs):
            for batch in dataloader:
                (
                    batch_states,
                    batch_actions,
                    batch_logprobs,
                    batch_advantages,
                    batch_returns,
                ) = batch

                batch_probs, batch_state_values_new = self.policy(batch_states)  # 获取动作的概率分布

                m = Categorical(batch_probs)
                batch_new_logprobs = m.log_prob(batch_actions)  ## # 计算采样动作的对数概率
                entropy = m.entropy().mean()  ##计算分布的熵
                # Finding the ratio
                ratios = torch.exp(batch_new_logprobs - batch_logprobs)
                # Finding Surrogate Loss
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                # final loss of clipped objective PPO
                loss = (
                    -torch.min(surr1, surr2).mean()
                    + 0.5 * nn.MSELoss()(batch_state_values_new.squeeze(), batch_returns)
                    - 0.01 * entropy
                )
                # print(f"loss={loss}")
                ###更新网络
                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

            # 每个 epoch 保存一次模型 TODO:优化保存逻辑
            self.save_model(self.policy_model_filename)

        # 清空缓存
        self.states_map = {sku: [] for sku in self.sku_id_ls}
        self.actions_map = {sku: [] for sku in self.sku_id_ls}
        self.logprobs_map = {sku: [] for sku in self.sku_id_ls}
        self.state_values_map = {sku: [] for sku in self.sku_id_ls}
        self.rewards_map = {sku: [] for sku in self.sku_id_ls}
        self.dones_map = {sku: [] for sku in self.sku_id_ls}
        self.episode_rewards_map = {sku: [] for sku in self.sku_id_ls}

    def train(self):
        """训练智能体"""
        for episode in range(self.max_episodes):
            start_time = time.time()
            # logging_once(f"episode={episode+1}")
            print("###############################")
            print(f"episode={episode + 1}")
            # logging_once(f"##########env initial#############")

            ####所有商品进行初始化
            ###商品network-state初始化，日期初始化
            state_map = {sku: self.reset_network_state(sku) for sku in self.sku_id_ls}  ##idx:state
            day_idx_map = {sku: self.reset_day_idx() for sku in self.sku_id_ls}
            done_map = {sku: False for sku in self.sku_id_ls}

            self.env.reset()  ###需要对所有sku进行初始化
            done_all = False
            total_reward = 0
            rts_qty = 0
            while not done_all:
                action_map = {}
                for sku in self.sku_id_ls:
                    sku_action_map = {}
                    state, action, action_detach, logprob, state_value = self.select_action(state_map[sku])
                    sku_action_map["get_multiplier"] = GetMultiplier(self.multiplier_ls, action)
                    sku_action_map["day_idx"] = day_idx_map[sku]
                    action_map[sku] = sku_action_map

                    if not done_map[sku]:
                        self.states_map[sku].append(state)
                        self.actions_map[sku].append(action_detach)
                        self.logprobs_map[sku].append(logprob)
                        self.state_values_map[sku].append(state_value)
                # print("begin rolling")
                st = time.time()

                rolling_state_map, result_map, _, _ = self.env.batch_step(action_map)
                # print("end rolling")
                # print("rolling time: ", time.time() - st)  # 计算运行时间

                ###state_dict
                for sku in self.sku_id_ls:
                    if not done_map[sku]:
                        reward = result_map[sku]["reward"]
                        done = result_map[sku]["done"]
                        ###生成policy-network-state
                        ###更新day_idx
                        day_idx_map[sku] += 1
                        new_network_state = self.gen_network_state(rolling_state_map[sku], sku, day_idx_map[sku])
                        ###更新sku的state
                        state_map[sku] = new_network_state
                        done_map[sku] = done
                        self.rewards_map[sku].append(reward)
                        self.dones_map[sku].append(done)
                        total_reward += reward
                        rts_qty += rolling_state_map[sku]["estimate_rts_qty"]

                if set(done_map.values()) == {True}:  ###所有商品都是done才能结束
                    done_all = True
            ##print("gen obs time: ", time.time() - start_time)  # 计算运行时间
            # print("begin update")
            self.update()
            # print("end update")
            self.episode_rewards.append(total_reward)

            if (episode + 1) % 50 == 0:
                # 保存为 JSON 文件
                with open("reward_data.json", "w") as file:
                    json.dump(self.episode_rewards, file)

                self.plot_rewards()

                # print(
                #             f"episode {episode + 1}\ttotal reward: {total_reward}\trts_qty: {rts_qty}"
                #         )

            end_time = time.time()  # 获取结束时间
            elapsed_time = end_time - start_time  # 计算运行时间

            print(
                f"episode {episode + 1}-\ttotal reward: {total_reward:.2f}\trts_qty: {rts_qty}\truntime:  {elapsed_time:.2f} s"
            )

    def cal_date(self, start_date, delta_days):
        end_date = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=delta_days)
        return end_date.strftime("%Y-%m-%d")

    def gen_multipliers(self, actions_map, start_date, task_name):

        key_ls = []
        res_data_ls = []

        for key, value in actions_map.items():
            key_ls.append(key)
            res_data_ls.append(value)

        df = pd.DataFrame(data={"idx": key_ls, "multilpier": res_data_ls})

        df["multiplier_index"] = df["multilpier"].apply(lambda x: list(range(len(x))))

        exploded_df = df.explode(["multilpier", "multiplier_index"])
        exploded_df["ds"] = exploded_df.apply(lambda s: self.cal_date(start_date, s["multiplier_index"]), axis=1)
        exploded_df["dt_version"] = task_name
        exploded_df = exploded_df[["idx", "ds", "multilpier", "dt_version"]]
        exploded_df.columns = ["idx", "ds", "multilpier", "dt_version"]
        return exploded_df

    def gen_simu_result(self, model_path):
        self.load_model(model_path)
        print("print load successfully")
        actions_all = []
        states_all = []
        ####所有商品进行初始化
        ###商品network-state初始化，日期初始化
        actions_map = {sku: [] for sku in self.sku_id_ls}
        state_map = {sku: self.reset_network_state(sku) for sku in self.sku_id_ls}  ##idx:state
        states_map = {sku: [] for sku in self.sku_id_ls}
        day_idx_map = {sku: self.reset_day_idx() for sku in self.sku_id_ls}
        done_map = {sku: False for sku in self.sku_id_ls}
        self.env.reset()  ###需要对所有sku进行初始化
        done_all = False
        total_reward = 0
        ###
        simu_rts_qty = 0
        simu_bind_qty = 0

        while not done_all:
            action_map = {}
            for sku in self.sku_id_ls:
                sku_action_map = {}
                states_map[sku].append(state_map[sku])
                action = self.select_action_deterministic(state_map[sku])
                actions_map[sku].append(self.multiplier_ls[int(action)])
                sku_action_map["get_multiplier"] = GetMultiplier(self.multiplier_ls, action)
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
                ###更新sku的state
                state_map[sku] = new_network_state
                done_map[sku] = day_idx_map[sku] >= self.end_date_map[sku]
                ##total_reward += reward

            if set(done_map.values()) == {True}:  ###所有商品都是done才能结束
                done_all = True
        ##self.start_date="2024-12-16"
        print(
            f"total_sales ={self.total_sales} ,simu_rts_qty= {simu_rts_qty}, simu_bind_qty={simu_bind_qty},acc_rate={simu_bind_qty/self.total_sales},rts_rate={simu_rts_qty/simu_bind_qty}"
        )
        multiplier_df = self.gen_multipliers(actions_map, "2024-12-16", self.task_name)

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
                "actions": actions_all,
                "states": states_all,
            }
        )
        res_df["rts_rate"] = res_df["rts_qty"] / res_df["bind_qty"]
        res_df["acc_rate"] = res_df["bind_qty"] / res_df["actual_qty"]
        res_data = os.path.join("output", self.task_name, f"simu_res_data.csv")
        multiplier_res_data = os.path.join("output", self.task_name, f"simu_multiplier_data.csv")
        res_df.to_csv(res_data, index=False)
        multiplier_df.to_csv(multiplier_res_data, index=False)

    def evaluate(self, model_path, num_episodes=10):  # 增加参数来控制评估次数

        self.load_model(model_path)
        print("print load successfully")

        rewards = []  # 存储每次评估的奖励
        actions_all = []
        states_all = []
        for episode in range(num_episodes):

            ####所有商品进行初始化
            ###商品network-state初始化，日期初始化
            rolling_cnt = 0
            actions_map = {sku: [] for sku in self.sku_id_ls}
            state_map = {sku: self.reset_network_state(sku) for sku in self.sku_id_ls}  ##idx:state
            states_map = {sku: [] for sku in self.sku_id_ls}
            day_idx_map = {sku: self.reset_day_idx() for sku in self.sku_id_ls}
            done_map = {sku: False for sku in self.sku_id_ls}
            self.env.reset()  ###需要对所有sku进行初始化
            done_all = False
            total_reward = 0

            all_bind, all_rts = 0, 0
            while not done_all:
                rolling_cnt += 1
                print(f"rolling_cnt={rolling_cnt}")
                action_map = {}
                for sku in self.sku_id_ls:
                    sku_action_map = {}
                    states_map[sku].append(state_map[sku])
                    action = self.select_action_deterministic(state_map[sku])
                    actions_map[sku].append(self.multiplier_ls[int(action)])
                    sku_action_map["get_multiplier"] = GetMultiplier(self.multiplier_ls, action)
                    sku_action_map["evaluate"] = True
                    sku_action_map["day_idx"] = day_idx_map[sku]
                    action_map[sku] = sku_action_map

                rolling_state_map, result_map, total_bind, total_rts = self.env.batch_step(action_map)
                all_bind += total_bind
                all_rts += total_rts

                ###state_dict
                for sku in self.sku_id_ls:
                    if not done_map[sku]:
                        reward = result_map[sku]["reward"]
                        done = result_map[sku]["done"]
                        ###生成policy-network-state
                        ###更新day_idx
                        day_idx_map[sku] += 1
                        new_network_state = self.gen_network_state(rolling_state_map[sku], sku, day_idx_map[sku])
                        ###更新sku的state
                        state_map[sku] = new_network_state
                        done_map[sku] = done
                        total_reward += reward
                if set(done_map.values()) == {True}:  ###所有商品都是done才能结束
                    done_all = True
            # print(f"rolling reward ={total_reward}")
            rewards.append(total_reward)
            actions_all.append(actions_map)
            states_all.append(states_map)
            print(f"No.{episode + 1} evaluate done，total_reward: {total_reward}")

        rewards = np.array(rewards)  # 将rewards转换为numpy数组
        average_reward = np.mean(rewards)
        std_dev = np.std(rewards)

        print(f"\nreward_statistics:")
        print(f"mean_reward: {average_reward:.2f}")
        print(f"std_reward: {std_dev:.2f}")
        print(f"max_reward: {max(rewards)}")
        print(f"min_reward: {min(rewards)}")
        res_df = pd.DataFrame(data={"rewards": rewards, "actions": actions_all, "states": states_all})
        res_data = os.path.join("output", self.task_name, f"res_data.csv")
        res_df.to_csv(res_data, index=False)

    def save_model(self, path):
        """保存模型和网络配置"""
        model_info = {
            "state_dict": self.policy.state_dict(),
            "network_type": "",
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }
        torch.save(model_info, path)

    def save_value_model(self, path):
        """保存模型和网络配置"""
        model_info = {
            "state_dict": self.value.state_dict(),
            "network_type": "",
            "state_dim": self.state_dim,
        }
        torch.save(model_info, path)

    def load_model(self, path):
        """加载模型和网络配置"""
        model_info = torch.load(path)
        # 加载模型参数
        self.policy.load_state_dict(model_info["state_dict"])
        self.policy.to(self.device)


if __name__ == "__main__":

    with open("replenish_model/config/base_config.json", "r") as f:
        cfg = json.load(f)
    ra = ReplenishAgentCombine(None, cfg)
    ra.train()
    ra.save_model("rep_model.pth")
    ra.plot_rewards()
    print("end")
