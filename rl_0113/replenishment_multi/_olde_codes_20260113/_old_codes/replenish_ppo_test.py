import gym
import torch
from PPOAgent import PPOAgent
from gym.wrappers import RecordVideo
import argparse
import pandas as pd

from atp_sim_sdk_py import strategy, replenish_scene, rolling_env, replenish_env
from replenish_agent import ReplenishAgent
from replenishment_env import ReplenishmentEnv


def train_with_no_ui():
    df = pd.read_csv('./data/test_data.csv')
    print(df.head())


    st = strategy.StrategyB()
    scene = replenish_scene.CacheOrder(st)
    roller = rolling_env.RollingEnv(scene, rts_day=14)
    env = replenish_env.ReplenishEnv(df, roller)
    agent = ReplenishAgent(env, device="cpu")
    agent.train(5000)
    agent.save_model("rep_model.pth")
    agent.plot_rewards()


if __name__ == "__main__":
    # 创建参数解析器
    # parser = argparse.ArgumentParser(description='PPO训练程序')
    # parser.add_argument('--ui', action='store_true',
    #                     help='使用界面模式训练，默认为无界面模式')
    #
    # args = parser.parse_args()
    #
    # if args.ui:
    #     train_with_ui()
    # else:
    #     train_with_no_ui()
    train_with_no_ui()



