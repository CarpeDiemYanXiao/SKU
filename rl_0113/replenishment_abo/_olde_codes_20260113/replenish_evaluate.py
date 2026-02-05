import argparse
import datetime
import os  # 添加os导入
import numpy as np
import pandas as pd
from agents.replenish_agent import ReplenishAgent, GetAction, Get_Continue_Action
from agents.replenish_agent_combine import ReplenishAgentCombine
from atp_sim_sdk.roller import Roller
import json
from datetime import datetime, timedelta
import time
from task import task_dict
from utils.io import read_json
from pathlib import Path
from utils.helper import get_conf_path, create_path_with_suffix
import matplotlib.pyplot as plt
import seaborn as sns


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="", help="task name")
    parser.add_argument(
        "--data_ver",
        type=str,
        required=True,
        help="数据版本,任务名+数据版本=数据文件夹,所有生成的数据都放在数据文件夹下面",
    )
    parser.add_argument(
        "--para_ver", type=str, required=True, help="实验版本,所有生成的数据在数据文件夹下面,由实验版本作为开头"
    )
    ##parser.add_argument("--is_simu", type=int, default=0, help="")
    parser.add_argument("--test_version", type=str, default=argparse.SUPPRESS, help="测试的后缀,若不指定则不带后缀")
    parser.add_argument(
        "--specified_model_path", type=str, default=argparse.SUPPRESS, help="指定模型文件，不指定则使用默认值"
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        required=True,
        help="配置测试数据路径",
    )
    args = parser.parse_args()
    args.base_dir = str(Path(__file__).parents[0])  # project dir
    return args


def cal_date(start_date, delta_days):
    end_date = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=delta_days)
    return end_date.strftime("%Y-%m-%d")


def gen_multipliers(actions_map, start_date, task_name):

    key_ls = []
    res_data_ls = []

    for key, value in actions_map.items():
        key_ls.append(key)
        res_data_ls.append(value)

    df = pd.DataFrame(data={"idx": key_ls, "multilpier": res_data_ls})

    df["multiplier_index"] = df["multilpier"].apply(lambda x: list(range(len(x))))

    exploded_df = df.explode(["multilpier", "multiplier_index"])

    exploded_df["ds"] = exploded_df.apply(lambda s: cal_date("2024-12-16", s["multiplier_index"]), axis=1)
    exploded_df["dt_version"] = task_name
    exploded_df = exploded_df[["idx", "ds", "multilpier", "dt_version"]]
    exploded_df.columns = ["idx", "ds", "multilpier", "dt_version"]
    return exploded_df


def gen_simu_result(agent: ReplenishAgent, config):
    actions_all = []
    head_sku_actions_ls = []
    tail_sku_actions_ls = []
    actions_ls = []
    states_all = []
    ####所有商品进行初始化
    ###商品network-state初始化，日期初始化
    actions_map = {sku: [] for sku in agent.sku_id_ls}
    state_map = {sku: agent.reset_network_state(sku) for sku in agent.sku_id_ls}  ##idx:state
    if agent.use_state_norm == 1:
        print("启动reward归一化")
        for sku in agent.sku_id_ls:
            state_map[sku] = agent.state_norm(state_map[sku], update=False)

    states_map = {sku: [] for sku in agent.sku_id_ls}
    day_idx_map = {sku: agent.reset_day_idx() for sku in agent.sku_id_ls}
    done_map = {sku: False for sku in agent.sku_id_ls}
    agent.env.reset()  ###需要对所有sku进行初始化
    done_all = False
    total_reward = 0
    ###
    simu_rts_qty = 0
    simu_bind_qty = 0
    simu_rep_qty = 0
    simu_actual_qty = 0
    rolling_cnt = 0

    while not done_all:
        rolling_cnt += 1
        action_map = {}
        for sku in agent.sku_id_ls:
            sku_action_map = {}
            states_map[sku].append(state_map[sku])
            action = agent.select_action_deterministic(state_map[sku])
            actions_map[sku].append(agent.action_ls[int(action)])
            if config.model_name == "ppo_continue_action":
                actions_ls.append(action)
                sku_action_map["get_action"] = Get_Continue_Action(action)
            else:
                actions_ls.append(agent.action_ls[int(action)])
                sku_action_map["get_action"] = GetAction(agent.action_ls, action)
                if agent.dq_map[sku][day_idx_map[sku]] >= agent.config.head_sku_standard: # 统计头部品/尾部品动作分布
                    head_sku_actions_ls.append(agent.action_ls[int(action)])
                else:
                    tail_sku_actions_ls.append(agent.action_ls[int(action)])

            sku_action_map["evaluate"] = True
            sku_action_map["day_idx"] = day_idx_map[sku]
            action_map[sku] = sku_action_map
            ###TODO如何标识，需要确认
        rolling_state_map, result_map, bind, rts = agent.env.batch_step(action_map, evaluate=True)

        simu_rts_qty += rts
        simu_bind_qty += bind

        ###state_dict
        for sku in agent.sku_id_ls:
            simu_actual_qty += agent.sales_map[sku][day_idx_map[sku]]
            simu_rep_qty += rolling_state_map[sku]["abo_qty"]
            ###生成policy-network-state
            ###更新day_idx
            day_idx_map[sku] += 1

            new_network_state = agent.gen_network_state(rolling_state_map[sku], sku, day_idx_map[sku])
            if agent.use_state_norm == 1:
                new_network_state = agent.state_norm(new_network_state, update=False)
            ###更新sku的state
            state_map[sku] = new_network_state
            done_map[sku] = day_idx_map[sku] >= agent.end_date_map[sku] - 1
            ##total_reward += reward
        if set(done_map.values()) == {True}:  ###所有商品都是done才能结束
            Roller(debug=os.getenv("DEBUG", "False").lower() == "true").add_result_to_csv(
                simu_bind_qty, simu_rep_qty, simu_rts_qty
            )
            done_all = True
    print("#######gen result data##########")
    print(f"rolling_cnt={rolling_cnt}")
    print(f"actual_sales={simu_actual_qty}")
    print(f"total_sales ={simu_actual_qty} ,simu_rts_qty= {simu_rts_qty}, simu_bind_qty={simu_bind_qty}")
    print(f"acc_rate={simu_bind_qty/(simu_actual_qty+0.001)},rts_rate={simu_rts_qty/(simu_bind_qty+0.001)}")
    ###TODO:stat_date更新
    actions_all.append(actions_map)
    states_all.append(states_map)
    total_sales_all = [simu_actual_qty]
    simu_rts_qty_all = [simu_rts_qty]
    simu_bind_qty_all = [simu_bind_qty]
    # 将列表转换为 Pandas Series
    actions_series = pd.Series(actions_ls)
    action_dist_dict = actions_series.value_counts().to_dict()

    head_actions_series = pd.Series(head_sku_actions_ls)
    head_action_dist_dict = head_actions_series.value_counts().to_dict()

    tail_actions_series = pd.Series(tail_sku_actions_ls)
    tail_action_dist_dict = tail_actions_series.value_counts().to_dict()
    # print(f"actions_dist: {action_dist_dict}")

    # # 绘制直方图
    # plt.hist(actions_ls, bins='auto')  # 'auto' 让matplotlib决定最佳的bin数量
    # # 添加标题和标签
    # plt.title('Distribution of actions')
    # plt.xlabel('Action')
    # plt.ylabel('Frequency')
    # # 指定完整的文件路径来保存图像
    # file_path = os.path.join(config.outs_dir, 'distribution_plot.png')
    # plt.savefig(file_path)
    # # 清除当前图形
    # plt.clf()
    
    # # 绘制直方图
    # plt.hist(head_sku_actions_ls, bins='auto')  # 'auto' 让matplotlib决定最佳的bin数量
    # # 添加标题和标签
    # plt.title('Distribution of head_sku actions')
    # plt.xlabel('Action')
    # plt.ylabel('Frequency')
    # # 指定完整的文件路径来保存图像
    # file_path = os.path.join(config.outs_dir, 'head_sku_distribution_plot.png')
    # plt.savefig(file_path)
    # # 清除当前图形
    # plt.clf()

    # # 绘制直方图
    # plt.hist(tail_sku_actions_ls, bins='auto')  # 'auto' 让matplotlib决定最佳的bin数量
    # # 添加标题和标签
    # plt.title('Distribution of tail_sku actions')
    # plt.xlabel('Action')
    # plt.ylabel('Frequency')
    # # 指定完整的文件路径来保存图像
    # file_path = os.path.join(config.outs_dir, 'tail_sku_distribution_plot.png')
    # plt.savefig(file_path)
    
    # 绘制三个列表的密度分布图
    # sns.kdeplot(actions_ls, fill=True, color="skyblue", label="所有品")
    sns.kdeplot(head_sku_actions_ls, fill=True, color="skyblue", label="head_sku")
    sns.kdeplot(tail_sku_actions_ls, fill=True, color="red", label="tail_sku")

    # 添加标题和标签
    plt.title('Density Plot of Actions')
    plt.xlabel('Action Value')
    plt.ylabel('Density')
    # 添加图例
    plt.legend()
    # 保存
    file_path = os.path.join(config.outs_dir, 'action_distribution.png')
    plt.savefig(file_path)
    
    if config.model_name == "ppo_continue_action":  # ；连续动作模型，csv中不保存action_dist
        res_df = pd.DataFrame(
            data={
                "actual_qty": total_sales_all,
                "bind_qty": simu_bind_qty_all,
                "rts_qty": simu_rts_qty_all,
            }
        )
    else:
        res_df = pd.DataFrame(
            data={
                "actual_qty": total_sales_all,
                "bind_qty": simu_bind_qty_all,
                "rts_qty": simu_rts_qty_all,
                "action_dist": [action_dist_dict],
                "head_action_dist_dict":[head_action_dist_dict],
                "tail_action_dist_dict":[tail_action_dist_dict]
            }
        )
    res_df["acc_rate"] = res_df["bind_qty"] / res_df["actual_qty"]
    res_df["rts_rate"] = res_df["rts_qty"] / res_df["bind_qty"]
    res_df.to_csv(config.res_data_path, index=False)
    # multiplier_res_data = os.path.join("output", agent.task_name, f"simu_multiplier_data.csv")
    # multiplier_df.to_csv(multiplier_res_data, index=False)

    agent.mlops.log_eval_metrics(
        task_name=agent.task_name,
        params={
            "actual_qty": total_sales_all,
            "bind_qty": simu_bind_qty_all,
            "rts_qty": simu_rts_qty_all,
            "action_dist": action_dist_dict,
        },
        action_dist_dict=action_dist_dict,
        metrics={
            "acc_rate": res_df["acc_rate"].values[0],
            "rts_rate": res_df["rts_rate"].values[0],
        },
    )


def simu(config):
    start_time = time.time()
    # args = arg_parser()
    # task_name = args.task_name
    # is_combine = args.is_combine
    # config_path = args.json_path
    # test_version = args.test_version
    # model_path = args.model_path
    # test_data_path = args.test_data_path
    # with open(config_path, "r") as f:
    #     cfg = json.load(f)
    # # print(cfg["data_path"])
    # cfg["data_path"] = test_data_path
    # cfg["test_model_output"] = "onnx"
    print(config.data_path)

    ra = ReplenishAgent(None, config)
    ra.load_model(config.specified_model_path)
    print(f"print load successfully, model_path: {config.specified_model_path}")
    print("begin to simu results")
    gen_simu_result(ra, config)
    # ra.gen_simu_result(model_path)
    print(f"elapsed_time= {time.time()-start_time}")


def to_onnx():
    args = arg_parser()
    config_path = args.cfg_path
    with open(config_path, "r") as f:
        cfg = json.load(f)

    ra = ReplenishAgent(None, cfg)
    model_path = os.path.join("output", ra.task_name, "repl_policy_model.pth")
    ra.load_model(model_path)
    ra.save_to_onnx(os.path.join("output", ra.task_name, "repl_policy_model.onnx"))
    print("save to onnx successfully")


if __name__ == "__main__":
    args = arg_parser()
    config = task_dict["base"]()
    config.update(read_json(get_conf_path(args)), priority="high") # 加载model.json文件，会加载state的字段和data的路径
    config.update(args, priority="high")
    # config.res_data_path = config.outs_dir.joinpath("simu_res_data.csv")
    config.res_data_path = os.path.join(config.outs_dir, "simu_res_data.csv")
    # 根据后缀更新结果保存的路径
    if hasattr(config, "test_version"):
        config.res_data_path = create_path_with_suffix(config.res_data_path, config.test_version)
    if not hasattr(config, "specified_model_path"):
        config.specified_model_path = config.policy_model_filename

    config.data_path = config.test_data_path
    # todo:让eval也兼容多卡的方法,目前先手动设置为单卡
    config.distributed = False
    config.rank = 0
    config.world_size = 1

    simu(config)
    # simu_test()
    # to_onnx()
