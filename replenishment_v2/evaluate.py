"""
评估入口
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import ReplenishmentDataset
from src.environment import create_env
from src.agent import PPOAgent
from src.reward import create_reward
from src.utils import load_config, set_seed


def evaluate_model(
    env,
    agent: PPOAgent,
    verbose: bool = True,
) -> dict:
    """
    详细评估模型
    """
    state_map = env.reset()
    sku_ids = list(state_map.keys())
    done_all = False
    
    # 每个 SKU 的详细记录
    sku_records = {sku_id: [] for sku_id in sku_ids}
    
    step = 0
    pbar = tqdm(total=max(env.dataset.n_days_map.values()), desc="Evaluating") if verbose else None
    
    while not done_all:
        action_map = {}
        
        for sku_id in sku_ids:
            if env.done_map.get(sku_id, False):
                continue
            
            state = state_map[sku_id]
            action, _ = agent.select_action(state, deterministic=True)
            action_map[sku_id] = action
        
        next_state_map, reward_map, done_map, info_map = env.step(action_map)
        
        # 记录详细信息
        for sku_id, info in info_map.items():
            sku_records[sku_id].append({
                "step": step,
                "multiplier": info.get("multiplier", 1.0),
                "replenish": info["replenish_qty"],
                "bind": info["step_info"]["bind"],
                "rts": info["step_info"]["rts"],
                "sold": info["step_info"]["sold"],
                "overnight": info["step_info"]["overnight"],
                "reward": reward_map[sku_id],
            })
        
        state_map = next_state_map
        done_all = all(done_map.values())
        step += 1
        
        if pbar:
            pbar.update(1)
    
    if pbar:
        pbar.close()
    
    # 计算全局指标
    global_metrics = env.get_metrics()
    
    # 计算每个 SKU 的指标
    sku_metrics = {}
    for sku_id, records in sku_records.items():
        if len(records) == 0:
            continue
        
        total_replenish = sum(r["replenish"] for r in records)
        total_bind = sum(r["bind"] for r in records)
        total_rts = sum(r["rts"] for r in records)
        total_sold = sum(r["sold"] for r in records)
        
        sku_metrics[sku_id] = {
            "total_replenish": total_replenish,
            "total_bind": total_bind,
            "total_rts": total_rts,
            "total_sold": total_sold,
            "rts_rate": total_rts / total_replenish * 100 if total_replenish > 0 else 0,
            "avg_multiplier": np.mean([r["multiplier"] for r in records]),
        }
    
    return {
        "global": global_metrics,
        "sku_metrics": sku_metrics,
        "sku_records": sku_records,
    }


def main():
    parser = argparse.ArgumentParser(description="库存补货 RL 评估")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="配置文件路径")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--data_path", type=str, default=None, help="测试数据路径")
    parser.add_argument("--output", type=str, default=None, help="结果输出路径")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖数据路径
    if args.data_path:
        config["data"]["eval_path"] = args.data_path
    
    # 设置随机种子
    set_seed(config["task"].get("seed", 42))
    
    # 设备
    device = config["task"].get("device", "cpu")
    print(f"[Eval] Device: {device}")
    
    # 加载数据
    print("[Eval] Loading dataset...")
    data_path = args.data_path or config["data"].get("eval_path", config["data"]["train_path"])
    static_features = config["env"]["state_features"]["static"]
    dataset = ReplenishmentDataset(
        file_path=data_path,
        static_features=static_features,
    )
    print(f"[Eval] Dataset: {dataset.n_skus} SKUs")
    
    # 创建环境
    print("[Eval] Creating environment...")
    env = create_env(dataset, config)
    
    # 加载模型
    print(f"[Eval] Loading model from {args.model_path}...")
    agent = PPOAgent(config, device=device)
    agent.load(args.model_path)
    
    # 评估
    print("[Eval] Running evaluation...")
    results = evaluate_model(env, agent, verbose=args.verbose)
    
    # 打印结果
    global_metrics = results["global"]
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"ACC (售出率):      {global_metrics['acc']:.2f}%")
    print(f"RTS (退货率):      {global_metrics['rts_rate']:.2f}%")
    print(f"总补货量:          {global_metrics['total_replenish']:.0f}")
    print(f"总售出量:          {global_metrics['total_sales']:.0f}")
    print(f"总退货量:          {global_metrics['total_rts']:.0f}")
    print(f"总缺货量:          {global_metrics['total_stockout']:.0f}")
    print(f"市场销量:          {global_metrics['market_sales']:.0f}")
    print("=" * 60)
    
    # 与 baseline 对比
    baseline_acc = 75.0
    baseline_rts = 2.4
    
    acc_diff = global_metrics["acc"] - baseline_acc
    rts_diff = global_metrics["rts_rate"] - baseline_rts
    
    print(f"\n与 Baseline 对比:")
    print(f"  ACC: {acc_diff:+.2f}% ({'✓ 达标' if acc_diff >= 5 else '✗ 未达标'})")
    print(f"  RTS: {rts_diff:+.2f}% ({'✓ 达标' if rts_diff <= 0 else '✗ 未达标'})")
    
    if acc_diff >= 5 and rts_diff <= 0:
        print("\n🎉 恭喜! 达成目标: ACC提升≥5%, RTS不升高!")
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存 SKU 级别指标
        sku_df = pd.DataFrame.from_dict(results["sku_metrics"], orient="index")
        sku_df.to_csv(output_path.with_suffix(".sku_metrics.csv"))
        
        # 保存全局指标
        with open(output_path, "w") as f:
            f.write("EVALUATION RESULTS\n")
            f.write("=" * 40 + "\n")
            for key, value in global_metrics.items():
                f.write(f"{key}: {value}\n")
            f.write("\nBASELINE COMPARISON\n")
            f.write(f"ACC diff: {acc_diff:+.2f}%\n")
            f.write(f"RTS diff: {rts_diff:+.2f}%\n")
        
        print(f"\n[Eval] Results saved to {args.output}")
    
    return global_metrics


if __name__ == "__main__":
    main()
