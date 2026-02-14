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
from src.utils import load_config, set_seed, StateNormalizer


def evaluate_model(
    env,
    agent: PPOAgent,
    state_normalizer=None,
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
            if state_normalizer is not None:
                state = state_normalizer(state, update=False)
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


DATASETS = [
    "100k_sku.parquet",
    "100k_sku_v2.parquet",
    "100k_sku_v3.parquet",
    "100k_sku_v4.parquet",
    "100k_sku_v6.parquet",
]


def eval_single(config, data_path, agent, device, verbose=False):
    """对单个数据集评估，返回 (acc, rts)"""
    static_features = config["env"]["state_features"]["static"]
    dataset = ReplenishmentDataset(
        file_path=data_path,
        static_features=static_features,
    )
    env = create_env(dataset, config)

    state_normalizer = None
    norm_clip = config.get("training", {}).get("norm_clip", 10.0)
    ckpt = agent._last_checkpoint
    if ckpt and "state_norm_state" in ckpt:
        state_normalizer = StateNormalizer(
            shape=(env.state_dim,), clip=norm_clip, update=False)
        state_normalizer.running_ms.load_state(ckpt["state_norm_state"])

    results = evaluate_model(env, agent, state_normalizer=state_normalizer, verbose=verbose)
    g = results["global"]
    return g["acc"], g["rts_rate"]


def main():
    parser = argparse.ArgumentParser(description="库存补货 RL 评估")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="配置文件路径")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--data_path", type=str, default=None, help="单个测试数据路径（不传则跑5个100k）")
    parser.add_argument("--output", type=str, default=None, help="结果输出路径")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["task"].get("seed", 42))
    device = config["task"].get("device", "cpu")

    # 加载模型（只加载一次）
    agent = PPOAgent(config, device=device)
    checkpoint = agent.load(args.model_path)
    agent._last_checkpoint = checkpoint

    # 单数据集模式
    if args.data_path:
        acc, rts = eval_single(config, args.data_path, agent, device, args.verbose)
        print(f"ACC={acc:.2f}%  RTS={rts:.2f}%")
        return

    # 批量模式: 依次评估5个100k测试集
    data_dir = Path(__file__).parent / ".." / "data"
    summary = []

    print("=" * 55)
    print(f"Batch Evaluate: {len(DATASETS)} datasets")
    print(f"Model: {args.model_path}")
    print("=" * 55)

    for i, ds in enumerate(DATASETS, 1):
        dp = str((data_dir / ds).resolve())
        print(f"\n[{i}/{len(DATASETS)}] {ds} ...", flush=True)
        acc, rts = eval_single(config, dp, agent, device, args.verbose)
        status = "PASS" if acc >= 80.0 and rts <= 2.4 else "FAIL"
        print(f"  ACC={acc:.2f}%  RTS={rts:.2f}%  [{status}]")
        summary.append({"dataset": ds, "acc": acc, "rts": rts, "status": status})

    # 汇总
    avg_acc = sum(r["acc"] for r in summary) / len(summary)
    avg_rts = sum(r["rts"] for r in summary) / len(summary)
    all_pass = all(r["status"] == "PASS" for r in summary)

    print("\n" + "=" * 55)
    print(f"{'Dataset':<25} {'ACC':>8} {'RTS':>8} {'Status':>8}")
    print("-" * 55)
    for r in summary:
        print(f"{r['dataset']:<25} {r['acc']:>7.2f}% {r['rts']:>7.2f}% {r['status']:>8}")
    print("-" * 55)
    print(f"{'Average':<25} {avg_acc:>7.2f}% {avg_rts:>7.2f}%")
    print(f"Verdict: {'ALL PASS' if all_pass else 'NOT ALL PASS'}")
    print(f"Target: ACC >= 80%, RTS <= 2.4%")

    # 保存日志
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(f"{'Dataset':<25} {'ACC':>8} {'RTS':>8} {'Status':>8}\n")
            f.write("-" * 55 + "\n")
            for r in summary:
                f.write(f"{r['dataset']:<25} {r['acc']:>7.2f}% {r['rts']:>7.2f}% {r['status']:>8}\n")
            f.write("-" * 55 + "\n")
            f.write(f"{'Average':<25} {avg_acc:>7.2f}% {avg_rts:>7.2f}%\n")
        print(f"\nLog saved to {out}")


if __name__ == "__main__":
    main()
