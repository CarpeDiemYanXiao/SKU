"""
训练入口
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import ReplenishmentDataset
from src.environment import create_env
from src.agent import PPOAgent, RolloutBuffer
from src.reward import create_reward
from src.utils import (
    load_config, 
    set_seed, 
    create_output_dir,
    EarlyStopping,
    MetricsTracker,
    format_metrics,
)

# TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

# NPU 支持 (华为昇腾)
HAS_NPU = False
try:
    import torch_npu
    HAS_NPU = torch.npu.is_available()
except ImportError:
    pass


def train_episode(
    env,
    agent: PPOAgent,
    buffer: RolloutBuffer,
) -> dict:
    """
    训练一个 episode (批量优化版本)
    
    Returns:
        episode 统计信息
    """
    # 注意：不在这里 reset buffer，让 buffer 可以累积多个 episode
    state_map = env.reset()
    
    sku_ids = list(state_map.keys())
    done_all = False
    episode_rewards = {sku_id: 0.0 for sku_id in sku_ids}
    
    while not done_all:
        # 收集所有活跃 SKU 的状态（批量处理）
        active_sku_ids = [sku_id for sku_id in sku_ids if not env.done_map.get(sku_id, False)]
        
        if not active_sku_ids:
            break
        
        # 批量构建状态矩阵
        states_batch = np.array([state_map[sku_id] for sku_id in active_sku_ids])
        
        # 批量推理（一次 GPU 调用）
        actions_batch, log_probs_batch = agent.select_actions_batch(states_batch)
        values_batch = agent.get_values_batch(states_batch)
        
        # 构建 action_map 和暂存数据
        action_map = {}
        buffer._temp = {}
        
        for i, sku_id in enumerate(active_sku_ids):
            action = int(actions_batch[i])
            action_map[sku_id] = action
            
            buffer._temp[sku_id] = {
                "state": states_batch[i],
                "action": action,
                "log_prob": log_probs_batch[i],
                "value": values_batch[i],
            }
        
        # 环境步进
        next_state_map, reward_map, done_map, info_map = env.step(action_map)
        
        # 存储到 buffer
        for sku_id in action_map.keys():
            temp = buffer._temp[sku_id]
            reward = reward_map.get(sku_id, 0.0)
            done = done_map.get(sku_id, False)
            
            buffer.add(
                state=temp["state"],
                action=temp["action"],
                reward=reward,
                done=done,
                log_prob=temp["log_prob"],
                value=temp["value"],
            )
            
            episode_rewards[sku_id] += reward
        
        # 更新状态
        state_map = next_state_map
        
        # 检查是否全部结束
        done_all = all(done_map.values())
    
    # 获取环境指标
    env_metrics = env.get_metrics()
    
    return {
        "total_reward": sum(episode_rewards.values()),
        "mean_reward": np.mean(list(episode_rewards.values())),
        "buffer_size": len(buffer),
        **env_metrics,
    }


def evaluate(
    env,
    agent: PPOAgent,
) -> dict:
    """
    评估模型
    """
    state_map = env.reset()
    sku_ids = list(state_map.keys())
    done_all = False
    
    while not done_all:
        action_map = {}
        
        for sku_id in sku_ids:
            if env.done_map.get(sku_id, False):
                continue
            
            state = state_map[sku_id]
            action, _ = agent.select_action(state, deterministic=True)
            action_map[sku_id] = action
        
        next_state_map, _, done_map, _ = env.step(action_map)
        state_map = next_state_map
        done_all = all(done_map.values())
    
    return env.get_metrics()


def main():
    parser = argparse.ArgumentParser(description="库存补货 RL 训练")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="配置文件路径")
    parser.add_argument("--data_path", type=str, default=None, help="训练数据路径 (覆盖配置)")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录 (覆盖配置)")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的模型路径")
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖配置
    if args.data_path:
        config["data"]["train_path"] = args.data_path
    
    # 设置随机种子
    seed = config["task"].get("seed", 42)
    set_seed(seed)
    
    # 创建输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = create_output_dir(config)
    
    print(f"[Train] Output directory: {output_dir}")
    
    # TensorBoard
    writer = None
    if HAS_TENSORBOARD and config["logging"].get("tensorboard", True):
        tb_dir = output_dir / "tensorboard"
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))
    
    # 设备 - 自动检测 NPU/CUDA 可用性
    device_config = config["task"].get("device", "auto")
    
    if device_config == "auto":
        # 自动选择：NPU > CUDA > CPU
        if HAS_NPU:
            device = "npu:0"
            print(f"[Train] Device: {device} (NPU available)")
        elif torch.cuda.is_available():
            device = "cuda:0"
            print(f"[Train] Device: {device} (CUDA available)")
        else:
            device = "cpu"
            print(f"[Train] Device: {device}")
    elif "npu" in device_config:
        if HAS_NPU:
            device = device_config
            print(f"[Train] Device: {device} (NPU available)")
        else:
            device = "cpu"
            print(f"[Train] Warning: NPU not available, using CPU instead")
    elif "cuda" in device_config:
        if torch.cuda.is_available():
            device = device_config
            print(f"[Train] Device: {device} (CUDA available)")
        else:
            device = "cpu"
            print(f"[Train] Warning: CUDA not available, using CPU instead")
    else:
        device = device_config
        print(f"[Train] Device: {device}")
    
    # 加载数据
    print("[Train] Loading dataset...")
    static_features = config["env"]["state_features"]["static"]
    dataset = ReplenishmentDataset(
        file_path=config["data"]["train_path"],
        static_features=static_features,
    )
    print(f"[Train] Dataset: {dataset.n_skus} SKUs")
    
    # 创建环境
    print("[Train] Creating environment...")
    reward_fn = create_reward(config)
    env = create_env(dataset, config)
    
    # 创建 Agent
    print("[Train] Creating agent...")
    agent = PPOAgent(config, device=device)
    
    # 恢复训练
    if args.resume:
        agent.load(args.resume)
        print(f"[Train] Resumed from {args.resume}")
    
    # 训练配置
    train_cfg = config["training"]
    max_episodes = train_cfg["max_episodes"]
    eval_interval = train_cfg["eval_interval"]
    save_interval = train_cfg["save_interval"]
    print_interval = config["logging"].get("print_interval", 5)
    accumulate_episodes = train_cfg.get("accumulate_episodes", 1)  # NPU优化：累积多个episode
    
    # 课程学习
    curriculum_cfg = train_cfg.get("curriculum", {})
    use_curriculum = curriculum_cfg.get("enabled", False)
    curriculum_stages = curriculum_cfg.get("stages", [])
    current_stage_idx = 0
    
    # 早停
    early_stopping = EarlyStopping(
        patience=train_cfg.get("early_stop_patience", 50),
        mode="max"
    )
    
    # 指标追踪
    metrics_tracker = MetricsTracker()
    best_acc = 0.0
    best_rts = float('inf')
    best_combined_score = -float('inf')  # ACC - RTS_penalty
    
    # Rollout buffer
    buffer = RolloutBuffer()
    
    print(f"[Train] Starting training for {max_episodes} episodes...")
    print(f"[Train] NPU optimization: accumulate {accumulate_episodes} episodes before update")
    print("=" * 60)
    
    # 创建进度条 (每5轮更新一次)
    pbar = tqdm(
        range(1, max_episodes + 1), 
        desc="Training", 
        miniters=5, 
        mininterval=1.0,
        dynamic_ncols=True,  # 自适应终端宽度
        leave=True,
    )
    
    for episode in pbar:
        # 课程学习: 更新 reward 权重
        if use_curriculum:
            for i, stage in enumerate(curriculum_stages):
                stage_end = sum(s["episodes"] for s in curriculum_stages[:i+1])
                if episode <= stage_end:
                    if i != current_stage_idx:
                        current_stage_idx = i
                        tqdm.write(f"[Curriculum] Entering stage: {stage['name']}")
                        # 更新 reward 权重
                        if hasattr(reward_fn, 'rts_weight'):
                            reward_fn.rts_weight = stage.get("rts_weight", 1.0)
                        if hasattr(reward_fn, 'bind_weight'):
                            reward_fn.bind_weight = stage.get("bind_weight", 0.25)
                    break
        
        # 训练一个 episode
        episode_stats = train_episode(env, agent, buffer)
        
        # PPO 更新（累积多个episode后再更新，提高NPU利用率）
        if len(buffer) > 0 and episode % accumulate_episodes == 0:
            rollout_data = buffer.get()
            train_stats = agent.update(rollout_data)
            buffer.reset()  # 更新后清空buffer
        else:
            train_stats = {"policy_loss": 0, "value_loss": 0, "entropy": 0}
        
        # 记录指标
        all_stats = {**episode_stats, **train_stats}
        metrics_tracker.add(all_stats, episode)
        
        if writer:
            for key, value in all_stats.items():
                writer.add_scalar(f"train/{key}", value, episode)
        
        # 打印进度
        if episode % print_interval == 0:
            # 更新进度条后缀信息（简化显示）
            pbar.set_postfix_str(
                f"ACC={episode_stats['acc']:.1f}% RTS={episode_stats['rts_rate']:.1f}% R={episode_stats['mean_reward']:.1f}"
            )
        
        # 评估
        if episode % eval_interval == 0:
            eval_metrics = evaluate(env, agent)
            
            # 暂停进度条输出评估信息
            tqdm.write(f"[Eval Episode {episode}] ACC={eval_metrics['acc']:.2f}% | RTS={eval_metrics['rts_rate']:.2f}%")
            
            if writer:
                writer.add_scalar("eval/acc", eval_metrics["acc"], episode)
                writer.add_scalar("eval/rts_rate", eval_metrics["rts_rate"], episode)
            
            # ========== 核心: 模型选择逻辑 ==========
            # 目标: RTS≤2.4% 且 ACC≥80%
            # 策略: 优先找满足RTS约束的，然后在其中选ACC最高的
            target_rts = train_cfg.get("target_rts", 2.4)
            target_acc = train_cfg.get("target_acc", 80.0)
            
            current_rts = eval_metrics["rts_rate"]
            current_acc = eval_metrics["acc"]
            
            # 计算综合得分
            # 如果RTS达标: score = ACC + bonus
            # 如果RTS不达标: score = ACC - heavy_penalty
            if current_rts <= target_rts:
                # RTS达标，ACC越高越好
                rts_bonus = 10.0  # 达标奖励
                combined_score = current_acc + rts_bonus
            else:
                # RTS超标，根据超出程度惩罚
                rts_excess = current_rts - target_rts
                # 超出越多惩罚越重，但不要太重导致模型不敢补货
                rts_penalty = rts_excess * 3.0
                combined_score = current_acc - rts_penalty
            
            # 保存最佳模型
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_acc = eval_metrics["acc"]
                best_rts = eval_metrics["rts_rate"]
                agent.save(str(output_dir / "best_model.pth"))
                status = "✓" if current_rts <= target_rts else "×"
                tqdm.write(f"  -> New best! ACC={best_acc:.2f}%, RTS={best_rts:.2f}% [{status}]")
            
            # 早停检查
            if early_stopping(combined_score):
                tqdm.write(f"[Train] Early stopping at episode {episode}")
                pbar.close()
                break
        
        # 定期保存
        if episode % save_interval == 0:
            agent.save(str(output_dir / f"model_ep{episode}.pth"))
    
    # 保存最终模型
    agent.save(str(output_dir / "final_model.pth"))
    
    print("=" * 60)
    print(f"[Train] Training completed!")
    print(f"[Train] Best ACC: {best_acc:.2f}%, Best RTS: {best_rts:.2f}%")
    print(f"[Train] Models saved to: {output_dir}")
    
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
