"""
训练入口（增强版）
参考 rl_0113/replenishment_abo 的优化实践

新增功能:
- 课程学习 (Curriculum Learning)
- 完整的 Checkpoint 保存/加载
- 状态/奖励在线归一化
- 分布式训练支持
- 更详细的训练日志
"""

import os
import sys
import argparse
import warnings
import time
import json
from pathlib import Path
from datetime import datetime

# 过滤掉 torch_npu 的兼容性警告
warnings.filterwarnings("ignore", message=".*AutoNonVariableTypeMode.*")
warnings.filterwarnings("ignore", message=".*owner does not match.*")

# 在导入 torch 之前，根据配置决定是否使用 NPU
# 先加载配置检查设备设置
_config_path = Path(__file__).parent / "config" / "default.yaml"
_use_npu = True  # 默认尝试使用 NPU
if _config_path.exists():
    with open(_config_path, 'r', encoding='utf-8') as f:
        import re
        content = f.read()
        match = re.search(r'device:\s*["\']?(\w+)', content)
        if match and match.group(1).lower() == 'cpu':
            _use_npu = False
            # 设置环境变量禁用 NPU
            os.environ['NPU_VISIBLE_DEVICES'] = ''

import numpy as np
import torch
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import ReplenishmentDataset
from src.environment import create_env, create_env_with_reward
from src.agent import PPOAgent
from src.reward import create_reward
from src.utils import (
    load_config, 
    save_config,
    set_seed, 
    create_output_dir,
    EarlyStopping,
    MetricsTracker,
    format_metrics,
    StateNormalizer,
    RewardNormalizer,
)

# TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

# NPU 支持 (华为昇腾) - 仅在需要时加载
HAS_NPU = False
if _use_npu:
    try:
        import torch_npu
        HAS_NPU = torch.npu.is_available()
    except ImportError:
        pass


DATASETS = [
    "100k_sku.parquet",
    "100k_sku_v2.parquet",
    "100k_sku_v3.parquet",
    "100k_sku_v4.parquet",
    "100k_sku_v6.parquet",
]


class RolloutBuffer:
    """
    Rollout Buffer（增强版）
    存储 rollout 数据，支持多 episode 累积
    """
    
    def __init__(self):
        self.reset()
        self._temp = {}  # 临时存储，用于批量处理
    
    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.sku_ids = []  # 追踪每条transition属于哪个SKU
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
        sku_id: str = None,
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.sku_ids.append(sku_id)
    
    def get(self) -> dict:
        return {
            "states": np.array(self.states),
            "actions": np.array(self.actions),
            "rewards": np.array(self.rewards),
            "dones": np.array(self.dones),
            "old_log_probs": np.array(self.log_probs),
            "values": np.array(self.values),
            "sku_ids": self.sku_ids,
        }
    
    def __len__(self):
        return len(self.states)


def compute_per_sku_gae(rollout_data: dict, gamma: float, gae_lambda: float) -> dict:
    """
    按SKU分段计算GAE（修复跨SKU GAE污染问题）
    
    原始实现将所有SKU的transition当做一条连续序列计算GAE，
    导致SKU切换处的next_value错误地使用了其他SKU的value。
    
    正确做法：每个SKU独立计算GAE，再拼接。
    参考rl_0113基线: refactor_agent.py 中也是按SKU分段计算。
    """
    from collections import defaultdict
    
    sku_ids = rollout_data["sku_ids"]
    rewards = rollout_data["rewards"]
    values = rollout_data["values"]
    dones = rollout_data["dones"]
    
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    returns = np.zeros(n, dtype=np.float32)
    
    # 按SKU分组索引
    sku_indices = defaultdict(list)
    for i, sid in enumerate(sku_ids):
        sku_indices[sid].append(i)
    
    # 每个SKU独立计算GAE
    for sid, indices in sku_indices.items():
        indices = sorted(indices)  # 确保时间顺序
        n_sku = len(indices)
        
        gae = 0.0
        for t in reversed(range(n_sku)):
            idx = indices[t]
            if t == n_sku - 1:
                next_val = 0.0
            else:
                next_val = values[indices[t + 1]]
            
            delta = rewards[idx] + gamma * next_val * (1 - dones[idx]) - values[idx]
            gae = delta + gamma * gae_lambda * (1 - dones[idx]) * gae
            advantages[idx] = gae
            returns[idx] = gae + values[idx]
    
    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns
    return rollout_data


def train_episode(
    env,
    agent: PPOAgent,
    buffer: RolloutBuffer,
    state_normalizer: StateNormalizer = None,
    reward_normalizer: RewardNormalizer = None,
    use_state_norm: bool = False,
    use_reward_norm: bool = False,
    use_terminal_reward: bool = True,
    sku_sample_size: int = 0,
) -> dict:
    """
    训练一个 episode (批量优化版本，支持状态/奖励归一化)
    
    Args:
        env: 训练环境
        agent: PPO Agent
        buffer: Rollout Buffer
        state_normalizer: 状态归一化器
        reward_normalizer: 奖励归一化器
        use_state_norm: 是否使用状态归一化
        use_reward_norm: 是否使用奖励归一化
        use_terminal_reward: 是否使用终止态奖励 (DeepStock论文)
    
    Returns:
        episode 统计信息
    """
    # 注意：不在这里 reset buffer，让 buffer 可以累积多个 episode
    state_map = env.reset(sku_sample_size=sku_sample_size)
    
    # 重置奖励归一化器的累积回报
    if reward_normalizer:
        reward_normalizer.reset()
    
    sku_ids = list(state_map.keys())
    done_all = False
    episode_rewards = {sku_id: 0.0 for sku_id in sku_ids}
    
    while not done_all:
        # 收集所有活跃 SKU 的状态（批量处理）
        active_sku_ids = [sku_id for sku_id in sku_ids if not env.done_map.get(sku_id, False)]
        
        if not active_sku_ids:
            break
        
        # 批量构建状态矩阵（可选归一化）
        if use_state_norm and state_normalizer:
            states_batch = np.array([
                state_normalizer(state_map[sku_id], update=True) 
                for sku_id in active_sku_ids
            ])
        else:
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
        
        # 存储到 buffer（可选奖励归一化）
        for sku_id in action_map.keys():
            temp = buffer._temp[sku_id]
            reward = reward_map.get(sku_id, 0.0)
            done = done_map.get(sku_id, False)
            
            # 奖励归一化
            if use_reward_norm and reward_normalizer:
                reward = reward_normalizer(reward, update=True)
            
            buffer.add(
                state=temp["state"],
                action=temp["action"],
                reward=reward,
                done=done,
                log_prob=temp["log_prob"],
                value=temp["value"],
                sku_id=sku_id,
            )
            
            episode_rewards[sku_id] += reward_map.get(sku_id, 0.0)  # 原始奖励用于统计
        
        # 更新状态
        state_map = next_state_map
        
        # 检查是否全部结束
        done_all = all(done_map.values())
    
    # ========== 终止态奖励 (DeepStock论文) ==========
    # 在episode结束时根据累计指标给予大的奖惩
    if use_terminal_reward and hasattr(env.reward_fn, 'compute_terminal_reward'):
        terminal_reward = env.reward_fn.compute_terminal_reward()
        
        # 将终止态奖励分配给最后几个时间步
        if len(buffer) > 0 and terminal_reward != 0:
            # 找到最后N步（每个SKU的最后一步）
            n_last_steps = min(len(sku_ids), len(buffer))
            for i in range(n_last_steps):
                idx = len(buffer.rewards) - 1 - i
                if idx >= 0:
                    # 终止态奖励按SKU数量均分
                    buffer.rewards[idx] += terminal_reward / len(sku_ids)
    
    # 重置reward_fn的累计统计
    if hasattr(env.reward_fn, 'reset'):
        env.reward_fn.reset()
    
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
    state_normalizer: StateNormalizer = None,
    use_state_norm: bool = False,
) -> dict:
    """
    评估模型
    
    Args:
        env: 环境
        agent: PPO Agent
        state_normalizer: 状态归一化器
        use_state_norm: 是否使用状态归一化
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
            # 评估时使用归一化但不更新统计量
            if use_state_norm and state_normalizer:
                state = state_normalizer(state, update=False)
            
            action, _ = agent.select_action(state, deterministic=True)
            action_map[sku_id] = action
        
        next_state_map, _, done_map, _ = env.step(action_map)
        state_map = next_state_map
        done_all = all(done_map.values())
    
    return env.get_metrics()


def main():
    parser = argparse.ArgumentParser(description="库存补货 RL 训练（增强版）")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="配置文件路径")
    parser.add_argument("--data_path", type=str, default=None, help="训练数据路径 (覆盖配置)")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录 (覆盖配置)")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的 checkpoint 路径")
    parser.add_argument("--resume_mode", type=str, default="full", choices=["full", "weights"], 
                        help="恢复模式: full=完全恢复, weights=只加载权重")
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
    
    # 保存配置副本
    save_config(config, str(output_dir / "config.yaml"))
    
    # TensorBoard
    writer = None
    if HAS_TENSORBOARD and config["logging"].get("tensorboard", True):
        tb_dir = output_dir / "tensorboard"
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))
    
    # 设备 - 自动检测 NPU/CUDA 可用性
    device_config = config["task"].get("device", "auto")
    
    if device_config == "auto":
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
    
    # 加载数据（多数据集）
    static_features = config["env"]["state_features"]["static"]
    data_dir = Path(__file__).parent / ".." / "data"
    single_data = config["data"].get("train_path", "")
    is_multi = any(k in single_data for k in ["100k"])

    reward_fn = create_reward(config)
    datasets = []
    envs = []

    if is_multi or args.data_path is None:
        available = [d for d in DATASETS if (data_dir / d).exists()]
        if len(available) >= 2:
            print(f"[Train] Multi-dataset mode: {len(available)} datasets")
            for ds_name in available:
                ds = ReplenishmentDataset(
                    file_path=str((data_dir / ds_name).resolve()),
                    static_features=static_features,
                )
                datasets.append(ds)
                envs.append(create_env_with_reward(ds, config, reward_fn))
                print(f"  - {ds_name}: {ds.n_skus} SKUs")
        else:
            ds = ReplenishmentDataset(
                file_path=config["data"]["train_path"],
                static_features=static_features,
            )
            datasets.append(ds)
            envs.append(create_env_with_reward(ds, config, reward_fn))
            print(f"[Train] Single dataset: {ds.n_skus} SKUs")
    else:
        ds = ReplenishmentDataset(
            file_path=config["data"]["train_path"],
            static_features=static_features,
        )
        datasets.append(ds)
        envs.append(create_env_with_reward(ds, config, reward_fn))
        print(f"[Train] Single dataset: {ds.n_skus} SKUs")

    n_envs = len(envs)
    env = envs[0]  # 默认环境（用于state_dim等）
    dataset = datasets[0]
    
    # 计算 state_dim
    n_dynamic = len(config["env"]["state_features"]["dynamic"])
    n_static = len(config["env"]["state_features"]["static"])
    state_dim = n_dynamic + n_static
    
    # 创建归一化器（参考 abo 项目）
    train_cfg = config["training"]
    use_state_norm = train_cfg.get("use_state_norm", False)
    use_reward_norm = train_cfg.get("use_reward_norm", False)
    norm_clip = train_cfg.get("norm_clip", 10.0)
    
    state_normalizer = None
    reward_normalizer = None
    
    if use_state_norm:
        state_normalizer = StateNormalizer(shape=(state_dim,), clip=norm_clip)
        state_normalizer.set_sample_num(dataset.n_skus)
        print(f"[Train] State normalization enabled (clip={norm_clip})")
    
    if use_reward_norm:
        gamma = config["ppo"]["gamma"]
        reward_normalizer = RewardNormalizer(clip=norm_clip, gamma=gamma)
        reward_normalizer.set_sample_num(dataset.n_skus)
        print(f"[Train] Reward normalization enabled (clip={norm_clip})")
    
    # 创建 Agent
    print("[Train] Creating agent...")
    agent = PPOAgent(config, device=device)
    
    # 恢复训练
    start_episode = 1
    if args.resume:
        checkpoint = agent.load(args.resume, mode=args.resume_mode)
        if args.resume_mode == "full" and "episode" in checkpoint:
            start_episode = checkpoint.get("episode", 1) + 1
        # 恢复归一化器状态
        if use_state_norm and "state_norm_state" in checkpoint:
            state_normalizer.running_ms.load_state(checkpoint["state_norm_state"])
        if use_reward_norm and "reward_norm_state" in checkpoint:
            reward_normalizer.running_ms.load_state(checkpoint["reward_norm_state"])
        print(f"[Train] Resumed from {args.resume}, starting at episode {start_episode}")
    
    # 训练配置
    max_episodes = train_cfg["max_episodes"]
    eval_interval = train_cfg["eval_interval"]
    save_interval = train_cfg["save_interval"]
    print_interval = config["logging"].get("print_interval", 5)
    accumulate_episodes = train_cfg.get("accumulate_episodes", 1)
    
    # 课程学习
    curriculum_cfg = train_cfg.get("curriculum", {})
    use_curriculum = curriculum_cfg.get("enabled", False)
    curriculum_stages = curriculum_cfg.get("stages", [])
    current_stage_idx = 0
    
    if use_curriculum:
        pass
    
    # 早停
    early_stopping = EarlyStopping(
        patience=train_cfg.get("early_stop_patience", 50),
        mode="max"
    )
    
    # 指标追踪
    metrics_tracker = MetricsTracker()
    best_acc = 0.0
    best_rts = float('inf')
    best_combined_score = -float('inf')
    
    # Rollout buffer
    buffer = RolloutBuffer()
    
    print(f"[Train] {max_episodes} episodes, {dataset.n_skus} SKUs")
    
    # 创建进度条
    pbar = tqdm(
        range(start_episode, max_episodes + 1), 
        desc="Training", 
        miniters=5, 
        mininterval=1.0,
        dynamic_ncols=True,
        leave=True,
    )
    
    for episode in pbar:
        episode_start_time = time.time()
        
        # 课程学习: 更新 reward 权重
        if use_curriculum:
            for i, stage in enumerate(curriculum_stages):
                stage_end = sum(s["episodes"] for s in curriculum_stages[:i+1])
                if episode <= stage_end:
                    if i != current_stage_idx:
                        current_stage_idx = i
                        pass  # stage transition
                        # 更新 reward 权重
                        if hasattr(reward_fn, 'rts_weight'):
                            reward_fn.rts_weight = stage.get("rts_weight", 1.0)
                        if hasattr(reward_fn, 'bind_weight'):
                            reward_fn.bind_weight = stage.get("bind_weight", 0.25)
                        if hasattr(reward_fn, 'overnight_weight'):
                            reward_fn.overnight_weight = stage.get("overnight_weight", 0.01)
                        if hasattr(reward_fn, 'safe_stock_weight'):
                            reward_fn.safe_stock_weight = stage.get("safe_stock_weight", 0.4)
                    break
        
        # 训练一个 episode（支持归一化）
        # 终止态奖励配置
        use_terminal_reward = train_cfg.get("use_terminal_reward", True)
        
        sku_sample_size = train_cfg.get("sku_sample_size", 0)
        
        cur_env = envs[episode % n_envs]
        episode_stats = train_episode(
            cur_env, agent, buffer,
            state_normalizer=state_normalizer,
            reward_normalizer=reward_normalizer,
            use_state_norm=use_state_norm,
            use_reward_norm=use_reward_norm,
            use_terminal_reward=use_terminal_reward,
            sku_sample_size=sku_sample_size,
        )
        
        # PPO 更新
        train_stats = {"policy_loss": 0, "value_loss": 0, "entropy": 0}
        if len(buffer) > 0 and episode % accumulate_episodes == 0:
            rollout_data = buffer.get()
            # 按SKU分段计算GAE（修复跨SKU GAE污染）
            rollout_data = compute_per_sku_gae(
                rollout_data, agent.gamma, agent.gae_lambda)
            train_stats = agent.update(rollout_data, writer=writer)
            buffer.reset()
        
        episode_time = time.time() - episode_start_time
        
        # 记录指标
        all_stats = {**episode_stats, **train_stats, "episode_time": episode_time}
        metrics_tracker.add(all_stats, episode)
        
        if writer:
            for key, value in all_stats.items():
                writer.add_scalar(f"train/{key}", value, episode)
        
        # 打印进度
        if episode % print_interval == 0:
            pbar.set_postfix_str(
                f"ACC={episode_stats['acc']:.1f}% RTS={episode_stats['rts_rate']:.1f}%"
            )
        
        # 评估（多数据集）
        if episode % eval_interval == 0:
            all_acc, all_rts = [], []
            for ei, ev in enumerate(envs):
                em = evaluate(ev, agent, state_normalizer, use_state_norm)
                all_acc.append(em["acc"])
                all_rts.append(em["rts_rate"])

            avg_acc = sum(all_acc) / len(all_acc)
            avg_rts = sum(all_rts) / len(all_rts)

            parts = "  ".join(f"{a:.1f}/{r:.1f}" for a, r in zip(all_acc, all_rts))
            tqdm.write(f"Avg ACC={avg_acc:.2f}% RTS={avg_rts:.2f}%  [{parts}]")

            if writer:
                writer.add_scalar("eval/acc", avg_acc, episode)
                writer.add_scalar("eval/rts_rate", avg_rts, episode)
                for ei in range(len(envs)):
                    writer.add_scalar(f"eval/acc_ds{ei}", all_acc[ei], episode)
                    writer.add_scalar(f"eval/rts_ds{ei}", all_rts[ei], episode)

            # 模型选择：用所有数据集的平均指标
            target_rts = train_cfg.get("target_rts", 2.4)

            if avg_rts <= target_rts:
                combined_score = avg_acc + 10.0
            else:
                combined_score = avg_acc - (avg_rts - target_rts) * 3.0

            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_acc = avg_acc
                best_rts = avg_rts

                extra_info = {
                    "episode": episode,
                    "best_acc": best_acc,
                    "best_rts": best_rts,
                    "best_combined_score": best_combined_score,
                }
                if use_state_norm:
                    extra_info["state_norm_state"] = state_normalizer.running_ms.save_state()
                if use_reward_norm:
                    extra_info["reward_norm_state"] = reward_normalizer.running_ms.save_state()

                agent.save(str(output_dir / "best_model.pth"), extra_info=extra_info)
                tqdm.write(f"  -> Best: ACC={best_acc:.2f}% RTS={best_rts:.2f}%")

            if early_stopping(combined_score):
                tqdm.write(f"Early stop at ep {episode}")
                pbar.close()
                break
        
        # 定期保存 checkpoint
        if episode % save_interval == 0:
            extra_info = {
                "episode": episode,
                "best_acc": best_acc,
                "best_rts": best_rts,
            }
            if use_state_norm:
                extra_info["state_norm_state"] = state_normalizer.running_ms.save_state()
            if use_reward_norm:
                extra_info["reward_norm_state"] = reward_normalizer.running_ms.save_state()
            
            agent.save(str(output_dir / f"model_ep{episode}.pth"), extra_info=extra_info)
    
    # 保存最终模型
    extra_info = {
        "episode": episode,
        "best_acc": best_acc,
        "best_rts": best_rts,
    }
    if use_state_norm:
        extra_info["state_norm_state"] = state_normalizer.running_ms.save_state()
    if use_reward_norm:
        extra_info["reward_norm_state"] = reward_normalizer.running_ms.save_state()
    
    agent.save(str(output_dir / "final_model.pth"), extra_info=extra_info)
    
    print("=" * 60)
    print(f"[Train] Training completed!")
    print(f"[Train] Best ACC: {best_acc:.2f}%, Best RTS: {best_rts:.2f}%")
    print(f"[Train] Models saved to: {output_dir}")
    
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
