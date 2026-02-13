"""
工具函数
"""

import os
import yaml
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


def load_config(config_path: str) -> Dict:
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict, save_path: str):
    """保存配置到 YAML 文件"""
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_output_dir(config: Dict) -> Path:
    """创建输出目录"""
    task_name = config["task"]["name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = Path(config["logging"]["log_dir"]) / f"{task_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建 tensorboard 子目录
    tb_dir = output_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    save_config(config, str(output_dir / "config.yaml"))
    
    return output_dir


class RunningMeanStd:
    """
    在线计算均值和标准差
    用于状态归一化
    """
    
    def __init__(self, shape: tuple, epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x: np.ndarray):
        """更新统计量"""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        """Welford's online algorithm"""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.var = m2 / total_count
        self.count = total_count
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """归一化"""
        return (x - self.mean) / np.sqrt(self.var + 1e-8)
    
    def save(self, path: str):
        np.savez(path, mean=self.mean, var=self.var, count=self.count)
    
    def load(self, path: str):
        data = np.load(path)
        self.mean = data["mean"]
        self.var = data["var"]
        self.count = data["count"]


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 50, min_delta: float = 0.0, mode: str = "max"):
        """
        Args:
            patience: 忍耐次数
            min_delta: 最小改善量
            mode: "max" 或 "min"
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        检查是否应该早停
        
        Returns:
            True if should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "max":
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        
        return False


class MetricsTracker:
    """指标追踪器"""
    
    def __init__(self):
        self.history = {}
    
    def add(self, metrics: Dict[str, float], step: int):
        """添加指标"""
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append((step, value))
    
    def get_best(self, metric: str, mode: str = "max") -> tuple:
        """获取最佳值"""
        if metric not in self.history:
            return None, None
        
        values = self.history[metric]
        if mode == "max":
            best_idx = max(range(len(values)), key=lambda i: values[i][1])
        else:
            best_idx = min(range(len(values)), key=lambda i: values[i][1])
        
        return values[best_idx]
    
    def get_latest(self, metric: str) -> Optional[float]:
        """获取最新值"""
        if metric not in self.history or len(self.history[metric]) == 0:
            return None
        return self.history[metric][-1][1]
    
    def summary(self) -> str:
        """生成摘要"""
        lines = []
        for metric, values in self.history.items():
            if len(values) > 0:
                latest = values[-1][1]
                best = max(v[1] for v in values) if "loss" not in metric else min(v[1] for v in values)
                lines.append(f"{metric}: latest={latest:.4f}, best={best:.4f}")
        return "\n".join(lines)


def format_metrics(metrics: Dict[str, float]) -> str:
    """格式化指标为字符串"""
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return " | ".join(parts)


# ==============================================================================
# 高级归一化模块（参考 rl_0113/replenishment_abo）
# ==============================================================================

class OnlineRunningMeanStd:
    """
    在线计算均值和标准差（支持分布式同步）
    使用 Welford 算法实现增量更新
    """
    
    def __init__(self, shape: tuple, epsilon: float = 1e-8):
        self.n = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.S = np.ones(shape, dtype=np.float64)  # 方差的累积和
        self.std = np.sqrt(self.S)
        self.epsilon = epsilon
        self.sample_num = 0  # 用于分布式同步
    
    def set_sample_num(self, sample_num: int):
        """设置样本数（用于分布式同步）"""
        self.sample_num = sample_num
    
    def update(self, x: np.ndarray):
        """增量更新统计量"""
        x = np.array(x, dtype=np.float64)
        self.n += 1
        if self.n == 1:
            self.mean = x.copy()
            self.std = np.abs(x) + self.epsilon
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n) + self.epsilon
    
    def update_batch(self, batch: np.ndarray):
        """批量更新统计量"""
        batch_mean = np.mean(batch, axis=0, dtype=np.float64)
        batch_var = np.var(batch, axis=0, dtype=np.float64)
        batch_count = batch.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        """通过矩更新（Welford 并行算法）"""
        if self.n == 0:
            self.mean = batch_mean
            self.S = batch_var * batch_count
            self.n = batch_count
        else:
            delta = batch_mean - self.mean
            total_count = self.n + batch_count
            self.mean = self.mean + delta * batch_count / total_count
            m_a = self.S
            m_b = batch_var * batch_count
            self.S = m_a + m_b + np.square(delta) * self.n * batch_count / total_count
            self.n = total_count
        self.std = np.sqrt(self.S / self.n) + self.epsilon
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """归一化"""
        return (x - self.mean) / (self.std + self.epsilon)
    
    def sync_distributed(self, distributed: bool = False, world_size: int = 1):
        """分布式同步均值和标准差"""
        if not distributed or world_size <= 1:
            return
        
        try:
            import torch.distributed as dist
            import torch
            
            # 将统计量收集到所有进程
            norm_dict = {
                "mean": torch.tensor(self.mean),
                "std": torch.tensor(self.std),
                "num": self.sample_num
            }
            norm_list = [None] * world_size
            dist.all_gather_object(norm_list, norm_dict)
            
            # 过滤有效数据并聚合
            norm_list = [i for i in norm_list if i is not None and i["num"] > 0]
            if len(norm_list) == 0:
                return
            
            nums_all = sum([i["num"] for i in norm_list])
            self.mean = np.sum([np.array(i["mean"] * i["num"]) for i in norm_list], axis=0) / nums_all
            
            # 计算合并方差
            var_weighted = np.sum(
                [(np.array(i["mean"])**2 + np.array(i["std"])**2) * i["num"] for i in norm_list],
                axis=0
            ) / nums_all - self.mean**2
            self.std = np.sqrt(np.maximum(var_weighted, 0)) + self.epsilon
            
        except ImportError:
            pass  # 非分布式环境
    
    def save_state(self) -> Dict:
        """保存状态"""
        return {
            "n": self.n,
            "mean": self.mean.tolist(),
            "S": self.S.tolist(),
            "std": self.std.tolist(),
        }
    
    def load_state(self, state: Dict):
        """加载状态"""
        self.n = state["n"]
        self.mean = np.array(state["mean"])
        self.S = np.array(state["S"])
        self.std = np.array(state["std"])


class StateNormalizer:
    """
    状态归一化器
    支持在线更新和分布式同步
    """
    
    def __init__(self, shape: tuple, clip: float = 10.0, update: bool = True):
        self.running_ms = OnlineRunningMeanStd(shape)
        self.clip = clip
        self.update_enabled = update
    
    def __call__(self, x: np.ndarray, update: bool = None) -> np.ndarray:
        """
        归一化状态
        
        Args:
            x: 输入状态
            update: 是否更新统计量（None 时使用默认设置）
        """
        if update is None:
            update = self.update_enabled
        
        if update:
            self.running_ms.update(x)
        
        normalized = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)
        
        if self.clip > 0:
            normalized = np.clip(normalized, -self.clip, self.clip)
        
        return normalized.astype(np.float32)
    
    def set_sample_num(self, n: int):
        self.running_ms.set_sample_num(n)
    
    def sync(self, distributed: bool = False, world_size: int = 1):
        self.running_ms.sync_distributed(distributed, world_size)


class RewardNormalizer:
    """
    奖励归一化器
    只除以标准差，不减均值（保持奖励符号）
    """
    
    def __init__(self, clip: float = 10.0, gamma: float = 0.99):
        self.running_ms = OnlineRunningMeanStd(shape=(1,))
        self.clip = clip
        self.gamma = gamma
        self.R = 0.0  # 折扣回报累积
    
    def __call__(self, reward: float, update: bool = True) -> float:
        """归一化奖励"""
        self.R = self.gamma * self.R + reward
        
        if update:
            self.running_ms.update(np.array([self.R]))
        
        # 只除以标准差
        normalized = reward / (self.running_ms.std[0] + 1e-8)
        
        if self.clip > 0:
            normalized = np.clip(normalized, -self.clip, self.clip)
        
        return float(normalized)
    
    def reset(self):
        """Episode 结束时重置累积回报"""
        self.R = 0.0
    
    def set_sample_num(self, n: int):
        self.running_ms.set_sample_num(n)
    
    def sync(self, distributed: bool = False, world_size: int = 1):
        self.running_ms.sync_distributed(distributed, world_size)


class AdvantageScaler:
    """
    优势值标准化和裁剪
    """
    
    @staticmethod
    def normalize(advantages: np.ndarray, clip: float = 5.0) -> np.ndarray:
        """标准化并裁剪优势值"""
        mean = advantages.mean()
        std = advantages.std() + 1e-8
        normalized = (advantages - mean) / std
        if clip > 0:
            normalized = np.clip(normalized, -clip, clip)
        return normalized
    
    @staticmethod
    def normalize_torch(advantages, clip: float = 5.0):
        """PyTorch 版本"""
        import torch
        mean = advantages.mean()
        std = advantages.std() + 1e-8
        normalized = (advantages - mean) / std
        if clip > 0:
            normalized = torch.clamp(normalized, -clip, clip)
        return normalized
