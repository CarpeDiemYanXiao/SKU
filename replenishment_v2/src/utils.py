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
