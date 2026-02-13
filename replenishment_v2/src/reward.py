"""
Reward 设计模块
实现多种 reward shaping 策略
"""

import numpy as np
from typing import Dict, Optional
from abc import ABC, abstractmethod


class BaseReward(ABC):
    """Reward 基类"""
    
    @abstractmethod
    def compute(self, info: Dict, state_info: Dict) -> float:
        """计算单步 reward"""
        pass
    
    @abstractmethod
    def get_components(self) -> Dict[str, float]:
        """获取各分量值 (用于 debug)"""
        pass


class DefaultReward(BaseReward):
    """
    默认 Reward 设计
    
    reward = bind_weight * bind 
           - rts_weight * rts 
           - overnight_weight * overnight
           - stockout_weight * stockout
    """
    
    def __init__(
        self,
        bind_weight: float = 0.25,
        rts_weight: float = 1.0,
        overnight_weight: float = 0.02,
        stockout_weight: float = 0.15,
        safe_stock_weight: float = 0.08,
        normalize: bool = True,
        clip_range: tuple = (-10, 10),
    ):
        self.bind_weight = bind_weight
        self.rts_weight = rts_weight
        self.overnight_weight = overnight_weight
        self.stockout_weight = stockout_weight
        self.safe_stock_weight = safe_stock_weight
        self.normalize = normalize
        self.clip_range = clip_range
        
        self._components = {}
    
    def compute(self, info: Dict, state_info: Dict) -> float:
        """
        计算 reward
        
        Args:
            info: 模拟器返回的当天指标
                - bind: 到货日售出量
                - rts: 退货量
                - overnight: 过夜库存
                - stockout: 缺货量
            state_info: 状态信息
                - avg_daily_sales: 平均日销量
                - pred_y: 预测销量
        """
        bind = info.get("bind", 0)
        rts = info.get("rts", 0)
        overnight = info.get("overnight", 0)
        stockout = info.get("stockout", 0)
        
        avg_sales = state_info.get("avg_daily_sales", 1.0)
        pred_y = state_info.get("pred_y", 1.0)
        
        # 归一化各指标 (相对于预测/平均销量)
        norm_factor = max(avg_sales, pred_y, 1.0)
        
        if self.normalize:
            bind_norm = bind / norm_factor
            rts_norm = rts / norm_factor
            overnight_norm = overnight / (norm_factor * 7)  # 7天库存作为基准
            stockout_norm = stockout / norm_factor
        else:
            bind_norm = bind
            rts_norm = rts
            overnight_norm = overnight
            stockout_norm = stockout
        
        # 安全库存奖励 (库存在合理范围内给正向奖励)
        end_stock = info.get("end_stock", 0)
        safe_stock_target = avg_sales * 3  # 3天安全库存
        safe_stock_reward = 0.0
        if 0 < end_stock <= safe_stock_target * 2:
            # 库存在0-2倍安全库存之间，给奖励
            safe_stock_reward = 1.0 - abs(end_stock - safe_stock_target) / (safe_stock_target * 2)
        
        # 计算总 reward
        reward = (
            self.bind_weight * bind_norm
            - self.rts_weight * rts_norm
            - self.overnight_weight * overnight_norm
            - self.stockout_weight * stockout_norm
            + self.safe_stock_weight * safe_stock_reward
        )
        
        # Clip
        reward = np.clip(reward, self.clip_range[0], self.clip_range[1])
        
        # 保存分量
        self._components = {
            "bind": self.bind_weight * bind_norm,
            "rts": -self.rts_weight * rts_norm,
            "overnight": -self.overnight_weight * overnight_norm,
            "stockout": -self.stockout_weight * stockout_norm,
            "safe_stock": self.safe_stock_weight * safe_stock_reward,
            "total": reward,
        }
        
        return reward
    
    def get_components(self) -> Dict[str, float]:
        return self._components.copy()


class HierarchicalReward(BaseReward):
    """
    分层 Reward 设计
    
    优先保证 RTS 约束，然后优化 ACC
    reward = base_reward + constraint_bonus + near_expire_penalty
    """
    
    def __init__(
        self,
        bind_weight: float = 0.25,
        rts_weight: float = 1.0,
        overnight_weight: float = 0.02,
        stockout_weight: float = 0.15,
        near_expire_weight: float = 2.0,  # 新增：快过期惩罚
        max_rts_rate: float = 0.024,
        rts_penalty_scale: float = 5.0,
        normalize: bool = True,
        clip_range: tuple = (-10, 10),
    ):
        self.bind_weight = bind_weight
        self.rts_weight = rts_weight
        self.overnight_weight = overnight_weight
        self.stockout_weight = stockout_weight
        self.near_expire_weight = near_expire_weight
        self.max_rts_rate = max_rts_rate
        self.rts_penalty_scale = rts_penalty_scale
        self.normalize = normalize
        self.clip_range = clip_range
        
        self._components = {}
        
        # 滚动统计
        self._total_rts = 0.0
        self._total_replenish = 0.0
    
    def reset(self):
        """重置统计"""
        self._total_rts = 0.0
        self._total_replenish = 0.0
    
    def compute(self, info: Dict, state_info: Dict) -> float:
        """计算分层 reward"""
        bind = info.get("bind", 0)
        rts = info.get("rts", 0)
        overnight = info.get("overnight", 0)
        stockout = info.get("stockout", 0)
        replenish = info.get("replenish", 0)
        sold = info.get("sold", 0)  # 当天实际销售
        end_stock = info.get("end_stock", 0)  # 期末库存
        estimate_rts = info.get("estimate_rts", 0)  # 预估RTS (关键!)
        
        # 更新滚动统计
        self._total_rts += rts
        self._total_replenish += replenish
        
        avg_sales = state_info.get("avg_daily_sales", 1.0)
        pred_y = state_info.get("pred_y", 1.0)
        norm_factor = max(avg_sales, pred_y, 1.0)
        
        if self.normalize:
            bind_norm = bind / norm_factor
            rts_norm = rts / norm_factor
            overnight_norm = overnight / (norm_factor * 7)
            stockout_norm = stockout / norm_factor
            estimate_rts_norm = estimate_rts / norm_factor
        else:
            bind_norm = bind
            rts_norm = rts
            overnight_norm = overnight
            stockout_norm = stockout
            estimate_rts_norm = estimate_rts
        
        # ========== 核心Reward设计 (参考rl_0113) ==========
        # 1. 绑定奖励：到货即售
        bind_reward = self.bind_weight * bind_norm
        
        # 2. RTS惩罚：实际发生的RTS + 预估RTS (关键!)
        #    预估RTS让模型在补货决策时就能预见风险
        rts_penalty = self.rts_weight * (rts_norm + estimate_rts_norm * 0.5)
        
        # 3. 过夜惩罚
        overnight_penalty = self.overnight_weight * overnight_norm
        
        # 4. 缺货惩罚 (适度，允许保守补货)
        stockout_penalty = self.stockout_weight * stockout_norm
        
        # 基础reward
        base_reward = bind_reward - rts_penalty - overnight_penalty - stockout_penalty
        
        # ========== 库存积压预警 ==========
        stock_warning = 0.0
        days_of_stock = end_stock / max(avg_sales, 0.1)
        
        if days_of_stock > 5:
            excess_days = days_of_stock - 5
            stock_warning = -0.3 * (excess_days ** 1.5) / 10
            stock_warning = max(stock_warning, -3.0)
        
        # ========== 约束惩罚 ==========
        constraint_penalty = 0.0
        if self._total_replenish > 0:
            current_rts_rate = self._total_rts / self._total_replenish
            if current_rts_rate > self.max_rts_rate:
                excess = current_rts_rate - self.max_rts_rate
                constraint_penalty = -self.rts_penalty_scale * excess
        
        reward = base_reward + stock_warning + constraint_penalty
        reward = np.clip(reward, self.clip_range[0], self.clip_range[1])
        
        self._components = {
            "bind": bind_reward,
            "rts": -self.rts_weight * rts_norm,
            "estimate_rts": -self.rts_weight * estimate_rts_norm * 0.5,
            "overnight": -overnight_penalty,
            "stockout": -stockout_penalty,
            "stock_warning": stock_warning,
            "constraint": constraint_penalty,
            "total": reward,
        }
        
        return reward
    
    def get_components(self) -> Dict[str, float]:
        return self._components.copy()


class AdaptiveReward(BaseReward):
    """
    自适应 Reward
    
    根据 SKU 类型动态调整权重:
    - 高频SKU: 更关注 ACC
    - 低频SKU: 更关注 RTS
    """
    
    def __init__(
        self,
        base_bind_weight: float = 0.25,
        base_rts_weight: float = 1.0,
        base_overnight_weight: float = 0.02,
        base_stockout_weight: float = 0.15,
        high_freq_threshold: float = 0.7,
        low_freq_threshold: float = 0.3,
        normalize: bool = True,
        clip_range: tuple = (-10, 10),
    ):
        self.base_bind_weight = base_bind_weight
        self.base_rts_weight = base_rts_weight
        self.base_overnight_weight = base_overnight_weight
        self.base_stockout_weight = base_stockout_weight
        self.high_freq_threshold = high_freq_threshold
        self.low_freq_threshold = low_freq_threshold
        self.normalize = normalize
        self.clip_range = clip_range
        
        self._components = {}
    
    def compute(self, info: Dict, state_info: Dict) -> float:
        """计算自适应 reward"""
        bind = info.get("bind", 0)
        rts = info.get("rts", 0)
        overnight = info.get("overnight", 0)
        stockout = info.get("stockout", 0)
        
        demand_freq = state_info.get("demand_freq", 0.5)
        avg_sales = state_info.get("avg_daily_sales", 1.0)
        pred_y = state_info.get("pred_y", 1.0)
        norm_factor = max(avg_sales, pred_y, 1.0)
        
        # 根据需求频率调整权重
        if demand_freq >= self.high_freq_threshold:
            # 高频: 更关注ACC，降低RTS惩罚
            bind_weight = self.base_bind_weight * 1.3
            rts_weight = self.base_rts_weight * 0.8
            stockout_weight = self.base_stockout_weight * 1.2
        elif demand_freq <= self.low_freq_threshold:
            # 低频: 更关注RTS，降低缺货惩罚
            bind_weight = self.base_bind_weight * 0.8
            rts_weight = self.base_rts_weight * 1.3
            stockout_weight = self.base_stockout_weight * 0.7
        else:
            # 中频: 使用基础权重
            bind_weight = self.base_bind_weight
            rts_weight = self.base_rts_weight
            stockout_weight = self.base_stockout_weight
        
        if self.normalize:
            bind_norm = bind / norm_factor
            rts_norm = rts / norm_factor
            overnight_norm = overnight / (norm_factor * 7)
            stockout_norm = stockout / norm_factor
        else:
            bind_norm = bind
            rts_norm = rts
            overnight_norm = overnight
            stockout_norm = stockout
        
        reward = (
            bind_weight * bind_norm
            - rts_weight * rts_norm
            - self.base_overnight_weight * overnight_norm
            - stockout_weight * stockout_norm
        )
        
        reward = np.clip(reward, self.clip_range[0], self.clip_range[1])
        
        self._components = {
            "bind": bind_weight * bind_norm,
            "rts": -rts_weight * rts_norm,
            "overnight": -self.base_overnight_weight * overnight_norm,
            "stockout": -stockout_weight * stockout_norm,
            "demand_freq": demand_freq,
            "total": reward,
        }
        
        return reward
    
    def get_components(self) -> Dict[str, float]:
        return self._components.copy()


class BalancedReward(BaseReward):
    """
    Balanced Reward — 参考 rl_0113 基线验证过的奖励设计
    
    核心思想（已验证有效）：
    1. bind奖励：到货即售的量（正向激励适度补货）
    2. 预估RTS惩罚：前瞻性RTS估计（让agent在补货时就考虑退货风险）
    3. 过夜惩罚：持有成本
    4. 安全库存缺口惩罚：头部SKU库存不足时惩罚（推动ACC提升的关键）
    5. 缺货惩罚：头部SKU缺货时惩罚
    
    通过区分头部/尾部SKU精准投放补货：
    - 头部SKU（高频需求 order_ratio_7d≥阈值）：加强补货保ACC
    - 尾部SKU（低频需求）：保守补货控RTS
    """
    
    def __init__(
        self,
        bind_weight: float = 0.18,
        rts_weight: float = 0.8,
        overnight_weight: float = 0.01,
        safe_stock_weight: float = 0.4,
        stockout_weight: float = 0.3,
        safe_stock_standard: float = 0.6,
        head_sku_threshold: float = 0.8,
        normalize: bool = False,
        clip_range: tuple = (-10, 10),
    ):
        self.bind_weight = bind_weight
        self.rts_weight = rts_weight
        self.overnight_weight = overnight_weight
        self.safe_stock_weight = safe_stock_weight
        self.stockout_weight = stockout_weight
        self.safe_stock_standard = safe_stock_standard
        self.head_sku_threshold = head_sku_threshold
        self.normalize = normalize
        self.clip_range = clip_range
        
        self._components = {}
        self._total_rts = 0.0
        self._total_replenish = 0.0
        self._total_sold = 0.0
        self._total_stockout = 0.0
    
    def reset(self):
        self._total_rts = 0.0
        self._total_replenish = 0.0
        self._total_sold = 0.0
        self._total_stockout = 0.0
    
    def compute(self, info: Dict, state_info: Dict) -> float:
        """参考 rl_0113 基线的奖励计算（已验证有效）"""
        bind = info.get("bind", 0)
        rts = info.get("rts", 0)
        estimate_rts = info.get("estimate_rts", 0)
        overnight = info.get("overnight", 0)
        stockout = info.get("stockout", 0)
        sold = info.get("sold", 0)
        end_stock = info.get("end_stock", 0)
        replenish = info.get("replenish", 0)
        
        avg_sales = state_info.get("avg_daily_sales", 1.0)
        order_ratio_7d = state_info.get("order_ratio_7d", 0.5)
        
        # 更新统计
        self._total_rts += rts
        self._total_replenish += replenish
        self._total_sold += sold
        self._total_stockout += stockout
        
        # 归一化因子（按SKU日均需求，使不同SKU的奖励量纲一致）
        norm = max(avg_sales, 0.1)
        
        # ========== 核心奖励 (参考基线 with_safe_stock) ==========
        
        # 1. Bind奖励：到货即售（正向激励补货）
        bind_reward = self.bind_weight * (bind / norm)
        
        # 2. 预估RTS惩罚（关键！前瞻性信号，非事后惩罚）
        rts_penalty = self.rts_weight * (estimate_rts / norm)
        
        # 3. 过夜持有成本
        overnight_penalty = self.overnight_weight * min(overnight / norm, 14.0)
        
        # 4&5. 头部SKU专属：安全库存缺口 + 缺货惩罚（推动ACC的关键）
        safe_stock_penalty = 0.0
        stockout_penalty = 0.0
        if order_ratio_7d >= self.head_sku_threshold:
            # safe_stock_standard 表示安全库存天数，stock_gap 也归一化为天数
            stock_gap = max(0, self.safe_stock_standard - end_stock / norm)
            safe_stock_penalty = self.safe_stock_weight * stock_gap
            stockout_penalty = self.stockout_weight * (stockout / norm)
        
        reward = bind_reward - rts_penalty - overnight_penalty - safe_stock_penalty - stockout_penalty
        reward = np.clip(reward, self.clip_range[0], self.clip_range[1])
        
        self._components = {
            "bind": bind_reward,
            "rts_penalty": -rts_penalty,
            "overnight": -overnight_penalty,
            "safe_stock": -safe_stock_penalty,
            "stockout": -stockout_penalty,
            "is_head": float(order_ratio_7d >= self.head_sku_threshold),
            "total": reward,
        }
        
        return reward
    
    def compute_terminal_reward(self) -> float:
        """终止态奖励（可选，默认不启用，依赖即时逐步奖励信号）"""
        return 0.0
    
    def get_components(self) -> Dict[str, float]:
        return self._components.copy()


def create_reward(config: dict) -> BaseReward:
    """根据配置创建 Reward 实例"""
    reward_cfg = config["reward"]
    reward_type = reward_cfg.get("type", "default")
    weights = reward_cfg.get("weights", {})
    constraints = reward_cfg.get("constraints", {})
    targets = reward_cfg.get("targets", {})
    
    if reward_type == "default":
        return DefaultReward(
            bind_weight=weights.get("bind", 0.25),
            rts_weight=weights.get("rts", 1.0),
            overnight_weight=weights.get("overnight", 0.02),
            stockout_weight=weights.get("stockout", 0.15),
            safe_stock_weight=weights.get("safe_stock", 0.08),
            normalize=reward_cfg.get("normalize", True),
            clip_range=tuple(reward_cfg.get("clip_range", [-10, 10])),
        )
    
    elif reward_type == "hierarchical":
        return HierarchicalReward(
            bind_weight=weights.get("bind", 0.25),
            rts_weight=weights.get("rts", 1.0),
            overnight_weight=weights.get("overnight", 0.02),
            stockout_weight=weights.get("stockout", 0.15),
            near_expire_weight=weights.get("near_expire", 2.0),
            max_rts_rate=constraints.get("max_rts_rate", 0.024),
            rts_penalty_scale=constraints.get("rts_penalty_scale", 5.0),
            normalize=reward_cfg.get("normalize", True),
            clip_range=tuple(reward_cfg.get("clip_range", [-10, 10])),
        )
    
    elif reward_type == "balanced":
        return BalancedReward(
            bind_weight=weights.get("bind", 0.18),
            rts_weight=weights.get("rts", 0.8),
            overnight_weight=weights.get("overnight", 0.01),
            safe_stock_weight=weights.get("safe_stock", 0.4),
            stockout_weight=weights.get("stockout", 0.3),
            safe_stock_standard=targets.get("safe_stock_standard", 0.6),
            head_sku_threshold=targets.get("head_sku_threshold", 0.8),
            normalize=reward_cfg.get("normalize", False),
            clip_range=tuple(reward_cfg.get("clip_range", [-10, 10])),
        )
    
    elif reward_type == "simple":
        return SimpleReward(
            sales_weight=weights.get("sales", 1.0),
            rts_weight=weights.get("rts", 50.0),
            stockout_weight=weights.get("stockout", 0.3),
            max_rts_rate=constraints.get("max_rts_rate", 0.024),
            rts_penalty_scale=constraints.get("rts_penalty_scale", 100.0),
            normalize=reward_cfg.get("normalize", True),
            clip_range=tuple(reward_cfg.get("clip_range", [-20, 20])),
        )
    
    elif reward_type == "adaptive":
        return AdaptiveReward(
            base_bind_weight=weights.get("bind", 0.25),
            base_rts_weight=weights.get("rts", 1.0),
            base_overnight_weight=weights.get("overnight", 0.02),
            base_stockout_weight=weights.get("stockout", 0.15),
            normalize=reward_cfg.get("normalize", True),
            clip_range=tuple(reward_cfg.get("clip_range", [-10, 10])),
        )
    
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")
