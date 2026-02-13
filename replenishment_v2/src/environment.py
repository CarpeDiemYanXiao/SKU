"""
强化学习环境
Gym 风格的库存补货环境
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
from .dataset import ReplenishmentDataset
from .simulator import InventorySimulator, SKUState
from .reward import BaseReward, create_reward


class ReplenishmentEnv(gym.Env):
    """
    库存补货环境
    
    同时管理多个 SKU，每个 SKU 独立模拟
    
    支持两种动作模式:
    1. multiplier: 乘数模式，补货量 = pred_y * multiplier
    2. stock_days: 库存天数模式，补货量基于目标库存覆盖天数计算
    """
    
    def __init__(
        self,
        dataset: ReplenishmentDataset,
        reward_fn: BaseReward,
        config: dict,
    ):
        super().__init__()
        
        self.dataset = dataset
        self.reward_fn = reward_fn
        self.config = config
        
        # 环境配置
        env_cfg = config["env"]
        self.rts_days = env_cfg["rts_days"]
        self.max_leadtime = env_cfg["max_leadtime"]
        
        # 动作配置
        action_cfg = config["action"]
        self.action_type = action_cfg["type"]
        self.action_mode = action_cfg.get("action_mode", "multiplier")
        
        if self.action_mode == "stock_days":
            # 库存天数模式（支持float步长，如0.5）
            stock_days_range = action_cfg.get("stock_days_range", [0, 10])
            step = action_cfg.get("stock_days_step", 1)
            self.action_list = np.arange(
                stock_days_range[0],
                stock_days_range[1] + step * 0.5,  # 包含上界
                step
            ).round(2).tolist()
            self.n_actions = len(self.action_list)
            self.multiplier_range = None
            self.max_replenish_days = action_cfg.get("max_replenish_days", 7)
        else:
            # 乘数模式
            self.multiplier_range = action_cfg["multiplier_range"]
            if self.action_type == "discrete":
                step = action_cfg["multiplier_step"]
                self.action_list = np.arange(
                    self.multiplier_range[0],
                    self.multiplier_range[1] + step,
                    step
                ).round(2).tolist()
                self.n_actions = len(self.action_list)
            else:
                self.action_list = None
                self.n_actions = 1
        
        # 状态特征配置
        self.dynamic_features = env_cfg["state_features"]["dynamic"]
        self.static_features = env_cfg["state_features"]["static"]
        self.state_dim = len(self.dynamic_features) + len(self.static_features)
        
        # 模拟器
        self.simulator = InventorySimulator(rts_days=self.rts_days)
        
        # SKU 列表
        self.sku_ids = dataset.sku_ids
        self.n_skus = len(self.sku_ids)
        
        # 状态
        self.sku_states: Dict[str, SKUState] = {}
        self.day_idx_map: Dict[str, int] = {}
        self.done_map: Dict[str, bool] = {}
        
        # 定义空间
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        
        if self.action_type == "discrete":
            self.action_space = spaces.Discrete(self.n_actions)
        else:
            self.action_space = spaces.Box(
                low=self.multiplier_range[0],
                high=self.multiplier_range[1],
                shape=(1,),
                dtype=np.float32
            )
    
    def reset(self, seed: Optional[int] = None, sku_sample_size: int = 0) -> Dict[str, np.ndarray]:
        """
        重置环境
        
        Returns:
            state_map: {sku_id: state_vector}
        """
        super().reset(seed=seed)
        
        self.sku_states = {}
        self.day_idx_map = {}
        self.done_map = {}
        
        if sku_sample_size > 0 and sku_sample_size < len(self.sku_ids):
            active_ids = list(np.random.choice(self.sku_ids, sku_sample_size, replace=False))
        else:
            active_ids = self.sku_ids
        self._active_ids = active_ids
        
        state_map = {}
        
        for sku_id in active_ids:
            # 初始化模拟器状态
            initial_stock = self.dataset.initial_stock_map.get(sku_id, 0)
            self.sku_states[sku_id] = self.simulator.init_sku(sku_id, initial_stock)
            self.day_idx_map[sku_id] = 0
            self.done_map[sku_id] = False
            
            # 构建初始状态向量
            state_map[sku_id] = self._get_state_vector(sku_id)
        
        # 重置 reward (如果有状态)
        if hasattr(self.reward_fn, 'reset'):
            self.reward_fn.reset()
        
        return state_map
    
    def step(
        self, 
        action_map: Dict[str, Any]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, dict]]:
        """
        执行一步
        
        Args:
            action_map: {sku_id: action}
                - 离散: action 是索引
                - 连续: action 是 multiplier 值
                
        Returns:
            next_state_map, reward_map, done_map, info_map
        """
        next_state_map = {}
        reward_map = {}
        info_map = {}
        
        for sku_id in getattr(self, '_active_ids', self.sku_ids):
            if sku_id not in self.done_map or self.done_map[sku_id]:
                continue
            
            if sku_id not in action_map:
                continue
            
            day_idx = self.day_idx_map[sku_id]
            sku_state = self.sku_states[sku_id]
            
            # 获取当天数据
            leadtime = self.dataset.get_leadtime(sku_id, day_idx)
            actual_sales = self.dataset.get_sales(sku_id, day_idx)
            pred_y = self.dataset.get_pred_y(sku_id, day_idx)
            avg_sales = self.dataset.get_avg_qty_7d(sku_id, day_idx)
            
            # 获取预测序列 (用于库存天数模式)
            predicts = self.dataset.get_predicts(sku_id, day_idx)  # 6天预测
            
            # 解析动作并计算补货量
            action = action_map[sku_id]
            
            if self.action_mode == "stock_days":
                # ========== 库存天数模式 ==========
                target_stock_days = self.action_list[action]
                replenish_qty = self._compute_replenish_by_stock_days(
                    sku_state, target_stock_days, leadtime, pred_y, avg_sales, predicts,
                    sku_id=sku_id, day_idx=day_idx,
                )
                multiplier = target_stock_days  # 记录用
            else:
                # ========== 乘数模式 (简化版，严格控制RTS) ==========
                if self.action_type == "discrete":
                    multiplier = self.action_list[action]
                else:
                    multiplier = float(action)
                
                # 基础补货量计算
                if pred_y > 0:
                    base_qty = pred_y
                elif avg_sales > 0:
                    base_qty = avg_sales * 0.3  # 零预测时用历史的30%
                else:
                    base_qty = 0.0
                
                replenish_qty = base_qty * multiplier
                
                # ===== 额外的RTS控制约束 =====
                current_stock = sku_state.current_stock
                transit_total = sum(sku_state.transit_stock)
                soc_stock = current_stock + transit_total
                daily_demand = max(avg_sales, pred_y, 0.1)
                current_days = soc_stock / daily_demand
                
                # 如果库存已经够用，大幅减少补货
                if current_days > 4:
                    replenish_qty *= 0.1  # 只补10%
                elif current_days > 3:
                    replenish_qty *= 0.3  # 只补30%
                elif current_days > 2:
                    replenish_qty *= 0.6  # 只补60%
                
                replenish_qty = max(0, replenish_qty)
            
            # 模拟一天
            cf = self.dataset.get_demand_factor(sku_id, day_idx)
            adj_preds = [max(0, p * cf) for p in predicts]
            new_state, step_info = self.simulator.step(
                state=sku_state,
                replenish_qty=replenish_qty,
                actual_sales=actual_sales,
                leadtime=leadtime,
                avg_sales=avg_sales * cf,
                pred_y=pred_y * cf,
                predicts=adj_preds,
            )
            self.sku_states[sku_id] = new_state
            
            # 计算 reward
            avg_sales = self.dataset.get_avg_qty_7d(sku_id, day_idx)
            demand_freq = 0.5
            if self.dataset.demand_freq_map:
                demand_freq = self.dataset.demand_freq_map[sku_id][day_idx]
            
            # 获取order_ratio_7d用于奖励中头部/尾部SKU区分
            order_ratio_7d = self.dataset.get_feature_by_name(
                sku_id, day_idx, "order_ratio_l7d")
            if order_ratio_7d == 0.0 and "order_ratio_l7d" not in self.dataset.static_features:
                order_ratio_7d = 0.5  # 默认值
            
            state_info = {
                "avg_daily_sales": avg_sales,
                "pred_y": pred_y,
                "demand_freq": demand_freq,
                "order_ratio_7d": order_ratio_7d,
            }
            
            reward = self.reward_fn.compute(step_info, state_info)
            
            # 更新天数
            self.day_idx_map[sku_id] = day_idx + 1
            
            # 检查是否结束
            n_days = self.dataset.get_n_days(sku_id)
            if self.day_idx_map[sku_id] >= n_days:
                self.done_map[sku_id] = True
            
            # 构建下一状态
            if not self.done_map[sku_id]:
                next_state_map[sku_id] = self._get_state_vector(sku_id)
            else:
                # 终止状态用零向量
                next_state_map[sku_id] = np.zeros(self.state_dim, dtype=np.float32)
            
            reward_map[sku_id] = reward
            info_map[sku_id] = {
                "step_info": step_info,
                "multiplier": multiplier,
                "replenish_qty": replenish_qty,
                "reward_components": self.reward_fn.get_components(),
            }
        
        return next_state_map, reward_map, self.done_map.copy(), info_map
    
    def _compute_replenish_by_stock_days(
        self,
        sku_state: SKUState,
        target_stock_days: float,
        leadtime: int,
        pred_y: float,
        avg_sales: float,
        predicts: list,
        sku_id: str = None,
        day_idx: int = 0,
    ) -> float:
        """
        Base Stock 正则化补货计算（改进版）
        
        补货量 = max(目标到货库存 - 预计到货时库存, 0)
        
        关键改进：使用前向模拟的 lt_begin_stock 替代简单的 current_level，
        这让补货决策更准确——它考虑了leadtime期间的到货和消耗。
        """
        if target_stock_days <= 0:
            return 0.0
        
        # ========== 日均需求估计 ==========
        if pred_y > 0 and avg_sales > 0:
            daily_demand = 0.5 * pred_y + 0.5 * avg_sales
        elif pred_y > 0:
            daily_demand = pred_y * 0.7
        elif avg_sales > 0:
            daily_demand = avg_sales * 0.5
        else:
            return 0.0
        
        cf = self.dataset.get_demand_factor(sku_id, day_idx) if sku_id else 1.0
        daily_demand = daily_demand * cf

        adj_predicts = [max(0, p * cf) for p in predicts]

        # ========== 前向模拟到货时库存 ==========
        lt_begin_stock = self._compute_lt_begin_stock_helper(
            sku_state, adj_predicts, leadtime)
        
        # ========== Base Stock 计算 ==========
        target_stock_at_arrival = daily_demand * target_stock_days
        replenish_qty = max(0, target_stock_at_arrival - lt_begin_stock)
        
        # 单次补货上限
        max_days = getattr(self, "max_replenish_days", 7)
        replenish_qty = min(replenish_qty, daily_demand * max_days)
        
        # 取整避免碎片化在途库存
        return max(0, round(replenish_qty))
    
    def _compute_lt_begin_stock_helper(
        self,
        sku_state: SKUState,
        predicts: list,
        leadtime: int,
    ) -> float:
        """
        前向模拟：计算 leadtime 天后到货时的预计库存
        
        参考 rl_0113 基线中的 get_next_state 滚动模拟逻辑。
        这是基线中最关键的状态特征——让agent知道
        "当补货到达时，仓库里还有多少库存"。
        """
        rolling = sku_state.current_stock
        for i in range(min(leadtime, len(sku_state.transit_stock))):
            rolling += sku_state.transit_stock[i]
            pred = predicts[i] if i < len(predicts) else 0.0
            rolling = max(0, rolling - max(0, pred))
        return max(0, rolling)
    
    def _get_state_vector(self, sku_id: str) -> np.ndarray:
        """构建状态向量"""
        day_idx = self.day_idx_map[sku_id]
        sku_state = self.sku_states[sku_id]
        
        # 动态特征
        dynamic = []
        
        for feat_name in self.dynamic_features:
            if feat_name == "current_stock":
                dynamic.append(sku_state.current_stock)
            elif feat_name.startswith("transit_day_"):
                idx = int(feat_name.split("_")[-1]) - 1
                dynamic.append(sku_state.transit_stock[idx] if idx < len(sku_state.transit_stock) else 0.0)
            elif feat_name == "stock_health":
                avg_sales = self.dataset.get_avg_qty_7d(sku_id, day_idx)
                dynamic.append(self.simulator.get_stock_health(sku_state, avg_sales))
            elif feat_name == "days_of_stock":
                avg_sales = self.dataset.get_avg_qty_7d(sku_id, day_idx)
                dynamic.append(self.simulator.get_days_of_stock(sku_state, avg_sales))
            elif feat_name == "near_expire_ratio":
                dynamic.append(self.simulator.get_near_expire_ratio(sku_state))
            elif feat_name == "avg_stock_age":
                dynamic.append(self.simulator.get_avg_stock_age(sku_state))
            elif feat_name == "total_transit":
                dynamic.append(sum(sku_state.transit_stock))
            elif feat_name == "lt_begin_stock":
                _predicts = self.dataset.get_predicts(sku_id, day_idx)
                _leadtime = self.dataset.get_leadtime(sku_id, day_idx)
                _cf = self.dataset.get_demand_factor(sku_id, day_idx)
                _adj = [max(0, p * _cf) for p in _predicts]
                dynamic.append(self._compute_lt_begin_stock_helper(
                    sku_state, _adj, _leadtime))
            elif feat_name == "lt_demand":
                _predicts = self.dataset.get_predicts(sku_id, day_idx)
                _leadtime = self.dataset.get_leadtime(sku_id, day_idx)
                _cf = self.dataset.get_demand_factor(sku_id, day_idx)
                _lt_demand = sum(max(0, p * _cf) for p in _predicts[:_leadtime]) if len(_predicts) > 0 else 0.0
                dynamic.append(_lt_demand)
            else:
                dynamic.append(0.0)
        
        # 静态特征
        static = self.dataset.get_static_features(sku_id, day_idx)
        
        # 合并
        state = np.array(dynamic + static, dtype=np.float32)
        
        # 处理 NaN 和 Inf
        state = np.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return state
    
    def get_metrics(self) -> Dict[str, float]:
        """获取全局指标"""
        total_replenish = 0.0
        total_bind = 0.0
        total_sales = 0.0
        total_rts = 0.0
        total_overnight = 0.0
        total_stockout = 0.0
        
        for sku_id in getattr(self, '_active_ids', self.sku_ids):
            summary = self.simulator.get_summary(self.sku_states[sku_id])
            total_replenish += summary["total_replenish"]
            total_bind += summary["total_bind"]
            total_sales += summary["total_sales"]
            total_rts += summary["total_rts"]
            total_overnight += summary["total_overnight"]
            total_stockout += summary["total_stockout"]
        
        # 计算指标
        active = getattr(self, '_active_ids', self.sku_ids)
        if len(active) < len(self.sku_ids):
            market_sales = sum(
                sum(self.dataset.sales_map[sid]) for sid in active
            )
        else:
            market_sales = self.dataset.total_sales
        acc = total_sales / market_sales * 100 if market_sales > 0 else 0.0
        rts_rate = total_rts / total_replenish * 100 if total_replenish > 0 else 0.0
        
        return {
            "acc": acc,
            "rts_rate": rts_rate,
            "total_replenish": total_replenish,
            "total_bind": total_bind,
            "total_sales": total_sales,
            "total_rts": total_rts,
            "total_overnight": total_overnight,
            "total_stockout": total_stockout,
            "market_sales": market_sales,
        }


def create_env(dataset: ReplenishmentDataset, config: dict) -> ReplenishmentEnv:
    """创建环境"""
    reward_fn = create_reward(config)
    return ReplenishmentEnv(dataset, reward_fn, config)


def create_env_with_reward(
    dataset: ReplenishmentDataset,
    config: dict,
    reward_fn: BaseReward,
) -> ReplenishmentEnv:
    """创建环境（使用外部Reward实例）"""
    return ReplenishmentEnv(dataset, reward_fn, config)
