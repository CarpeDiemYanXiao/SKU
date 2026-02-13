"""
库存模拟器
精确模拟库存的滚动更新、到货、销售、RTS等过程
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class SKUState:
    """单个 SKU 的状态"""
    sku_id: str
    day_idx: int = 0
    
    # 库存状态
    current_stock: float = 0.0          # 当前可用库存
    transit_stock: List[float] = field(default_factory=lambda: [0.0] * 5)  # 在途库存 (未来5天到货)
    
    # 历史记录 (用于计算RTS)
    stock_age_queue: List[Tuple[int, float]] = field(default_factory=list)  # [(入库日, 数量), ...]
    
    # 累计指标
    total_replenish: float = 0.0        # 总补货量
    total_bind: float = 0.0             # 总售出量 (到货日)
    total_sales: float = 0.0            # 总销量
    total_rts: float = 0.0              # 总退货量
    total_overnight: float = 0.0        # 总过夜库存
    total_stockout: float = 0.0         # 总缺货量
    
    # 配置
    rts_days: int = 14                  # RTS天数阈值


class InventorySimulator:
    """
    库存模拟器
    
    核心流程 (每天):
    1. 到货: 在途库存到达
    2. 销售: 按FIFO消耗库存
    3. RTS: 超过14天的库存退货
    4. 补货: 根据action决定补货量，加入在途
    5. 过夜: 记录期末库存
    """
    
    def __init__(self, rts_days: int = 14):
        self.rts_days = rts_days
    
    def init_sku(self, sku_id: str, initial_stock: float) -> SKUState:
        """初始化 SKU 状态"""
        state = SKUState(
            sku_id=sku_id,
            current_stock=initial_stock,
            transit_stock=[0.0] * 5,
            rts_days=self.rts_days,
        )
        # 初始库存的入库日设为 -rts_days+1，避免第一天就RTS
        if initial_stock > 0:
            state.stock_age_queue.append((-self.rts_days + 1, initial_stock))
        return state
    
    def step(
        self,
        state: SKUState,
        replenish_qty: float,
        actual_sales: float,
        leadtime: int,
        avg_sales: float,
        pred_y: float,
        predicts: List[float],
    ) -> Tuple[SKUState, Dict[str, float]]:
        """
        执行一天的模拟
        
        Args:
            state: 当前SKU状态
            replenish_qty: 补货量
            actual_sales: 实际销量
            leadtime: 当前leadtime
            
        Returns:
            new_state: 更新后的状态
            info: 当天的各项指标
        """
        day = state.day_idx
        info = {}
        
        # ========== 1. 到货 ==========
        arrived_qty = state.transit_stock[0]
        state.transit_stock = state.transit_stock[1:] + [0.0]
        
        if arrived_qty > 0:
            state.current_stock += arrived_qty
            state.stock_age_queue.append((day, arrived_qty))
        
        info["arrived_qty"] = arrived_qty
        
        # ========== 2. 销售 (FIFO) ==========
        demand = actual_sales
        sold = 0.0
        bind_sold = 0.0  # 到货当天售出的量
        
        remaining_demand = demand
        new_queue = []
        
        for (entry_day, qty) in state.stock_age_queue:
            if remaining_demand <= 0:
                new_queue.append((entry_day, qty))
            elif qty <= remaining_demand:
                # 全部卖掉
                sold += qty
                if entry_day == day:  # 今天到的货今天卖掉
                    bind_sold += qty
                remaining_demand -= qty
            else:
                # 部分卖掉
                sold += remaining_demand
                if entry_day == day:
                    bind_sold += remaining_demand
                new_queue.append((entry_day, qty - remaining_demand))
                remaining_demand = 0
        
        state.stock_age_queue = new_queue
        state.current_stock -= sold
        state.total_sales += sold
        state.total_bind += bind_sold
        
        # 缺货量
        stockout = max(0, demand - sold)
        state.total_stockout += stockout
        
        info["sold"] = sold
        info["bind"] = bind_sold
        info["stockout"] = stockout
        
        # ========== 3. RTS ==========
        rts = 0.0
        new_queue = []
        for (entry_day, qty) in state.stock_age_queue:
            age = day - entry_day
            if age >= self.rts_days:
                rts += qty
            else:
                new_queue.append((entry_day, qty))
        
        state.stock_age_queue = new_queue
        state.current_stock -= rts
        state.total_rts += rts
        
        info["rts"] = rts
        
        # ========== 4. 补货 ==========
        replenish_qty = max(0, replenish_qty)
        if leadtime >= 1 and leadtime <= 5:
            state.transit_stock[leadtime - 1] += replenish_qty
        state.total_replenish += replenish_qty
        
        info["replenish"] = replenish_qty
        
        # ========== 5. 过夜 ==========
        overnight = state.current_stock
        state.total_overnight += overnight
        
        info["overnight"] = overnight
        info["end_stock"] = state.current_stock
        
        # ========== 6. 预估RTS (关键优化!) ==========
        # 参考rl_0113: 预估到货后未来14天卖不掉的量
        # 这让模型能在补货时就预见到RTS风险
        info["estimate_rts"] = self._estimate_future_rts(
            state, replenish_qty, leadtime, avg_sales, pred_y, predicts
        )
        
        # 更新天数
        state.day_idx += 1
        
        return state, info
    
    def _estimate_future_rts(
        self, 
        state: SKUState, 
        replenish_qty: float, 
        leadtime: int,
        avg_sales: float,
        pred_y: float,
        predicts: List[float],
    ) -> float:
        """
        预估当前补货可能导致的RTS量
        
        两个来源:
        1. 现有库存中即将过期的部分
        2. 补货导致的过量库存（供过于求）
        
        注意: 调用时 replenish_qty 已经加入了 transit_stock，
        所以滚动模拟只需用 current_stock + transit_stock，无需再加 replenish_qty。
        """
        # 1. 现有库存中快过期部分
        near_expire = 0.0
        current_day = state.day_idx
        for (entry_day, qty) in state.stock_age_queue:
            remaining = self.rts_days - (current_day - entry_day)
            if 0 < remaining <= 5:
                near_expire += qty * (1.0 - remaining / 5.0)
        
        if replenish_qty <= 0:
            return near_expire
        
        # 2. 滚动模拟估算过量风险（transit_stock 已包含本次补货）
        daily_demand = max(0.5 * max(pred_y, 0) + 0.5 * max(avg_sales, 0), 0.1)
        rolling = state.current_stock
        
        for d in range(self.rts_days):
            # 在途到货
            if d < len(state.transit_stock):
                rolling += state.transit_stock[d]
            # 每日消耗
            pred = predicts[d] if d < len(predicts) else daily_demand
            rolling = max(0, rolling - max(pred, 0))
        
        # rolling 为 rts_days 后的剩余库存，即潜在RTS
        return near_expire + max(0, rolling) * 0.3
    
    def get_transit_stock(self, state: SKUState) -> List[float]:
        """获取在途库存列表"""
        return state.transit_stock.copy()
    
    def get_stock_health(self, state: SKUState, avg_daily_sales: float) -> float:
        """
        计算库存健康度
        健康度 = 1 - (快过期库存占比)
        """
        if state.current_stock <= 0:
            return 1.0
        
        total = 0.0
        near_expire = 0.0
        
        for (entry_day, qty) in state.stock_age_queue:
            age = state.day_idx - entry_day
            total += qty
            if age >= self.rts_days - 3:  # 还剩3天就过期
                near_expire += qty
        
        if total == 0:
            return 1.0
        
        return 1.0 - (near_expire / total)
    
    def get_near_expire_ratio(self, state: SKUState) -> float:
        """
        获取快过期库存比例 (7天内过期)
        """
        if state.current_stock <= 0:
            return 0.0
        
        total = 0.0
        near_expire = 0.0
        
        for (entry_day, qty) in state.stock_age_queue:
            age = state.day_idx - entry_day
            total += qty
            if age >= self.rts_days - 7:  # 还剩7天过期
                near_expire += qty
        
        if total == 0:
            return 0.0
        
        return near_expire / total
    
    def get_avg_stock_age(self, state: SKUState) -> float:
        """
        获取库存平均年龄 (归一化到0-1)
        """
        if state.current_stock <= 0:
            return 0.0
        
        total_qty = 0.0
        weighted_age = 0.0
        
        for (entry_day, qty) in state.stock_age_queue:
            age = state.day_idx - entry_day
            total_qty += qty
            weighted_age += age * qty
        
        if total_qty == 0:
            return 0.0
        
        avg_age = weighted_age / total_qty
        return min(1.0, avg_age / self.rts_days)  # 归一化
    
    def get_days_of_stock(self, state: SKUState, avg_daily_sales: float) -> float:
        """
        计算可售天数
        可售天数 = 当前库存 / 平均日销量
        """
        if avg_daily_sales <= 0:
            return 14.0 if state.current_stock > 0 else 0.0
        
        return min(14.0, state.current_stock / max(0.1, avg_daily_sales))
    
    def get_summary(self, state: SKUState) -> Dict[str, float]:
        """获取累计指标汇总"""
        return {
            "total_replenish": state.total_replenish,
            "total_bind": state.total_bind,
            "total_sales": state.total_sales,
            "total_rts": state.total_rts,
            "total_overnight": state.total_overnight,
            "total_stockout": state.total_stockout,
        }
