from typing import Callable, Optional
from .sku import SKU
from .strategy import StrategyB
import pandas as pd


class Roller:
    """
    Roller 类负责管理 SKU 的补货过程。
    它处理基于给定策略的日常操作，如销售、补货和库存回退。他模拟一段时间内的补货过程, 记录的是最终结果.
    Attributes:
        skus (list[SKU]): SKU 对象的列表。
        rts_day (int): 库存回退的天数。
        strategy (StrategyB): 用于补货计算的策略对象。
        rts_qty (int): 回退的库存数量。
        bind_qty (int): 绑定的库存数量。
        abo_qty (int): 补货的库存数量。
        total_sales (int): 总销售数量。
    Methods:
        reset(begin_stock_dict: dict[int, int]):
            根据提供的初始库存字典重置所有 SKU 的库存数量。
        rolling(multiplier: float, predicts: list[float], sales: int):
            执行日常操作，包括销售、补货和库存回退。
        _selling(sku: SKU, sale: int):
            处理给定 SKU 的销售过程。
        _replenish(sku: SKU, multiplier: float, predicts: list[float]):
            计算并预订给定 SKU 的补货数量。
        _next_day(sku: SKU):
            将 SKU 推进到下一天。
        _summary(sku: SKU, sales: float, predicts: list[float]):
            总结日常操作并更新数量。
        _revert_stock(sku: SKU):
            根据 rts_day 回退给定 SKU 的库存。
        summary_result():
            返回补货过程的总结，包括 abo_qty、bind_qty 和 rts_qty。
    """

    def __init__(
        self,
        rts_day: int = 14,
        debug: bool = False,
    ):
        self.rts_day = rts_day
        self.strategy = StrategyB()
        self.debug = debug
        self.reset()
        # self.snapshots = Snapshots(
        #     {"sku": rts_day + 1, "global": rts_day + 1},
        #     {"sku": 2, "global": 0.1},
        # )
        # 指标

    def reset(self):
        self.set_state()
        self.overnight = []
        if self.debug:
            self.debug_df = pd.DataFrame(
                columns=[
                    "Date",
                    "ModelID",
                    "LeadTime",
                    "BeginningQty",
                    "BindingQty",
                    "EndingQty",
                    "ReplenishQty",
                    "PredictQty",
                    "SellingQty",
                    "RtsQty",
                    "PredictArriveStock",
                    "PredictQtyList",
                ]
            )

    def save_cur_row(self, name, header=False):
        self.debug_df.to_csv(name, mode="a", header=header, index=False)

    def add_result_to_csv(self, a, b, c):
        if self.debug:
            self.debug_df.loc[len(self.debug_df)] = ["", "", "", "", a, "", b, "", "", c, "", ""]
            self.save_cur_row("debug.csv")

    def set_state(self):
        self.rts_qty = 0
        self.bind_qty = 0
        self.abo_qty = 0
        self.total_sales = 0

    def rolling(
        self,
        rolling_day: int,
        sku: SKU,
        predicts: list[list[float]],
        sales_list: list[int],
        get_multiplier: Callable[[int], float],
        overnight_key: int,  # 稳定使用第一天
    ):
        # 预先获取 sku.id，避免重复访问
        sku_id = sku.id
        lead_time_bind = 0
        for i in range(1, rolling_day):
            multiplier = get_multiplier(sku_id)
            self.rolling_one_day(sku, predicts[i], sales_list[i], multiplier, overnight_key)
            if i == sku.lead_time:
                lead_time_bind = sku.bind_stock

        # return self.abo_qty, self.total_sales, self.bind_qty, self.rts_qty, lead_time_bind
        return lead_time_bind

    def _add_debug_info(self, sku: SKU, predicts: list[float], sales: int):
        if not self.debug:
            return
        self.debug_df.loc[len(self.debug_df)] = [
            sku.day_index,
            sku.id,
            sku.lead_time,
            sku.begin_stock + sku.today_arrived,
            sku.bind_stock,
            sku.end_stock,
            sku.replenish_qty,
            int(predicts[0]),
            sales,
            sku.rts_qty,
            sku.today_arrived,
            predicts,
        ]

    def rolling_one_day(
        self,
        sku: SKU,
        predicts: list[float],
        sales: int,
        multiplier: float,
        overnight_key: int,
    ):
        self.total_sales += sales

        # 合并多个操作减少方法调用开销
        sku.next_day()

        # 合并销售相关操作
        sku.bind_stock = self.strategy.bind_stock(sku, sales)

        # 晚上卖完
        sku.selling(sku.bind_stock)

        # 补货
        replenish_qty = self.strategy.calculate_rep(sku, predicts, multiplier)
        sku.book_order(replenish_qty)

        # 晚上标记为退货
        self.rts_qty += sku.revert_stock(self.rts_day)
        sku.end_stock = self.strategy.end_stock(sku)

        # 记录overnight数据
        self.overnight.append(sku.ending_stock_group.get(overnight_key, 0))

        self._add_debug_info(sku, predicts, sales)

    # def _snapshot(self):
    #     self.snapshots.draw_line()

    # def _snapshot_table(self):
    #     self.snapshots.draw_table("sku")

    def _next_day(self, sku: SKU):
        sku.next_day()

    def _summary(self, sku: SKU, sales: float, predicts: list[float]):
        self.bind_qty += sku.bind_stock
        self.abo_qty += sku.replenish_qty

        # self.snapshots.record("sku", "sku_begin_stock", sku.day_index, sku.begin_stock)
        # self.snapshots.record("sku", "sku_arrive_stock", sku.day_index, sku.today_arrived)
        # self.snapshots.record("sku", "sku_can_bind", sku.day_index, sku.begin_stock + sku.today_arrived)
        # self.snapshots.record("sku", "sales", sku.day_index, sales)
        # self.snapshots.record("sku", "sku_bind_stock", sku.day_index, sku.bind_stock)
        # self.snapshots.record("sku", "sku_end_stock", sku.day_index, sku.end_stock)
        # self.snapshots.record("sku", "predicts", sku.day_index, predicts)
        # self.snapshots.record("sku", "sku_replenish_qty", sku.day_index, sku.replenish_qty)
        # self.snapshots.record("sku", "rts_qty", sku.day_index, sku.rts_qty)
        # self.snapshots.record("global", "total_rts_rate", sku.day_index, self.rts_qty / self.total_sales)
        # self.snapshots.record("global", "total_bind_rate", sku.day_index, self.bind_qty / self.total_sales)
        # # self.snapshots.re_draw_map("ending_stock", sku.ending_stock_group)

    def _revert_stock(self, sku: SKU):
        self.rts_qty += sku.revert_stock(self.rts_day)

    def summary_result(self):
        return self.abo_qty, self.bind_qty, self.rts_qty
