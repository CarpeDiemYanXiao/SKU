from typing import Optional
from matplotlib import pyplot as plt

from .snapshot import Snapshots
from .entity import SKU
from .replenish_scene import ReplenishScene


class RollingEnv(object):
    rts_qty = 0
    bind_qty = 0
    spx_ado = 0
    abo_qty = 0
    total_sales = 0

    def __init__(
        self,
        replenish_scene: ReplenishScene,
        rts_day: int = 14,
    ):
        self.rts_day = rts_day
        self.replenish_scene = replenish_scene
        self.snapshots = Snapshots(
            {"sku": rts_day + 1, "global": rts_day + 1},
            {"sku": 2, "global": 0.1},
        )
        # 指标

    def reset(self, sku: Optional[SKU] = None):
        self.rts_qty = 0
        self.bind_qty = 0
        self.abo_qty = 0
        self.total_sales = 0
        if sku:
            sku.reset()

    def rolling(
        self,
        sku: SKU,
        multiplier: float,
        predicts: list[float],
        sales: int,
    ):
        self.total_sales += sales  # 统计用
        self._next_day(sku)  # 凌晨更新到下一天
        self._revert_stock(sku)  # 退货超过rts_day的库存
        self._selling(sku, sales)  # 卖商品
        self._replenish(sku, multiplier, predicts)  # 对商品补货
        self._summary(sku, sales, predicts)  # 总结

        return self.abo_qty, self.total_sales, self.bind_qty, self.rts_qty

    def _selling(self, sku: SKU, sale: int):
        sku.bind_stock = self.replenish_scene.strategy.bind_stock(sku, sale)
        sku.selling(sku.bind_stock)  # 卖出
        sku.end_stock = self.replenish_scene.strategy.end_stock(sku)  # 先计算当天期末

    def _replenish(self, sku: SKU, multiplier: float, predicts: list[float]):
        # 计算补货量, 下ABO单
        replenish_qty = self.replenish_scene.do_replenish(sku, multiplier, predicts)
        sku.book_order(replenish_qty)

    def _snapshot_wait(self):
        plt.ioff()
        plt.show()

    def _snapshot(self):
        self.snapshots.draw_line()

    def _snapshot_table(self):
        self.snapshots.draw_table("sku")

    def _next_day(self, sku: SKU):
        sku.next_day()

    def _summary(self, sku: SKU, sales: float, predicts: list[float]):
        self.bind_qty += sku.bind_stock
        self.abo_qty += sku.replenish_qty

        self.snapshots.record("sku", "sku_begin_stock", sku.day_index, sku.begin_stock)
        self.snapshots.record("sku", "sku_arrive_stock", sku.day_index, sku.today_arrived)
        self.snapshots.record("sku", "sku_can_bind", sku.day_index, sku.begin_stock + sku.today_arrived)
        self.snapshots.record("sku", "sales", sku.day_index, sales)
        self.snapshots.record("sku", "sku_bind_stock", sku.day_index, sku.bind_stock)
        self.snapshots.record("sku", "sku_end_stock", sku.day_index, sku.end_stock)
        self.snapshots.record("sku", "predicts", sku.day_index, predicts)
        self.snapshots.record("sku", "sku_replenish_qty", sku.day_index, sku.replenish_qty)
        self.snapshots.record("sku", "rts_qty", sku.day_index, sku.rts_qty)
        self.snapshots.record("global", "total_rts_rate", sku.day_index, self.rts_qty / self.total_sales)
        self.snapshots.record("global", "total_bind_rate", sku.day_index, self.bind_qty / self.total_sales)
        # # self.snapshots.re_draw_map("ending_stock", sku.ending_stock_group)

    def _revert_stock(self, sku: SKU):
        self.rts_qty += sku.revert_stock(self.rts_day)

    def summary_result(self):
        return self.abo_qty, self.bind_qty, self.rts_qty
