from typing import Optional
from .order_pool import OrderPool


class SKU(object):
    """
    SKU类用于对单个SKU进行滚动计算, 每次迭代到下一个状态.
    Attributes:
        id (int): SKU的唯一标识符。
        lead_time (int): 交货时间, 默认为1。
        reset_end_stock (int): 昨日库存, 用于重置, 默认为0。
        day_index (int): 当前天的索引，初始为-1表示昨天。
        begin_stock (int): 当天开始时的库存。
        bind_stock (int): 绑定库存。
        rts_qty (int): 可用库存数量。
        today_arrived (int): 今天到货的数量。
        end_stock (int): 当天结束时的库存。
        replenish_qty (int): 补货数量。
        ending_stock_group (dict): 结束库存的分组。
        order_pool (OrderPool): 订单池对象。
    Methods:
        reset(yesterday_stock: int = 0):
            重置SKU的状态。
        arriving_stock(index: int) -> int:
            获取指定索引的到货库存。
        selling(qty: int):
            销售指定数量的库存。
        order_arrive(qty: int):
            处理到货订单。
        next_day():
            进入下一天，更新相关状态。
        revert_stock(rts_day: int) -> int:
            退回指定天数的库存。
        book_order(qty: int):
            预定订单。
        snapshot_result() -> tuple:
            获取当前状态的快照结果。
    """

    def __init__(
        self,
        id: int,
        lead_time: int = 1,
        yesterday_end_stock: int = 0,
    ):
        # for reset
        self.reset_end_stock = yesterday_end_stock
        self.id = id
        self.lead_time = lead_time
        self.reset()

    def reset(self):
        self.day_index = -1
        self.ending_stock_group = {}
        self.order_pool = OrderPool()
        self.bind_stock = 0
        self.rts_qty = 0
        self.today_arrived = 0
        self.replenish_qty = 0
        self.begin_stock = 0
        self.end_stock = self.reset_end_stock
        self.end_stock_group_key = 0

    def set_state(self, lead_time: Optional[int]):
        self.lead_time = lead_time or self.lead_time

    def arriving_stock(self, index: int):
        return self.order_pool.get_order_stock(index)

    def selling(self, qty: int):
        for d in sorted(self.ending_stock_group):  ##self.end_stock_group_key
            self.end_stock_group_key = d
            if qty <= 0:
                break
            use_amount = min(self.ending_stock_group[d], qty)
            self.ending_stock_group[d] -= use_amount
            qty -= use_amount

    def order_arrive(self, qty: int):
        self.ending_stock_group[self.day_index] = qty  # 因为按照day_index滚动, 所以是有序的
        self.today_arrived = qty

    def next_day(self):
        self.day_index += 1
        qty = self.order_pool.get_order_stock(self.day_index)  # 早上到货了
        self.order_arrive(qty)
        self.begin_stock = self.end_stock

    def revert_stock(self, rts_day: int) -> int:
        if rts_day > self.day_index + 1:
            return 0

        # 还剩这么多需要退
        self.rts_qty = self.ending_stock_group.get(self.day_index + 1 - rts_day, 0)
        self.order_pool.return_order(self.day_index + 1 - rts_day, self.rts_qty)
        self.ending_stock_group[self.day_index + 1 - rts_day] = 0  # 退掉了

        # TODO 当晚还要过夜?
        return self.rts_qty

    def book_order(self, qty: int):
        self.replenish_qty = qty
        self.order_pool.place_order(self.day_index + self.lead_time, qty)

    def snapshot_result(self):
        return (
            self.day_index,
            self.begin_stock,
            self.order_pool.get_predict_arrive_stock(self.day_index),
            self.bind_stock,
            self.end_stock,
            self.replenish_qty,
        )
