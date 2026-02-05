from abc import ABC


class SKU(object):
    def __init__(
        self,
        soc_id: int,
        model_id: int,
        item_id: int,
        shop_id: int,
        lead_time: int = 1,
        yesterday_end_stock: int = 0,
    ):
        # for reset
        self.reset_end_stock = yesterday_end_stock
        self.soc_id = soc_id
        self.model_id = model_id
        self.item_id = item_id
        self.shop_id = shop_id
        self.lead_time = lead_time
        self.reset()

    def reset(self):
        self.day_index = -1  # 昨天为-1
        self.begin_stock = 0
        self.bind_stock = 0
        self.rts_qty = 0
        self.today_arrived = 0
        self.end_stock = self.reset_end_stock
        self.replenish_qty = 0
        self.ending_stock_group = {}
        self.order_pool = OrderPool(1, self.lead_time)

    def arriving_stock(self, index: int):
        return self.order_pool.predict_arrive_stock.get(index, 0)

    def selling(self, qty: int):
        for d in sorted(self.ending_stock_group):
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
        qty = self.order_pool.get_rep(self.day_index)  # 早上到货了
        self.order_arrive(qty)
        self.begin_stock = self.end_stock

    def revert_stock(self, rts_day: int) -> int:
        if rts_day > self.day_index:
            return 0

        reverted_qty = self.ending_stock_group[self.day_index - rts_day]
        self.ending_stock_group[self.day_index - rts_day] = 0  # 退掉了

        # TODO 当晚还要过夜?
        return reverted_qty

    def book_order(self, qty: int):
        self.replenish_qty = qty
        self.order_pool.add_rep(self.day_index, qty, self.lead_time)

    def snapshot_result(self):
        return (
            self.day_index,
            self.begin_stock,
            self.order_pool.predict_arrive_stock,
            self.bind_stock,
            self.end_stock,
            self.replenish_qty,
        )


class OrderPool(ABC):
    def __init__(self, rolling_times: int, lead_time: int) -> None:
        super().__init__()
        self.rep_at = {}
        # self.rep_arrive_pre_sum = [0] * (10 + lead_time + 1)
        self.predict_arrive_stock = {}

    def add_rep(self, index: int, qty: int, lead_time: int):
        self.rep_at[index] = qty
        # self.rep_arrive_pre_sum[index + lead_time] = self.rep_arrive_pre_sum[index + lead_time - 1] + qty
        self.predict_arrive_stock[index + lead_time] = qty

    def get_rep(self, index: int) -> int:
        return self.predict_arrive_stock.get(index, 0)
