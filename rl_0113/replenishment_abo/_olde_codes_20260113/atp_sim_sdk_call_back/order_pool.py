from abc import ABC
from copy import deepcopy


class OrderPool(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.place_order_stock = {}
        self.return_order_stock = {}

    def place_order(self, arrive_at: int, qty: int):
        self.place_order_stock[arrive_at] = self.place_order_stock.get(arrive_at, 0) + qty

    def get_order_stock(self, index: int) -> int:
        return self.place_order_stock.get(index, 0) - self.return_order_stock.get(index, 0)

    def return_order(self, arrived_at: int, qty: int) -> int:
        self.return_order_stock[arrived_at] = qty
        return self.return_order_stock[arrived_at]

    def get_predict_arrive_stock(self, index: int, lead_time: int = 5) -> list[int]:
        return [self.place_order_stock.get(i, 0) for i in range(index, index + lead_time)]

    def get_place_order(self, index: int) -> int:
        return self.place_order_stock.get(index, 0)

    def get_return_order(self, index: int) -> int:
        return self.return_order_stock.get(index, 0)
