from abc import ABC, abstractmethod
import math

from .sku import SKU


class Strategy(ABC):
    def begin_stock(self, sku: SKU) -> int:
        return sku.begin_stock

    @abstractmethod
    def bind_stock(self, sku: SKU, sales: int) -> int:
        raise Exception("not implemented")

    def end_stock(self, sku: SKU) -> int:
        return sku.end_stock

    @abstractmethod
    def calculate_rep(self, sku: SKU, predicts: list[float], multiplier: float) -> int:
        raise Exception("not implemented")

    # @abstractmethod
    # def soc_stock(self, sku: SKU) -> int:
    #     raise Exception("not implemented")

    # @abstractmethod
    # def soc_in_station_available_stock(self, sku: SKU) -> int:
    #     raise Exception("not implemented")


class StrategyA(Strategy):
    def bind_stock(self, sku: SKU, sales: int) -> int:
        return min(sku.begin_stock + sku.today_arrived, sales)

    def calculate_rep(self, sku: SKU, predicts: list[float], multiplier: float) -> int:
        r = sku.replenish_qty
        p = sku.end_stock
        ld = sku.lead_time
        for i in range(1, sku.lead_time):
            f = p  # Fn
            p = max(f + sku.arriving_stock(sku.day_index + i) - predicts[i], 0)  # Pn
            r = int(max(predicts[ld] * multiplier - max(f + r - sum(predicts[0 : ld - 1]), 0), 0))
        return r


class StrategyB(Strategy):
    def bind_stock(self, sku: SKU, sales: int) -> int:
        updated_begin_stock = max(sku.begin_stock, 0)
        return int(min(updated_begin_stock + sku.today_arrived, sales))

    def end_stock(self, sku: SKU) -> int:
        updated_begin_stock = max(sku.begin_stock - sku.rts_qty, 0)
        return int(max(updated_begin_stock + sku.today_arrived - sku.bind_stock, 0))

    def calculate_rep(self, sku: SKU, predicts: list[float], multiplier: float) -> int:
        p = max(sku.begin_stock, 0)
        for i in range(0, sku.lead_time):
            p = max(p + sku.arriving_stock(sku.day_index + i) - predicts[i], 0)  # Pn
        return int(max(predicts[sku.lead_time] * multiplier - p, 0))


class StrategyC(Strategy):
    def bind_stock(self, sku: SKU, sales: int) -> int:
        return int(min(sku.begin_stock, sales))

    def end_stock(self, sku: SKU) -> int:
        return self._end_stock(sku.begin_stock, sku.bind_stock, sku.today_arrived)

    # def _end_stock(self, begin: int, bind: int | float, arrived: int):
    #     return int(max(begin - bind, 0) + arrived)
    def _end_stock(self, begin: int, bind: float, arrived: int):
        return int(max(begin - bind, 0) + arrived)

    def calculate_rep(self, sku: SKU, predicts: list[float], multiplier: float) -> int:
        p = sku.end_stock
        for i in range(1, sku.lead_time):
            f = p  # Fn
            p = self._end_stock(f, predicts[i], sku.arriving_stock(sku.day_index + i))  # Pn
        return int(max(predicts[sku.lead_time + 1] * multiplier - p, 0))
