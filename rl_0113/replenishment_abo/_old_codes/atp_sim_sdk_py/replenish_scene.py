from abc import ABC, abstractmethod

from .entity import SKU
from .strategy import Strategy


# 补货场景
class ReplenishScene(ABC):
    def __init__(self, strategy: Strategy):
        self.strategy = strategy

    @abstractmethod
    def do_replenish(
        self,
        sku: SKU,
        multiplier: float,
        predicts: list[float],  # 今天开始的预测结果
    ) -> int:
        raise Exception("not implemented")


class CacheOrder(ReplenishScene):
    def do_replenish(
        self,
        sku: SKU,
        multiplier: float,
        predicts: list[float],  # 今天开始的预测结果
    ) -> int:
        return self.strategy.calculate_rep(sku, predicts, multiplier)
