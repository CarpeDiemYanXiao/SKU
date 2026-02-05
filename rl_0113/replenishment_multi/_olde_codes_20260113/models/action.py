from typing import Any


class Action:
    def __init__(self, multiplier_ls, action, policy, day_idx):
        self.multiplier_ls = multiplier_ls
        self.action = action
        self.policy = policy
        self.day_idx = day_idx

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass
