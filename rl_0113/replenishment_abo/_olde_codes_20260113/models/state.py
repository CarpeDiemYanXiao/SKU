from abc import ABC, abstractmethod


class State(ABC):
    @abstractmethod
    def __array__(self):
        return []


class ReplenishState(State):
    day_idx = 0

    def __init__(self, predict_leadtime_day):
        self.state = [0, predict_leadtime_day]

    def __array__(self):
        return self.state

    def set_state(self, state):
        self.state = state
