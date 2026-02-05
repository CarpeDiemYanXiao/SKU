import numpy as np
import torch.distributed as dist
import torch


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.ones(shape)
        self.std = np.sqrt(self.S)
        # 定义一个Sample_num，用于同步多个进程之间的mean和std
        self.sample_num = 0

    def reset_sample_num(self):
        self.sample_num = 0

    def set_sample_num(self, sample_num):
        self.sample_num = sample_num

    def sync_mean_std(self, distributed=False, rank=0, world_size=1):
        norm_list = [0] * world_size
        if distributed:
            # 先转tensor才能传，ndarray不能序列化
            norm_dict = {"mean": torch.tensor(self.mean), "std": torch.tensor(self.std), "num": self.sample_num}
            dist.all_gather_object(norm_list, norm_dict)
            norm_list = [i for i in norm_list if i["num"] > 0]
            nums_all = sum([i["num"] for i in norm_list])
            self.mean = np.sum([np.array(i["num"] * i["mean"]) for i in norm_list], axis=0) / nums_all
            std_2 = (
                np.sum([(i["mean"] ** 2 + i["std"] ** 2) * i["num"] for i in norm_list], axis=0) / nums_all
                - self.mean**2
            )
            self.std = np.sqrt(std_2)
        else:
            pass

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)

class Normalization:
    def __init__(self, shape, config):
        self.running_ms = RunningMeanStd(shape=shape)
    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x

class ZFilter:
    def __init__(self, shape, config):
        self.shape = shape  # reward shape=1
        self.center = config.center
        self.scale = config.scale
        self.clip = config.norm_clip
        # self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x, update=True):
        # self.R = self.gamma * self.R + x
        self.R = x
        if update:
            self.running_ms.update(self.R)
        if self.scale:
            if self.center:
                x = x / (self.running_ms.std + 1e-8)  # Only divided std
            else:
                diff = x - self.running_ms.mean
                diff = diff/(self.running_ms.std + 1e-8)
                x = diff + self.running_ms.mean
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x[0] if isinstance(x, np.ndarray) else x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)

class RewardFilter:
    def __init__(self, shape, config):
        self.shape = shape  # reward shape=1
        self.gamma = config.gamma
        self.clip = config.norm_clip
        # self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x, update=True):
        self.R = self.gamma * self.R + x
        #return_item = True if not isinstance(x, np.ndarray) else False
        self.R = x
        if update:
            self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x[0] if isinstance(x, np.ndarray) else x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)

class DefaultFilter:
    def __init__(self, shape, config):
        self.shape = shape  # reward shape=1
        self.clip = config.norm_clip

    def __call__(self, x, update=True):
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        pass