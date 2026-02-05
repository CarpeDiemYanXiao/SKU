import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MIN
import gym
from gym import spaces
import numpy as np
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
import ray
from ray.rllib.algorithms.ppo import PPOConfig


class MyEnv(gym.Env):
    def __init__(self, env_config):
        self.action_space = spaces.Discrete(2)  # 例如，两个离散动作
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)  # 例如，四维连续观测
        self.state = None

    def reset(self, *, seed=None, options=None):
        self.state = np.zeros(4)
        return self.state, {}

    def step(self, action):
        # 根据动作更新状态
        self.state = self.state + (action - 0.5) * 2
        reward = -np.sum(self.state**2)  # 例如，奖励为状态的负平方和
        terminated = np.linalg.norm(self.state) > 10  # 例如，状态范数超过10则终止
        truncated = False
        return self.state, reward, bool(terminated), truncated, {}


class MyModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(obs_space.shape[0], 128)
        self.fc2 = nn.Linear(128, num_outputs)
        self.value_head = nn.Linear(128, 1)

    def forward(self, input_dict, state, seq_lens):
        x = torch.relu(self.fc1(input_dict["obs"]))
        logits = self.fc2(x)
        self._value = self.value_head(x).squeeze(1)
        return logits, state

    def value_function(self):
        return self._value


def env_creator(env_config):
    return MyEnv(env_config)


register_env("my_custom_env", env_creator)
ModelCatalog.register_custom_model("my_model", MyModel)


ray.init()

config = (
    PPOConfig()
    .environment(env="my_custom_env", env_config={})
    .framework(framework="torch")
    .env_runners(num_env_runners=2)
    .resources(num_gpus=0)
    .training(model={"custom_model": "my_model", "custom_model_config": {}})
)
algo = config.build()

for i in range(5):
    result = algo.train()
    print(f"Iteration {i}: reward = {result['episode_reward_mean']}")
