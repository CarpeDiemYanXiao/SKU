from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print


algo = PPOConfig().env_runners(num_env_runners=1).resources(num_gpus=0).environment(env="CartPole-v1").build()

for i in range(10):
    result = algo.train()
    print(pretty_print(result))

    if i % 5 == 0:
        checkpoint_dir = algo.save().checkpoint.path
        print(f"Checkpoint saved in directory {checkpoint_dir}")
