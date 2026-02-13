"""端到端验证：per-SKU GAE + 奖励归一化 + PPO更新"""
import sys, numpy as np
sys.path.insert(0, '.')
from src.utils import load_config, set_seed, StateNormalizer, RewardNormalizer
from src.dataset import ReplenishmentDataset
from src.environment import create_env_with_reward
from src.agent import PPOAgent
from src.reward import create_reward
from train import RolloutBuffer, train_episode, compute_per_sku_gae

config = load_config('config/default.yaml')
set_seed(42)

# Use small subset for quick test
ds = ReplenishmentDataset(
    file_path=config['data']['train_path'],
    static_features=config['env']['state_features']['static'])
ds.sku_ids = ds.sku_ids[:50]  # Only 50 SKUs
ds.n_skus = 50

reward_fn = create_reward(config)
env = create_env_with_reward(ds, config, reward_fn)

agent = PPOAgent(config, device='cpu')
state_norm = StateNormalizer(shape=(env.state_dim,), clip=10.0)
reward_norm = RewardNormalizer(clip=10.0, gamma=config['ppo']['gamma'])

buffer = RolloutBuffer()
stats = train_episode(
    env, agent, buffer, state_norm, reward_norm,
    use_state_norm=True, use_reward_norm=True, use_terminal_reward=False)

acc = stats["acc"]
rts = stats["rts_rate"]
print(f"Episode stats: ACC={acc:.2f}%, RTS={rts:.2f}%")
print(f"Buffer size: {len(buffer)} transitions")
print(f"SKU IDs tracked: {len(set(buffer.sku_ids))} unique")

# Test per-SKU GAE
rollout_data = buffer.get()
assert "sku_ids" in rollout_data, "sku_ids missing from rollout_data!"
rollout_data = compute_per_sku_gae(rollout_data, agent.gamma, agent.gae_lambda)
assert "advantages" in rollout_data, "advantages missing!"
assert "returns" in rollout_data, "returns missing!"

adv = rollout_data["advantages"]
ret = rollout_data["returns"]
print(f"Advantages: mean={adv.mean():.4f}, std={adv.std():.4f}, "
      f"min={adv.min():.4f}, max={adv.max():.4f}")
print(f"Returns: mean={ret.mean():.4f}, std={ret.std():.4f}")

# Verify no NaN/Inf
assert not np.isnan(adv).any(), "NaN in advantages!"
assert not np.isnan(ret).any(), "NaN in returns!"
assert not np.isinf(adv).any(), "Inf in advantages!"

# Test agent update with pre-computed GAE
train_stats = agent.update(rollout_data)
print(f"Update OK: policy_loss={train_stats['policy_loss']:.6f}, "
      f"value_loss={train_stats['value_loss']:.4f}, "
      f"entropy={train_stats['entropy']:.4f}")

# Verify reward components are reasonable (normalized)
print(f"\nReward components from last step:")
comps = reward_fn.get_components()
for k, v in comps.items():
    print(f"  {k}: {v:.4f}")

# Verify get_feature_by_name works
sku_id = ds.sku_ids[0]
or7d = ds.get_feature_by_name(sku_id, 0, "order_ratio_l7d")
print(f"\nget_feature_by_name test: order_ratio_l7d={or7d:.4f}")

print("\n" + "=" * 60)
print("=== All end-to-end tests passed! ===")
print("=" * 60)
