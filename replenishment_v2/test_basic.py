"""
快速测试脚本
验证代码能否正常运行
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_config, set_seed
from src.dataset import ReplenishmentDataset
from src.environment import create_env
from src.agent import PPOAgent, RolloutBuffer
from src.reward import create_reward


def test_basic():
    """基础功能测试"""
    print("=" * 60)
    print("Running basic tests...")
    print("=" * 60)
    
    # 1. 加载配置
    print("\n[1] Loading config...")
    config = load_config("config/default.yaml")
    print(f"    Task: {config['task']['name']}")
    print(f"    Action type: {config['action']['type']}")
    
    # 2. 设置随机种子
    print("\n[2] Setting seed...")
    set_seed(42)
    print("    Seed: 42")
    
    # 3. 加载数据集
    print("\n[3] Loading dataset...")
    data_path = config["data"]["train_path"]
    static_features = config["env"]["state_features"]["static"]
    
    try:
        dataset = ReplenishmentDataset(
            file_path=data_path,
            static_features=static_features,
        )
        print(f"    SKUs: {dataset.n_skus}")
        print(f"    Total sales: {dataset.total_sales}")
    except Exception as e:
        print(f"    ERROR: {e}")
        print("    Please check data path in config/default.yaml")
        return False
    
    # 4. 创建环境
    print("\n[4] Creating environment...")
    env = create_env(dataset, config)
    print(f"    State dim: {env.state_dim}")
    print(f"    Action dim: {env.n_actions}")
    
    # 5. 创建 Agent
    print("\n[5] Creating agent...")
    agent = PPOAgent(config, device="cpu")
    print(f"    Policy network: {type(agent.policy_net).__name__}")
    print(f"    Value network: {type(agent.value_net).__name__}")
    
    # 6. 测试一个 episode
    print("\n[6] Running one episode...")
    buffer = RolloutBuffer()
    
    state_map = env.reset()
    sku_ids = list(state_map.keys())[:10]  # 只测试前10个SKU
    
    steps = 0
    max_steps = 5  # 只测试5步
    
    for step in range(max_steps):
        action_map = {}
        
        for sku_id in sku_ids:
            if env.done_map.get(sku_id, False):
                continue
            
            state = state_map[sku_id]
            action, log_prob = agent.select_action(state)
            action_map[sku_id] = action
        
        if not action_map:
            break
        
        next_state_map, reward_map, done_map, info_map = env.step(action_map)
        state_map = next_state_map
        steps += 1
        
        print(f"    Step {step + 1}: Actions={len(action_map)}, "
              f"Avg Reward={sum(reward_map.values())/len(reward_map):.4f}")
    
    print(f"    Completed {steps} steps")
    
    # 7. 获取指标
    print("\n[7] Getting metrics...")
    metrics = env.get_metrics()
    print(f"    ACC: {metrics['acc']:.2f}%")
    print(f"    RTS: {metrics['rts_rate']:.2f}%")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_basic()
    sys.exit(0 if success else 1)
