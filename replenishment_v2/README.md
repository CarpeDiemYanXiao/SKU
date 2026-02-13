# 电商库存补货强化学习系统 V2（增强版）

## 项目目标
在 RTS（退货率）不升高的前提下，ACC（售出率）提升 ≥5%

**Baseline**: ACC ~75%, RTS ~2.4%  
**Target**: ACC ≥80%, RTS ≤2.4%

## 优化策略（参考 rl_0113/replenishment_abo）

### 1. State 设计优化
- 增加库存周转特征
- 引入库存健康度指标
- **在线状态归一化**（RunningMeanStd）
- 特征标准化改进

### 2. Reward 工程
- 动态权重调整
- 引入 RTS 约束惩罚
- 分层奖励设计
- **在线奖励归一化**

### 3. 网络结构
- 添加残差连接
- LayerNorm 处理不同量纲
- 更深的特征提取器
- **正交初始化**提高训练稳定性

### 4. 训练策略（增强版）
- **课程学习** (Curriculum Learning)
- **完整 Checkpoint** 保存/加载
- **优势值裁剪**防止极端值
- **Log ratio 裁剪**防止数值溢出
- 分布式训练支持
- 诊断日志和 TensorBoard

## 目录结构
```
replenishment_v2/
├── config/
│   └── default.yaml         # 配置文件（增强版）
├── src/
│   ├── dataset.py           # 数据加载
│   ├── environment.py       # 强化学习环境
│   ├── agent.py             # PPO Agent（增强版）
│   ├── networks.py          # 神经网络
│   ├── reward.py            # Reward 设计
│   ├── simulator.py         # 库存模拟器
│   └── utils.py             # 工具函数（含归一化模块）
├── train.py                 # 训练入口（增强版）
├── evaluate.py              # 评估入口
└── experiments/
    └── optimization_log.md  # 调优日志
```

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 训练
```bash
# 基础训练
python train.py --config config/default.yaml --data_path data/2000_sku.parquet

# 从 checkpoint 恢复训练
python train.py --resume output/xxx/best_model.pth --resume_mode full

# 只加载权重进行微调
python train.py --resume output/xxx/best_model.pth --resume_mode weights
```

## 新增功能说明

### 1. 状态/奖励归一化
配置文件中启用：
```yaml
training:
  use_state_norm: true    # 状态在线归一化
  use_reward_norm: true   # 奖励在线归一化
  norm_clip: 10.0         # 归一化后的裁剪范围
```

### 2. 课程学习
分阶段调整奖励权重，让模型逐步学习：
```yaml
training:
  curriculum:
    enabled: true
    stages:
      - name: "warmup"
        episodes: 100
        rts_weight: 0.5   # 初期降低 RTS 惩罚
      - name: "refine"
        episodes: 200
        rts_weight: 1.0   # 后期提高 RTS 权重
```

### 3. 完整 Checkpoint
保存内容包括：
- 网络权重（policy_net, value_net）
- 优化器状态
- 训练进度（episode, update_step）
- 归一化器状态
- 最佳指标

### 评估
```bash
python evaluate.py --model_path output/best_model.pth --data_path data/100k_sku.parquet
```

## 核心改进点

1. **双目标优化**: 显式建模 ACC 和 RTS 的 trade-off
2. **自适应动作空间**: 根据 SKU 特性动态调整 multiplier 范围
3. **分层策略**: 先保证 RTS 约束，再优化 ACC
4. **特征工程**: 引入库存健康度、需求波动性等高阶特征
