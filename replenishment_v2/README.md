# 电商库存补货强化学习系统 V2

## 项目目标
在 RTS（退货率）不升高的前提下，ACC（售出率）提升 ≥5%

**Baseline**: ACC ~75%, RTS ~2.4%  
**Target**: ACC ≥80%, RTS ≤2.4%

## 优化策略

### 1. State 设计优化
- 增加库存周转特征
- 引入库存健康度指标
- 特征标准化改进

### 2. Reward 工程
- 动态权重调整
- 引入 RTS 约束惩罚
- 分层奖励设计

### 3. 网络结构
- 添加残差连接
- 引入注意力机制
- 更深的特征提取器

### 4. 训练策略
- 课程学习
- 自适应探索
- 分阶段训练

## 目录结构
```
replenishment_v2/
├── config/
│   └── default.yaml         # 配置文件
├── src/
│   ├── dataset.py           # 数据加载
│   ├── environment.py       # 强化学习环境
│   ├── agent.py             # PPO Agent
│   ├── networks.py          # 神经网络
│   ├── reward.py            # Reward 设计
│   ├── simulator.py         # 库存模拟器
│   └── utils.py             # 工具函数
├── train.py                 # 训练入口
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
python train.py --config config/default.yaml --data_path data/2000_sku.parquet
```

### 评估
```bash
python evaluate.py --model_path output/best_model.pth --data_path data/100k_sku.parquet
```

## 核心改进点

1. **双目标优化**: 显式建模 ACC 和 RTS 的 trade-off
2. **自适应动作空间**: 根据 SKU 特性动态调整 multiplier 范围
3. **分层策略**: 先保证 RTS 约束，再优化 ACC
4. **特征工程**: 引入库存健康度、需求波动性等高阶特征
