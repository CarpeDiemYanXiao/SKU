# replenishment_v2 优化方案

## 项目目标
- **Baseline**: ACC ~75%, RTS ~2.4%
- **Target**: ACC ≥80%, RTS ≤2.4%（ACC提升≥5%，RTS不升高）
- **测试基准**: ≥5个测试集，每个100k SKU，共≥50万SKU

---

## 一、相对于 rl_0113 基线的架构改进

### 1. 动作空间重新设计
- 基线使用 **连续动作空间**（乘数 multiplier），策略梯度方差大、收敛不稳定
- V2 改为 **离散动作空间**，stock_days 模式：目标库存覆盖天数 [0, 7]，步长 0.5，共 15 个动作
- 离散空间直接输出 softmax 概率，训练更稳定，避免连续空间的方差问题

### 2. 状态特征扩展 (8维 → 23维)
- 基线仅 8 维（current_stock + 5天在途 + lt_begin_stock + lt_demand）
- V2 扩展至 23 维，新增：
  - **库存健康指标**: stock_health, days_of_stock, near_expire_ratio, total_transit
  - **需求统计特征**: demand_freq, order_ratio_l7d/l14d, avg/std_daily_item_qty_l7d/l14d
  - **趋势特征**: trend_item_qty_l3d_minus_l7d, trend_item_qty_l3d_minus_l14d
- 保留基线最关键的 lt_begin_stock（前向模拟到货时库存）和 lt_demand（到货日需求）

### 3. 奖励工程重构
- 基线使用简单的 bind - rts 线性组合
- V2 使用 **BalancedReward** 多目标设计：
  - bind 奖励（售出量，权重 0.3）
  - rts 惩罚（退货率，权重 0.45）
  - safe_stock 奖励（安全库存达标，权重 0.45，standard=0.9）
  - stockout 惩罚（缺货，权重 0.4）
  - overnight 惩罚（过夜库存，权重 0.005）
  - 区分头部SKU（order_ratio_7d > 0.3）和尾部SKU，差异化权重
- **在线奖励归一化**（RunningMeanStd），处理不同SKU的规模差异

### 4. 网络结构改进
- 基线: 单层 128 维全连接
- V2: [128, 128] 双层 + LayerNorm + 正交初始化
- 支持残差连接（当前配置关闭，避免小网络过拟合）
- 输入层 LayerNorm 处理不同量纲特征

### 5. PPO 训练改进
- **Per-SKU GAE**: 修复跨SKU GAE污染——每个SKU独立计算GAE再拼接，而非将所有SKU当一条连续序列
- **优势值裁剪** (advantage_clip=5.0): 防止极端优势值导致策略剧变
- **Log ratio 裁剪** (log_ratio_clip=20.0): 防止数值溢出
- **在线状态归一化** (RunningMeanStd): 自适应跟踪特征分布
- k_epochs=4, grad_norm=1.0, entropy_coef=0.015

### 6. 课程学习 (Curriculum Learning)
三阶段渐进式训练，逐步调整奖励权重：
| 阶段 | Episode | rts_weight | safe_stock_weight | 目标 |
|------|---------|------------|-------------------|------|
| warmup_bind | 1-15 | 0.35 | 0.40 | 先学基本补货，温和约束RTS |
| balance | 16-35 | 0.45 | 0.45 | 均衡优化，提升safe_stock标准 |
| acc_focus | 36-50 | 0.45 | 0.45 | 聚焦ACC提升，保持RTS约束 |

### 7. 大规模SKU子采样
- 100k SKU 全量训练太慢（~13min/episode）
- 每个episode随机采样 10,000 SKU 训练，速度提升约 10x
- 不同episode采样不同子集，总体覆盖所有SKU
- ACC 分母使用采样子集的 market_sales，保证指标准确

---

## 二、补货量计算改进

### Base Stock 正则化补货
```
补货量 = max(目标到货库存 − 预计到货时库存, 0)
```
- **目标到货库存** = daily_demand × target_stock_days
- **预计到货时库存** (lt_begin_stock) = 前向模拟 leadtime 天的到货与消耗
- daily_demand 使用 pred_y 和 avg_sales 的 50/50 混合
- 单次补货上限: daily_demand × 7 天
- 结果取整避免碎片化在途库存

### 需求序列校准
- 利用前向窗口的历史销售数据统计，校准预测值与实际的系统性偏差
- 混合系数 alpha=0.2，校准因子 clamp 在 [0.5, 2.0] 范围内
- 校准后的需求应用于: lt_begin_stock 计算、lt_demand 特征、补货量计算

---

## 三、修复的关键 Bug

1. **GAE 跨SKU污染**: 原始实现把所有SKU transition当一条序列做GAE，SKU-A末尾的TD error传递到SKU-B开头。修复为 per-SKU 独立计算。
2. **奖励归一化**: 训练和评估使用不同的 RunningMeanStd 实例，导致评估指标失真。统一为共享实例。
3. **estimate_rts 修复**: RTS 估算公式错误导致惩罚信号不准确。
4. **评估归一化器**: 评估阶段误用训练归一化器参数。
5. **补货量取整**: 浮点补货量导致碎片化在途库存。
6. **lt_demand 累积错误**: lt_demand 使用了错误的 leadtime 范围求和。
7. **特征安全访问**: 静态特征索引越界导致偶发崩溃。

---

## 四、工程优化

- **批量测试脚本** (run_all_tests.py): 自动遍历5个100k测试集，每完成一个立即写日志
- **TensorBoard 日志**: 记录 ACC、RTS、reward、loss 等训练曲线
- **完整 Checkpoint**: 保存 actor/critic/optimizer/normalizer 全部状态，支持断点续训
- **GPU 大批量**: batch_size=8192, mini_batch_size=2048，充分利用 32GB 显存
- **输出精简**: 仅输出 ACC/RTS 关键指标，减少日志噪声

---

## 五、当前最优超参数

```yaml
action: stock_days [0,7], step=0.5, 15个动作
reward: bind=0.3, rts=0.45, overnight=0.005, safe_stock=0.45, stockout=0.4
        safe_stock_standard=0.9, head_sku_threshold=0.3
network: [128, 128] + LayerNorm + orthogonal init
ppo: lr=3e-4, k_epochs=4, clip=0.2, entropy=0.015, grad_norm=1.0
     batch=8192, mini_batch=2048, gamma=0.99, gae_lambda=0.95
training: 50 episodes (3-stage curriculum), sku_sample_size=10000
```
