# 调优思路与操作建议（2026-02-08）

## 目标
- 在 RTS 不升高前提下，ACC 提升 ≥5%。

## 关键思路
1. **约束式优化优先**：RTS 作为硬约束，用 Lagrange 乘子自动调节惩罚强度，允许在安全区间内放松补货以抬升 ACC。
2. **补货决策前瞻**：引入预估 RTS（含预测需求与库存天数）作为即时风险信号，让策略提前规避过量补货。
3. **库存天数策略更合理**：stock_days 动作为主，补货上限允许更高覆盖天数，同时加入 leadtime 缓冲避免到货前断供。
4. **训练一致性**：确保训练过程使用同一个 Reward 实例，避免课程学习/权重更新失效。

## 建议的实验顺序
1. **Exp-002（已实现）**：BalancedReward + Lagrange 约束 + 预估RTS
2. **Exp-003**：提高 `sell_ratio` 权重至 0.8，微调 `lagrange_lr` 到 0.02
3. **Exp-004**：stock_days_range 提升到 [0,7]，同时将 `overstock_days` 降为 4.5
4. **Exp-005**：打开 `use_reward_norm`，检查稳定性（若 acc 震荡再关闭）

## 观测指标
- 全局：ACC, RTS, total_replenish
- 过程：lagrange 轨迹、estimate_rts 惩罚占比、overstock 惩罚占比

## 风险与回退
- 若 RTS 上升：提高 `rts_penalty` 或 `estimate_rts_penalty`，降低 `max_replenish_days`
- 若 ACC 无提升：提高 `sell_ratio` 或增大 `stock_days_range`，同时观察 overstock

## 结果记录
请在优化日志中记录每次实验：
- 配置变更（超参、奖励项）
- ACC/RTS 变化
- 是否达到验收阈值
