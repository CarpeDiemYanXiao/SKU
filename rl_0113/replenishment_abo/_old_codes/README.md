# replenishment

安装指南：
0 建议用conda创建虚拟环境
conda create --name ppo_env python=3.10
conda activate ppo_env

1. 安装依赖：
pip install -r requirements.txt

2. 运行训练：
python ppo_test.py --ui  #使用界面模式训练，默认为无界面模式


其他相关：
CartPole-v0 的观察空间是一个四维的连续空间，具体如下：
序号	观察值	范围
0	小车位置（Cart Position）	[-4.8, 4.8]，但实际任务中当小车位置超出[-2.4, 2.4]时，任务会终止
1	小车速度（Cart Velocity）	(-∞, +∞)
2	杆子角度（Pole Angle）	[-0.418 弧度, 0.418 弧度]，但实际任务中当杆子角度超出[-0.2095 弧度, 0.2095 弧度]时，任务会终止
3	杆子角速度（Pole Angular Velocity）	(-∞, +∞)

CartPole-v0 的动作空间是一个离散空间，具体如下：
序号	动作	含义
0	向左推（Push Cart to the Left）	小车向左移动
1	向右推（Push Cart to the Right）	小车向右移动
动作空间的类型为 Discrete，表示一个包含两个可能动作的离散空间

在 CartPole-v0 环境中，调用 env.step(1) 会执行一个动作，即向右推小车。这个函数调用的返回值是一个四元组，具体如下：
观察值（Observation）：一个包含四个元素的数组，表示当前环境的状态。这四个元素分别是小车的位置、小车的速度、杆子的角度和杆子的角速度。
奖励（Reward）：一个浮点数，表示执行动作后获得的奖励。在 CartPole-v0 环境中，如果杆子没有倒下且小车没有超出边界，奖励为 1.0；否则，奖励为 0.0。
完成标志（Done）：一个布尔值，表示任务是否已经完成。在 CartPole-v0 环境中，如果杆子倒下（角度超出 [-0.2095, 0.2095] 弧度）或小车超出边界（位置超出 [-2.4, 2.4]），则任务完成，done 为 True；否则，done 为 False。
信息（Info）：一个字典，包含一些额外的信息。在 CartPole-v0 环境中，这个字典通常是空的。
因此，env.step(1) 的返回值可以表示为：
Python
复制
(observation, reward, done, info)
其中，observation 是一个包含四个元素的数组，reward 是一个浮点数，done 是一个布尔值，info 是一个字典。