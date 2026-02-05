class TrainerConfig:
    """TrainerConfig: 普通类"""

    def __init__(self):
        self.k_epochs = 10
        self.batch_size = 2048
        self.learning_rate = 0.0003
        self.norm_clip = 0.0  # 归一化时的梯度裁剪
        self.clip_grad = 0.0  # 梯度裁剪
        self.clip_grad_decay = 1.0  # 仅 clip_grad > 0 才生效: 梯度裁剪Exponential衰减系数, 梯度裁剪最小0.1
        self.gamma = 0.99
        self.print_every = 4
        self.sample = 1.0  # 样本采样
        self.l2 = 0.0001
        self.best_reward = -1e18  # 记录最优的 valid loss
        # self.save_every_eposide = 50  # 隔几个eposide记录一次模型
        self.save_every_eposide = 2
        self.seed = 42
        self.episode_lower_limit = 1
        # self.episode_lower_limit = 199
        # 归一化相关设置
        self.use_state_norm = True
        self.use_discount_reward_norm = True
        self.reward_scaling = "ZFilter"
        self.state_scaling = "Normalization"
        self.center = True
        self.scale = True
