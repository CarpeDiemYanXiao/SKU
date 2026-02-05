from pathlib import Path


# 基类
class TaskConfig:
    """TaskConfig
    保存任务相关的必要信息
    """

    def __init__(self):
        self.task_name = ""  # 项目名
        self.data_ver = ""  # 数据版本
        self.para_ver = ""  # 实验版本
        self.data_path = ""  # 原始训练数据名
        self.base_dir = ""  # 从main文件传进来的, 项目文件夹的绝对路径, 不需要自己指定
        self.model_name = ""
        self.optim_name = ""
        self.loss_name = ""
        self.value_loss_name = ""
        self.advantages_fun = ""
        self.label_map = ""
        self.cat_cols = []
        self.con_cols = []

    def initialize(self):
        """
        根据基本参数初始化出某些必要的参数。
        目前包括各种规定好的路径。
        """
        base_dir = Path(self.base_dir)
        outs_dir = base_dir.joinpath("output", self.task_name, self.data_ver, self.para_ver)
        outs_dir.mkdir(parents=True, exist_ok=True)
        tb_log_dir = outs_dir.joinpath("tensorboard_logs")
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        self.outs_dir = str(outs_dir)
        self.logs_path = str(outs_dir.joinpath("training.log"))  # log
        self.tb_log_path = str(tb_log_dir)
        # self.tsbd_path = str(outs_dir.joinpath("tensorboard"))  # tensorboard 文件夹
        self.conf_path = str(outs_dir.joinpath("model.json"))  # config json
        self.policy_model_best_reward_filename = str(outs_dir.joinpath("repl_policy_best_reward_model.pth"))
        self.model_filename = str(outs_dir.joinpath("repl_model.pth")) # 同时保存actor和critic的模型权重
        self.policy_model_filename = str(outs_dir.joinpath("repl_policy_model.pth"))
        self.value_model_filename = str(outs_dir.joinpath("repl_value_model.pth"))
        self.reward_trend_filename = str(outs_dir.joinpath("reward_trend.jpg"))
        # self.tensorboard_log_dir = 
        self.ckpt_path = str(outs_dir.joinpath("model.pth"))  # model state dict + config 实例
        self.onnx_path = str(outs_dir.joinpath("model.onnx"))  # model onnx

    def update(self, to_update, priority: str):
        """
        合并另一些属性。
        如果存在冲突的话根据优先级决定合并方式。
        """
        assert priority in ["low", "high"]
        if isinstance(to_update, dict):
            pass
        else:
            assert hasattr(to_update, "__dict__")
            to_update = to_update.__dict__
        # filter __
        config = {k: v for k, v in self.__dict__.items()}
        to_update = {k: v for k, v in to_update.items() if not k.startswith("__")}
        # priority
        if priority == "low":
            self.__dict__.update(to_update | config)
        elif priority == "high":
            self.__dict__.update(config | to_update)

    def __str__(self):
        """打印格式"""
        strs = []
        for k, v in self.__dict__.items():
            if isinstance(v, str):
                s = f'\n\t{k} = "{v}"'
            else:
                s = f"\n\t{k} = {v}"
            strs.append(s)
        return "config: " + "".join(strs)
