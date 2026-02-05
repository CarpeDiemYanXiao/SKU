from .base import TaskConfig
import json

class CacheRpConfig(TaskConfig):
    def __init__(self, json_path=""):
        self.task_name = "cacherp"
        self.data_ver = "v01"  # 数据版本
        self.para_ver = "default"  # 实验版本
        self.data_path = ""  # 原始训练数据名
        self.use_mlflow = False
        self.mlflow_experiment_name = "replenishment_experiments"
        self.mlflow_host = None
        self.model_name = "base_ppo"
        self.optim_name = "AdamW"
        self.loss_name = "Base_loss"
        self.value_loss_name = "mae"
        self.advantages_fun = "base_advantages"
        self.state_static_columns = ["leadtime", "pred_y"]
        self.state_static_columns_map = {"leadtime": "lead_time", "pred_y": "predict_sales_in_lt"}
        # self.state_label_dict = {
        #                         "with_rts_split":["end_of_stock", "next_day_rts", "transit_stock_0", "transit_stock_1", "transit_stock_2", "transit_stock_3", "transit_stock_4"],
        #                         "with_rts_combine":["start_stock", "transit_stock_0", "transit_stock_1", "transit_stock_2", "transit_stock_3", "transit_stock_4"],
        #                         "default":["end_of_stock", "transit_stock_0", "transit_stock_1", "transit_stock_2", "transit_stock_3", "transit_stock_4"],
        #                         }
            
        self.state_label_dict = {
            "with_rts_split": [
                    "yesterday_end_stock",
                    "today_morning_rts",
                    "predict_arrive_day_1",
                    "predict_arrive_day_2",
                    "predict_arrive_day_3",
                    "predict_arrive_day_4",
                    "predict_arrive_day_5",
            ],
                "with_rts_combine": [
                        "start_stock",
                        "predict_arrive_day_1",
                        "predict_arrive_day_2",
                        "predict_arrive_day_3",
                        "predict_arrive_day_4",
                        "predict_arrive_day_5",
                    ],
            "with_stock_level": [
                    "start_stock",
                    "predict_arrive_day_1",
                    "predict_arrive_day_2",
                    "predict_arrive_day_3",
                    "predict_arrive_day_4",
                    "predict_arrive_day_5",
                    "stock_level",
                ],
            "default": [
                    "yesterday_end_stock",
                    "predict_arrive_day_1",
                    "predict_arrive_day_2",
                    "predict_arrive_day_3",
                    "predict_arrive_day_4",
                    "predict_arrive_day_5",
                ],
        }

        # self.rts_days = 14
        # self.rts_weight = 0.8
        # self.coverage_weight = 0.15
        # self.overnight_weight = 0.01
        # self.stockout_weight = 0.01
        # self.reward_type = "default"
        # self.safe_stock_weight = 0.01
        # self.safe_stock_standard = 0.5
        # self.head_sku_standard = 0.5
        # self.action_ls = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.3, 2.5, 2.8, 3]
        # self.state_label = "without_rts"
        

    def initialize(self):
        super().initialize()
        # 默认取6
        self.state_static_columns = self.curriculum_stages[-1]["state_static_columns"] if hasattr(self,"curriculum_stages") else self.state_static_columns # 取最后一个stage的state_static_columns

        # self.state_dim = len(self.state_label_dict.get(self.state_label, self.state_label_dict["default"])) + len(
        #     self.state_static_columns) + 5 + 1 # 增加rts_qty_list(5) + next_state_leadtime_begin_stock(1)
        
        self.state_dim = 2+len(self.state_static_columns)
        
        # self.state_dim = 3
        
        if "predicted_demand" in self.state_static_columns:self.state_dim += 5
        # # 产出state_dim
        # if self.state_label == "with_rts_split":
        #     self.state_dim = len(self.state_static_columns)+
        # else:
        #     self.state_dim = len(self.state_static_columns)+6
        # 产出feature_name
        # feature_names拼接起来
        # self.columns的顺序不能更改，关系到模型推理结果
        self.state_static_column_online = [
            self.state_static_columns_map[i] if i in self.state_static_columns_map.keys() else i
            for i in self.state_static_columns
        ]
        self.columns = (
            self.state_label_dict.get(self.state_label, self.state_label_dict["default"])
            + self.state_static_column_online
        )
        # todo：假如类别特征cat_cols
        self.con_cols = self.columns
