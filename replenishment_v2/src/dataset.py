"""
数据加载模块
负责加载和预处理 parquet/csv 格式的库存补货数据
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class ReplenishmentDataset:
    """库存补货数据集"""
    
    def __init__(
        self,
        file_path: str,
        static_features: List[str],
        rank: int = 0,
        world_size: int = 1
    ):
        """
        Args:
            file_path: 数据文件路径 (parquet/csv)
            static_features: 静态特征列名列表
            rank: 分布式训练的进程rank
            world_size: 分布式训练的总进程数
        """
        self.file_path = Path(file_path)
        self.static_features = static_features
        self.rank = rank
        self.world_size = world_size
        
        # 加载数据
        self._load_data()
        
    def _load_data(self):
        """加载并预处理数据"""
        # 读取文件
        if self.file_path.suffix == ".csv":
            df = pd.read_csv(self.file_path)
        else:
            df = pd.read_parquet(self.file_path)
        
        # 处理 predicted_demand 列 (可能是字符串格式的列表)
        if "predicted_demand" in df.columns:
            if isinstance(df["predicted_demand"].iloc[0], str):
                df["predicted_demand"] = df["predicted_demand"].apply(eval)
        
        # 按 idx 和 date 排序
        df = df.sort_values(by=["idx", "date"]).reset_index(drop=True)
        
        # 分布式训练: 分片数据
        unique_idx = df["idx"].unique()
        if self.world_size > 1:
            chunk_size = len(unique_idx) // self.world_size
            start = self.rank * chunk_size
            end = start + chunk_size if self.rank < self.world_size - 1 else len(unique_idx)
            selected_idx = unique_idx[start:end]
            df = df[df["idx"].isin(selected_idx)]
        
        # 去重
        df = df.drop_duplicates(subset=["idx", "date"])
        
        # 按 SKU 分组
        grouped = df.groupby("idx")
        
        # 构建各种映射
        self.sku_ids = list(grouped.groups.keys())
        self.n_skus = len(self.sku_ids)
        self.total_sales = df["actual"].sum()
        
        # 销量序列
        self.sales_map = grouped["actual"].apply(list).to_dict()
        
        # 预测序列 (每天的6天预测)
        self.predicts_map = grouped["predicted_demand"].apply(
            lambda s: np.array(s.tolist())
        ).to_dict()
        
        # Leadtime 序列
        self.leadtime_map = grouped["leadtime"].apply(list).to_dict()
        
        # LT天预测销量
        self.pred_y_map = grouped["pred_y"].apply(list).to_dict()
        
        # 初始库存
        self.initial_stock_map = grouped["initial_stock"].first().to_dict()
        
        # 静态特征 (每天可能不同，所以是列表)
        self.static_features_map = grouped.apply(
            lambda x: x[self.static_features].values.tolist(),
            include_groups=False
        ).to_dict()
        
        # 每个 SKU 的天数
        self.n_days_map = grouped["date"].count().to_dict()
        
        # 未来真实销量 (用于模拟环境)
        actual_cols = ['actual_0', 'actual_1', 'actual_2', 'actual_3', 'actual_4', 'actual_5']
        if all(c in df.columns for c in actual_cols):
            self.actuals_map = grouped.apply(
                lambda x: x[actual_cols].values.tolist(),
                include_groups=False
            ).to_dict()
        else:
            self.actuals_map = None
        
        # 统计特征 (用于特征工程)
        if "avg_daily_item_qty_l7d" in df.columns:
            self.avg_qty_7d_map = grouped["avg_daily_item_qty_l7d"].apply(list).to_dict()
        else:
            self.avg_qty_7d_map = None
            
        if "demand_freq" in df.columns:
            self.demand_freq_map = grouped["demand_freq"].apply(list).to_dict()
        else:
            self.demand_freq_map = None
        
        self._build_demand_calibration(df, grouped)
        
        print(f"[Dataset] Loaded {self.n_skus} SKUs, {len(df)} records")
    
    def _build_demand_calibration(self, df, grouped):
        self._calib_map = {}
        alpha = 0.3
        for sku_id in self.sku_ids:
            n_days = self.n_days_map[sku_id]
            preds_arr = self.predicts_map[sku_id]
            seq = self.sales_map[sku_id]
            factors = []
            for d in range(n_days):
                pred = preds_arr[d]
                horizon = min(len(pred), n_days - d)
                p_val = sum(max(0, float(v)) for v in pred[:horizon])
                s_val = sum(max(0, seq[d + i]) for i in range(horizon))
                if p_val > 0.5:
                    w = alpha * (s_val / p_val) + (1 - alpha)
                else:
                    w = 1.0
                factors.append(max(0.5, min(w, 2.0)))
            self._calib_map[sku_id] = factors

    def get_demand_factor(self, sku_id: str, day_idx: int) -> float:
        if sku_id not in self._calib_map:
            return 1.0
        fl = self._calib_map[sku_id]
        if day_idx < len(fl):
            return fl[day_idx]
        return fl[-1] if fl else 1.0

    def get_sku_data(self, sku_id: str) -> Dict:
        """获取单个 SKU 的所有数据"""
        return {
            "sales": self.sales_map[sku_id],
            "predicts": self.predicts_map[sku_id],
            "leadtime": self.leadtime_map[sku_id],
            "pred_y": self.pred_y_map[sku_id],
            "initial_stock": self.initial_stock_map[sku_id],
            "static_features": self.static_features_map[sku_id],
            "n_days": self.n_days_map[sku_id],
            "actuals": self.actuals_map[sku_id] if self.actuals_map else None,
        }
    
    def get_sales(self, sku_id: str, day_idx: int) -> int:
        """获取某SKU某天的实际销量"""
        return self.sales_map[sku_id][day_idx]
    
    def get_leadtime(self, sku_id: str, day_idx: int) -> int:
        """获取某SKU某天的leadtime"""
        return self.leadtime_map[sku_id][day_idx]
    
    def get_predicts(self, sku_id: str, day_idx: int) -> List[float]:
        """获取某SKU某天的6天预测销量"""
        return self.predicts_map[sku_id][day_idx].tolist()
    
    def get_pred_y(self, sku_id: str, day_idx: int) -> float:
        """获取某SKU某天的LT天预测销量"""
        return self.pred_y_map[sku_id][day_idx]
    
    def get_actuals_future(self, sku_id: str, day_idx: int) -> List[int]:
        """获取某SKU某天开始的未来6天真实销量"""
        if self.actuals_map is None:
            return [0] * 6
        return self.actuals_map[sku_id][day_idx]
    
    def get_static_features(self, sku_id: str, day_idx: int) -> List[float]:
        """获取某SKU某天的静态特征"""
        features = self.static_features_map[sku_id][day_idx]
        # 展平嵌套列表
        flat_features = []
        for f in features:
            if isinstance(f, (list, tuple, np.ndarray)):
                flat_features.extend(f)
            else:
                flat_features.append(f)
        return flat_features
    
    def get_n_days(self, sku_id: str) -> int:
        """获取某SKU的总天数"""
        return self.n_days_map[sku_id]
    
    def get_avg_qty_7d(self, sku_id: str, day_idx: int) -> float:
        """获取某SKU某天的7天平均销量"""
        if self.avg_qty_7d_map is None:
            return 1.0
        return self.avg_qty_7d_map[sku_id][day_idx]
    
    def get_feature_by_name(self, sku_id: str, day_idx: int, feature_name: str) -> float:
        """按特征名获取单个静态特征值（避免展平后索引错位）"""
        if feature_name not in self.static_features:
            return 0.0
        feat_idx = self.static_features.index(feature_name)
        row = self.static_features_map[sku_id][day_idx]
        if feat_idx < len(row):
            val = row[feat_idx]
            if isinstance(val, (list, tuple, np.ndarray)):
                return float(val[0]) if len(val) > 0 else 0.0
            return float(val)
        return 0.0


def create_dataset(config: dict, split: str = "train") -> ReplenishmentDataset:
    """
    根据配置创建数据集
    
    Args:
        config: 配置字典
        split: "train" 或 "eval"
    """
    if split == "train":
        file_path = config["data"]["train_path"]
    else:
        file_path = config["data"]["eval_path"]
    
    static_features = config["env"]["state_features"]["static"]
    
    return ReplenishmentDataset(
        file_path=file_path,
        static_features=static_features,
    )
