import pandas as pd
import numpy as np

def flatten_row(row):
    """
    把 Series 的一行展平成一维 list。
    如果某个元素是 list/tuple/ndarray，就把它拆开；否则保持原值。
    """
    flat = []
    for v in row:
        if isinstance(v, (list, tuple, np.ndarray)):
            flat.extend(v)          # 展开
        else:
            flat.append(v)          # 单个值
    return flat



class Datasets:
    def __init__(self, file_path: str, state_static_columns: list, rank: int = 0, world_size: int = 1) -> None:
        self.file_path = file_path

        # TODO 通过sku信息加载部分结果, 不要全量加载到内存
        if file_path.endswith(".csv"):
            input_data = pd.read_csv(file_path)
        else:
            input_data = pd.read_parquet(file_path)
        if type(input_data["predicted_demand"].tolist()[0]) == str:
            input_data["predicted_demand"] = input_data["predicted_demand"].apply(eval)
        # input_data = input_data[input_data["idx"] == "257082622036-4894"]

        input_data = input_data.sort_values(by=["idx", "date"])
        self.total_sales = input_data["actual"].sum()
        unique_idx = input_data["idx"].unique()
        if world_size > 1:
            # 计算该进程应处理的idx范围
            chunk_size = len(unique_idx) // world_size
            start_idx = rank * chunk_size
            end_idx = start_idx + chunk_size if rank < world_size - 1 else len(unique_idx)
            process_idx_list = unique_idx[start_idx:end_idx]
            input_data = input_data[input_data["idx"].isin(process_idx_list)]
            input_data.drop_duplicates(subset=["idx", "date"], inplace=True)

        idx_data = input_data.groupby("idx")
        self.sales_map = idx_data["actual"].apply(list).to_dict()

        # print(f"pre_demand_type:{type(input_data['predicted_demand'].iloc[0])}❗️")
        # print(f"pre_demand_type:{type(input_data['predicted_demand'].iloc[0][0])}❗️")

        self.predicts_map = idx_data["predicted_demand"].apply(lambda s: np.array(s.tolist())).to_dict()
        self.state_static_columns = state_static_columns
        # self.static_state_map = idx_data.apply(lambda x: x[self.state_static_columns].values.tolist()).to_dict()
        self.static_state_map = (idx_data.apply(lambda sub_df: [flatten_row(r) for _, r in sub_df[self.state_static_columns].iterrows()]).to_dict()) # 解开列表
        self.end_date_map = idx_data["date"].count().to_dict()
        self.leadtime_map = idx_data["leadtime"].apply(list).to_dict()
        self.predict_leadtime_day = idx_data["pred_y"].apply(list).to_dict() # leadtime天的预测销量
        self.initial_stock_map = idx_data["initial_stock"].max().to_dict()
        self.avg_item_qty_7d_map = idx_data["avg_daily_item_qty_l7d"].apply(list).to_dict()  # 过去7天平均销量
        self.dq_map = idx_data["demand_freq"].apply(list).to_dict()
        self.order_ratio_7d_map = idx_data["order_ratio_l7d"].apply(list).to_dict()
        self.order_ratio_14d_map = idx_data["order_ratio_l14d"].apply(list).to_dict() # 这里先注释掉，跑base模型

        actual_cols = ['actual_0', 'actual_1', 'actual_2', 'actual_3', 'actual_4', 'actual_5']
        self.actuals_map = idx_data.apply(
            lambda x: x[actual_cols].values.tolist()
        ).to_dict()

        date_idx_data = input_data.groupby(["date", "idx"])
        self.sku_id_ls = date_idx_data["pred_y"].sum().unstack().columns.tolist()
        self.predicted_demand = date_idx_data["pred_y"].sum().unstack().fillna(0).values
        print(f"Process {rank}/{world_size}: #############gen data done#############")

    def get_initial_stock_map(self):
        return self.initial_stock_map

    def get_end_date_map(self):
        return self.end_date_map

    def sku_ids(self) -> list:
        return self.sku_id_ls

    def get_static_state_map(self):
        return self.static_state_map

    def sku_lead_time(self, sku_id: int, day_idx: int) -> int:
        return self.leadtime_map.get(sku_id, [1])[day_idx]

    def range_lead_time(self, day_st, day_ed, sku_id) -> list:
        return self.leadtime_map[sku_id][day_st:day_ed]

    def range_prdicts(self, day_st, day_ed, sku_id) -> list:
        return self.predicts_map[sku_id][day_st:day_ed]

    def get_predicts(self, day_idx: int, sku_id: int) -> list:
        return self.predicts_map[sku_id][day_idx].tolist()

    def get_leadtime_predict(self, day_idx: int, sku_id: int) -> list:
        return self.predict_leadtime_day[sku_id][day_idx]

    def range_sales(self, day_st, day_ed, sku_id) -> list:
        return self.sales_map[sku_id][day_st:day_ed]

    def get_sales(self, day_idx: int, sku_id: int):
        return self.sales_map[sku_id][day_idx]
