import pandas as pd
import numpy as np



stat_ls=[  "demand_freq","order_ratio_l3d_gap1","order_ratio_l5d_gap1", "order_ratio_l7d_gap1","order_ratio_l14d_gap1", "avg_daily_item_qty_lag1",
    "avg_daily_item_qty_modify_l3d_lag1",
    "avg_daily_item_qty_modify_l5d_lag1",
    "avg_daily_item_qty_modify_l7d_lag1",
    "avg_daily_item_qty_modify_l14d_lag1",
    "std_daily_item_qty_l3d_lag1",
    "std_daily_item_qty_l5d_lag1",
    "std_daily_item_qty_l7d_lag1",
    "std_daily_item_qty_l14d_lag1",
    "avg_daily_item_qty_3_minus_5",
    "avg_daily_item_qty_3_minus_7",
    "avg_daily_item_qty_3_minus_14"]
# pred_df_path="/home/work/apb-project/ais-deploy-demo-cache/ais-deploy-demo/outs/cache/rl_0415_dq2_new/tft_stream/simulation_65.v202504.parquet"
# static_df_path="/home/work/apb-project/ais-deploy-demo-cache/replenishment/data/20250708/tmp_60_dwd_cache_replenish_new_features_static_0415_20250707_sub1"
# result_df_path="/home/work/apb-project/ais-deploy-demo-cache/ais-deploy-demo/outs/cache/rl_0415_dq2_new/tft_stream/simulation_65.v202504.repl_feature1.parquet"
# static_df=pd.read_parquet(static_df_path)
# static_df["repl_features"] = [np.array(row) for row in static_df[stat_ls].values]
# static_df=static_df[["idx","ds","repl_features"]]
# # 将 datetime.date 转换为 Pandas 的 datetime64 类型
# static_df['ds'] = pd.to_datetime(static_df['ds'])
# # 使用向量化操作将日期转换为字符串格式
# static_df['ds'] = static_df['ds'].dt.strftime('%Y-%m-%d')
# #static_df['ds'] = static_df['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))
# pred_df=pd.read_parquet(pred_df_path)

# print(pred_df.shape)
# pred_df=pd.merge(pred_df,static_df,how="left",on=["idx","ds"])
# print(pred_df.repl_features.isnull().sum())
# default_val=[0 for i in range(len(stat_ls))]
# pred_df.fillna(default_val, inplace=True)
# print(pred_df.shape)
# pred_df.to_parquet(result_df_path)


import pandas as pd
import numpy as np

# pred_df_path="/home/work/apb-project/ais-deploy-demo-cache/ais-deploy-demo/outs/cache/rl_0415_dq2_new/tft_stream/simulation_65.v202505.parquet"
# static_df_path="/home/work/apb-project/ais-deploy-demo-cache/replenishment/data/20250708/tmp_60_dwd_cache_replenish_new_features_static_0415_20250707_sub1"
# result_df_path="/home/work/apb-project/ais-deploy-demo-cache/ais-deploy-demo/outs/cache/rl_0415_dq2_new/tft_stream/simulation_65.v202505.repl_feature1.parquet"

# static_df=pd.read_parquet(static_df_path)
# pred_df=pd.read_parquet(pred_df_path)

# pred_df=pred_df[pred_df.ds!='2025-06-01']
# pred_df=pd.merge(pred_df,static_df,how="left",on=["idx","ds"])
# pred_df.to_parquet(result_df_path)


# ###06数据
# pred_df_path="/home/work/apb-project/ais-deploy-demo-cache/ais-deploy-demo/outs/cache/rl_0415_dq2_new/tft_stream/simulation_65.v202506.parquet"
# static_df_path="/home/work/apb-project/ais-deploy-demo-cache/replenishment/data/20250708/tmp_60_dwd_cache_replenish_new_features_static_with_stock_0415_20250707_v1_sub2"
# result_df_path="/home/work/apb-project/ais-deploy-demo-cache/ais-deploy-demo/outs/cache/rl_0415_dq2_new/tft_stream/simulation_65.v202506.repl_feature_with_stock_v1.parquet"

# static_df=pd.read_parquet(static_df_path)
# pred_df=pd.read_parquet(pred_df_path)

# pred_df=pd.merge(pred_df,static_df,how="left",on=["idx","ds"])
# pred_df.to_parquet(result_df_path)


# ###05数据
# pred_df_path="/home/work/apb-project/ais-deploy-demo-cache/ais-deploy-demo/outs/cache/rl_0415_dq2_new/tft_stream/simulation_65.v202505.parquet"
# static_df_path="/home/work/apb-project/ais-deploy-demo-cache/replenishment/data/20250708/tmp_60_dwd_cache_replenish_new_features_static_with_stock_0415_20250707_v1_sub1"
# result_df_path="/home/work/apb-project/ais-deploy-demo-cache/ais-deploy-demo/outs/cache/rl_0415_dq2_new/tft_stream/simulation_65.v202505.repl_feature_with_stock_v1.parquet"

# static_df=pd.read_parquet(static_df_path)
# pred_df=pd.read_parquet(pred_df_path)

# pred_df=pd.merge(pred_df,static_df,how="left",on=["idx","ds"])
# pred_df.to_parquet(result_df_path)




# ###06数据
# pred_df_path="/home/work/apb-project/ais-deploy-demo-cache/ais-deploy-demo/outs/cache/rl_0415_dq2_new/tft_stream/simulation_50.v202506.parquet"
# static_df_path="/home/work/apb-project/ais-deploy-demo-cache/replenishment/data/20250708/tmp_60_dwd_cache_replenish_new_features_static_with_stock_0415_20250707_v1_sub2"
# result_df_path="/home/work/apb-project/ais-deploy-demo-cache/ais-deploy-demo/outs/cache/rl_0415_dq2_new/tft_stream/simulation_50.v202506.repl_feature_with_stock_v1.parquet"

# static_df=pd.read_parquet(static_df_path)
# pred_df=pd.read_parquet(pred_df_path)


# pred_df=pd.merge(pred_df,static_df,how="left",on=["idx","ds"])
# pred_df.to_parquet(result_df_path)


# ###05数据
# pred_df_path="/home/work/apb-project/ais-deploy-demo-cache/ais-deploy-demo/outs/cache/rl_0415_dq2_new/tft_stream/simulation_50.v202505.parquet"
# static_df_path="/home/work/apb-project/ais-deploy-demo-cache/replenishment/data/20250708/tmp_60_dwd_cache_replenish_new_features_static_with_stock_0415_20250707_v1_sub1"
# result_df_path="/home/work/apb-project/ais-deploy-demo-cache/ais-deploy-demo/outs/cache/rl_0415_dq2_new/tft_stream/simulation_50.v202505.repl_feature_with_stock_v1.parquet"

# static_df=pd.read_parquet(static_df_path)
# pred_df=pd.read_parquet(pred_df_path)

# pred_df=pd.merge(pred_df,static_df,how="left",on=["idx","ds"])
# pred_df.to_parquet(result_df_path)



##0728 合并特征数据
# ###06数据
# pred_df_path="/home/work/apb-project/ais-deploy-demo-cache/ais-deploy-demo/outs/cache/rl_test_cff_0415_dq2_new/tft_stream/simulation_65.v202506.parquet"
# static_df_path="/home/work/apb-project/ais-deploy-demo-cache/replenishment/data/20250718/tmp_60_dwd_cache_replenish_new_features_static_0415_20250728_sub2"
# result_df_path="/home/work/apb-project/ais-deploy-demo-cache/ais-deploy-demo/outs/cache/rl_test_cff_0415_dq2_new/tft_stream/simulation_65.v202506.repl_feature_v1_0728.parquet"

# static_df=pd.read_parquet(static_df_path)
# pred_df=pd.read_parquet(pred_df_path)

# print("data_loaded")
# pred_df=pd.merge(pred_df,static_df,how="left",on=["idx","ds"])
# print("data_merge")
# pred_df.to_parquet(result_df_path)


###05数据
pred_df_path="/home/work/apb-project/ais-deploy-demo-cache/ais-deploy-demo/outs/cache/rl_test_cff_0415_dq2_new/tft_stream/simulation_65.v202505.parquet"
static_df_path="/home/work/apb-project/ais-deploy-demo-cache/replenishment_wb/replenishment_ppo/data_simu/0911_stat_with_stock_level"
result_df_path="/home/work/apb-project/ais-deploy-demo-cache/replenishment_wb/replenishment_ppo/data_simu/0911_simu_May_safe.parquet"

static_df=pd.read_parquet(static_df_path)
print("stat_data_loaded")
pred_df=pd.read_parquet(pred_df_path)
print("pred_data_loaded")
pred_df=pd.merge(pred_df,static_df,how="left",on=["idx","ds"])
print("data_merge")
static_df = ''
pred_df.to_parquet(result_df_path)
print("data_save")
