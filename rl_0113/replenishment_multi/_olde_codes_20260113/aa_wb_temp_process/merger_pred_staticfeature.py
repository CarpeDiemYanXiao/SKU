import pandas as pd

def merge_prediction_and_static_data(pred_df_path: str, static_df_path: str, result_df_path: str):
    """
    合并预测数据和静态特征数据，只保留两个DataFrame中idx和ds都能匹配的行
    
    Args:
        pred_df_path: 预测数据文件路径
        static_df_path: 静态特征数据文件路径  
        result_df_path: 输出结果文件路径
    """
    print("开始加载静态特征数据...")
    static_df = pd.read_parquet(static_df_path)
    print(f"静态特征数据加载完成，共 {len(static_df)} 行")
    
    print("开始加载预测数据...")
    pred_df = pd.read_parquet(pred_df_path)
    print(f"预测数据加载完成，共 {len(pred_df)} 行")
    
    print("开始合并数据...")
    # 使用inner连接，只保留两个DataFrame中idx和ds都能匹配的行
    merged_df = pd.merge(pred_df, static_df, how="inner", on=["idx", "ds"])
    print(f"数据合并完成，合并后共 {len(merged_df)} 行")
    
    print("开始保存结果...")
    static_df = 0
    pred_df = 0
    merged_df.to_parquet(result_df_path)
    print(f"结果已保存到: {result_df_path}")

    print("开始检查None值...")
    none_check = merged_df.isnull().sum()
    columns_with_none = none_check[none_check > 0]
    
    if len(columns_with_none) > 0:
        print("⚠️  发现包含None值的列:")
        for col, count in columns_with_none.items():
            print(f"  - {col}: {count} 个None值")
        
        # 显示包含None值的行数
        rows_with_none = merged_df.isnull().any(axis=1).sum()
        print(f"⚠️  共有 {rows_with_none} 行包含None值")
        
        # 可以选择是否继续保存
        print("是否继续保存包含None值的数据？(y/n)")
        # 这里可以添加用户交互逻辑，或者直接继续
    else:
        print("✅ 合并后的数据中没有发现None值")
    
    return merged_df


# 示例使用代码
if __name__ == "__main__":
    pred_df_path = "/home/work/apb-project/ais-deploy-demo-cache/ais-deploy-demo/outs/cache/rl_0415_dq2_new/tft_stream/simulation_65.v202505.parquet"
    static_df_path = "/home/work/apb-project/ais-deploy-demo-cache/replenishment_wb/simulation_data/20250728_static_feature_with_begin_stock_factor"
    result_df_path = "/home/work/apb-project/ais-deploy-demo-cache/replenishment_wb/simulation_data/20250728_whole_data_May_with_stock_factor.parquet"
    
    # 使用inner连接合并数据
    result_df = merge_prediction_and_static_data(pred_df_path, static_df_path, result_df_path)