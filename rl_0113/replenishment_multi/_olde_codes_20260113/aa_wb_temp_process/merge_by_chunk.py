import pandas as pd
import os
import gc


def pandas_merge_with_chunks(pred_path, static_dir, result_path, chunk_size=50000):
    """
    使用pandas分块处理，更稳定
    """
    print("读取pred文件...")
    pred_df = pd.read_parquet(pred_path)
    print(f"pred数据: {len(pred_df)} 行")
    
    # 读取static目录中的所有文件
    static_files = []
    for file in os.listdir(static_dir):
        file_path = os.path.join(static_dir, file)
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(4)
                    if header == b'PAR1':
                        static_files.append(file_path)
            except Exception:
                continue
    
    print(f"找到 {len(static_files)} 个parquet文件")
    
    # 分块处理每个static文件
    first_chunk = True
    total_merged = 0
    
    for file_idx, static_file in enumerate(static_files):
        print(f"\n处理文件 {file_idx + 1}/{len(static_files)}: {os.path.basename(static_file)}")
        
        # 分块读取static文件
        for chunk_num, static_chunk in enumerate(pd.read_parquet(static_file, chunksize=chunk_size)):
            print(f"  处理chunk {chunk_num + 1}...")
            
            # 合并
            merged_chunk = pd.merge(pred_df, static_chunk, how="inner", on=["idx", "ds"])
            
            if len(merged_chunk) > 0:
                print(f"  合并后 {len(merged_chunk)} 行")
                total_merged += len(merged_chunk)
                
                # 保存
                if first_chunk:
                    merged_chunk.to_parquet(result_path, index=False)
                    first_chunk = False
                else:
                    merged_chunk.to_parquet(result_path, index=False, append=True)
            
            # 清理内存
            del static_chunk, merged_chunk
            gc.collect()
    
    print(f"\n合并完成，总共 {total_merged} 行")
    return total_merged


# 使用示例
if __name__ == "__main__":
    pred_df_path = "/home/work/apb-project/ais-deploy-demo-cache/ais-deploy-demo/outs/cache/rl_0415_dq2_new/tft_stream/simulation_65.v202506.parquet"
    static_df_path = "/home/work/apb-project/ais-deploy-demo-cache/replenishment_wb/simulation_data/20250728_static_feature_with_begin_stock_factor"
    result_df_path = "/home/work/apb-project/ais-deploy-demo-cache/replenishment_wb/simulation_data/20250728_whole_data_June_with_stock_factor111.parquet"
    
    # 使用pandas分块处理
    result = pandas_merge_with_chunks(pred_df_path, static_df_path, result_df_path)