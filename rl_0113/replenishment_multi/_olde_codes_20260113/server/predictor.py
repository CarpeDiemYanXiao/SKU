import json
import importlib
import os
import torch
import torch.nn.functional as F
import numpy as np
import onnxruntime as ort
import sys
import pandas as pd
from datetime import datetime, timedelta
from oneservice import AIPBasePredictor

def read_feature_map(feature_map_path: str) -> dict:
    with open(feature_map_path, "r") as f:
        std_dictionary = json.load(f)
    return std_dictionary

def parse_feature_map(feature_map_path) -> tuple:

    std_dictionary = read_feature_map(feature_map_path)
    multiplier_ls = std_dictionary["action_ls"]
    task_name = std_dictionary["task_name"]
    use_state_norm = std_dictionary["use_state_norm"]
    columns = std_dictionary["columns"]
    state_dim = std_dictionary["state_dim"]
    state_norm_mean = std_dictionary["state_norm_mean"]
    state_norm_std = std_dictionary["state_norm_std"]
    # todo:引入类别变量后，需要对cat和con分别进行处理
    # cat_cols = std_dictionary["cat_cols"]
    # con_cols = std_dictionary["con_cols"]
    #multiplier_map = {i:multiplier_ls[i] for i in range(len(multiplier_ls))}
    return multiplier_ls, task_name, use_state_norm, columns, state_dim, state_norm_mean, state_norm_std



class Predictor(AIPBasePredictor):
    def __init__(self, model_path, model_config, feature_map_path, num_threads=10):
        # todo: 加载任务的task_name，不同task_name加入不同的策略
        # 但现在取task_name的方式比较冗余，可以在parse_feature_map中直接读取：
        self.feature_map = read_feature_map(feature_map_path)
        # 从config中加载所需的实验设置
        self.multiplier_ls, self.task_name, self.use_state_norm, self.columns, self.state_dim, self.state_norm_mean, self.state_norm_std = parse_feature_map(
            feature_map_path)
        print(
            f'-------task_name:{self.task_name}-------use_state_norm:{self.use_state_norm}-------columns:{self.columns}')
        try:
            self._check_state_norm_param(self.use_state_norm, self.state_dim, self.state_norm_mean, self.state_norm_std)
        except (ValueError, TypeError, KeyError) as e:
            # Log error and re-raise exception
            print(f"Input data format error: {str(e)}")
            raise ValueError(f"Input data format error: {str(e)}")
        if self.use_state_norm:
            self.state_norm_mean_np = np.array(self.state_norm_mean)
            self.state_norm_std_np = np.array(self.state_norm_std)
        # 创建 SessionOptions 对象
        sess_options = ort.SessionOptions()
        # 设置单个操作内部使用的线程数
        sess_options.intra_op_num_threads = num_threads
        # 设置是否顺序执行操作图内部的算子，还是并行执行
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.sess = ort.InferenceSession(model_path, sess_options=sess_options)
    
    def _check_state_norm_param(self, use_state_norm: int, state_dim: int, state_norm_mean: list, state_norm_std: list) -> None:
        if use_state_norm:
            if len(state_norm_mean) != state_dim or len(state_norm_std) != state_dim:
                raise ValueError("State normalization parameter is not suitable")
    
    def _check_input_features(self, inputs_dict: dict) -> None:
        """Check if input features are complete"""
        if not inputs_dict:
            raise ValueError("Input data is empty")
            
        required_fields = {"features", "info"}
        required_features = set(self.columns)
        
        for idx, input_data in enumerate(inputs_dict):
            # Check required top-level fields
            missing_fields = required_fields - set(input_data.keys())
            if missing_fields:
                raise ValueError(f"Data at index {idx} is missing required fields: {missing_fields}")
            
            # Check feature fields
            features = input_data.get("features", {})
            missing_features = required_features - set(features.keys())
            if missing_features:
                raise ValueError(f"Data at index {idx} is missing required features: {missing_features}")

    def _validate_input_format(self, inputs_dict):
        """Validate input data format"""
        if not isinstance(inputs_dict, (list, tuple)):
            raise ValueError("Input data must be a list or tuple")
        
        for idx, item in enumerate(inputs_dict):
            if not isinstance(item, dict):
                raise ValueError(f"Data at index {idx} must be a dictionary")
            
            # Check features field format
            if 'features' not in item or not isinstance(item['features'], dict):
                raise ValueError(f"Data at index {idx}: 'features' field must be a dictionary")
            
            # Check info field format
            if 'info' not in item or not isinstance(item['info'], dict):
                raise ValueError(f"Data at index {idx}: 'info' field must be a dictionary")
            
            # Check unique_id field
            if 'unique_id' not in item['info']:
                raise ValueError(f"Data at index {idx}: missing 'unique_id' field in info")

    def predict(self, inputs_dict: dict) -> list:
        try:
            # 验证输入数据格式
            self._validate_input_format(inputs_dict)
            
            # 检查输入特征
            self._check_input_features(inputs_dict)

            with torch.no_grad():
                inputs = {}
                inputs["input"] = np.array([[diction["features"][f"{name}"] for idx, name in enumerate(
                                    self.columns)] for diction in inputs_dict])
                if self.use_state_norm:
                    inputs["input"] = (inputs["input"] - self.state_norm_mean_np)/(self.state_norm_std_np+1e-8)
                out = self.sess.run(
                    None,
                    {"x": inputs["input"].astype(np.float32)},
                )[0]
                out_prob = out
                print(f"out_prob={out_prob}")
                out = np.argmax(out, axis=1, keepdims=True)
                # multiplier_mapping = np.vectorize(lambda x: self.multiplier_ls[x])
                # # 将index映射回multiplier
                # out = multiplier_mapping(out)
                # out = out.tolist()
                out = [self.multiplier_ls[idx[0]] for idx in out]
                out_prob = out_prob.tolist()

                result = []
                for idx, diction in enumerate(inputs_dict):
                    result_tmp = {}
                    for k, v in diction.items():
                        if k in ["features","info"]:
                            continue
                        result_tmp[k] = v
                    result_tmp['unique_id'] = diction['info']['unique_id']
                    result_tmp["output"] = out[idx]
                    out_dict = {}
                    out_dict["multiplier"] = out[idx][0]
                    out_dict["probability"] = out_prob[idx]
                    result_tmp["output"] = out_dict
                    result += [result_tmp]
            return result

        except (ValueError, TypeError, KeyError) as e:
            # Log error and re-raise exception
            print(f"Input data format error: {str(e)}")
            raise ValueError(f"Input data format error: {str(e)}")
        except Exception as e:
            # Handle other unexpected exceptions
            print(f"Prediction process error: {str(e)}")
            raise RuntimeError(f"Prediction process error: {str(e)}")