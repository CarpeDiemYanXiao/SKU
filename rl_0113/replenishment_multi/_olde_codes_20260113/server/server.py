import yaml
import typing
import os
import glob
from typing import Dict
from oneservice import BaseProcessor, MonitorTimer, ReqParams, ResParams, server
from predictor import Predictor

num_threads = os.environ.get("NUM_THREADS", "4")
num_threads = int(num_threads) # 线程数量


class Processor(BaseProcessor):

    def __init__(self, config_file: str,region):
        with open(config_file, "r") as f:
            config = typing.cast(Dict, yaml.safe_load(f))
        predictor = config["predictors"]["default"]
        gunicorn = config["gunicorn"]
        model_config = config["model"]
        model_repo_dir = predictor["model_repo_dir"]

        versions = os.listdir(os.path.join(model_repo_dir,region))

        self.models = {}
        for version in versions:
            fm_path = os.path.join(model_repo_dir,region, version, "model.json")
            mo_path = os.path.join(model_repo_dir, region, version, "model.onnx")
            print("mo_path is ", mo_path)
            mo_path = glob.glob(mo_path)[0] # 返回更准确的路径
            print(f"fm_path: {fm_path}")
            print(f"mo_path: {mo_path}")
            self.models[version] = Predictor(mo_path, model_config, fm_path, gunicorn["threads"])

    def __call__(self, req_params: ReqParams, res_params: ResParams) -> None:
        # req_params:传入接口，res_params：传出接口
        if req_params.data is None:
            msg = "data in request is empty"
            print(msg)
            res_params.err_msg = msg
            res_params.err_num = 1
            return

        with MonitorTimer("infer"):
            params = req_params.data["input"]
            version = req_params.data["ModelVersion"]
            trt_features = self.models[version].predict(params) # 推演

        res_params.result = {"output": trt_features, "ModelVersion": version}
        return


# 一个model_repo就用一个
config_file = os.environ.get("MODEL_CONFIG_PATH", "./conf/config.yaml")
region = os.environ.get("REGION", "id")
app = server.app
app.register(Processor(config_file=config_file,region=region)) # 注册模型

if __name__ == "__main__":
    print("启动服务")
    server.app.run(host="127.0.0.1", port=80, threaded=True) # 监听端口，查看结果
