import mlflow
import mlflow.models
import mlflow.pytorch
import os
from typing import Dict, Any, Union, List, Optional


class MLflowManager:
    def __init__(self, mlflow_host: str, experiment_name: str, run_name: str, enabled: bool = True):
        self.mlflow_host = mlflow_host
        self.enabled = enabled
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.active_run = None

        if not self.enabled:
            return

        try:
            if mlflow_host:
                mlflow.set_tracking_uri(mlflow_host)
            mlflow.set_experiment(experiment_name)
            self.active_run = mlflow.start_run(run_name=self.run_name)
        except Exception as e:
            print(f"警告: MLflow 初始化失败: {str(e)}")
            self.enabled = False

    def start_run(self) -> None:
        if not self.enabled:
            return

        try:
            if mlflow.active_run():
                mlflow.end_run()
            if self.mlflow_host:
                mlflow.set_tracking_uri(self.mlflow_host)
            self.active_run = mlflow.start_run(run_name=self.run_name)
        except Exception as e:
            print(f"警告: 启动 MLflow run 失败: {str(e)}")

    def end_run(self) -> None:
        if self.enabled and mlflow.active_run():
            mlflow.end_run()
            self.active_run = None

    def log_params(self, params: Dict[str, Any]) -> None:
        if not self.enabled:
            return

        for key, value in params.items():
            if not key.startswith("__") and isinstance(value, (int, float, str, bool)):
                mlflow.log_param(key, value)
            elif isinstance(value, list) and len(value) > 0:
                try:
                    mlflow.log_param(key, str(value))
                except:
                    pass

    def log_config(self, config) -> None:
        if not self.enabled:
            return

        params = {}
        for key, value in config.__dict__.items():
            if not key.startswith("__") and isinstance(value, (int, float, str, bool, list)):
                params[key] = value
        self.log_params(params)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        if self.enabled:
            mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if not self.enabled:
            return

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value, step=step)

    def log_dict_items_as_metrics(self, prefix: str, dict_obj: Dict, step: Optional[int] = None) -> None:
        if not self.enabled:
            return

        for key, value in dict_obj.items():
            metric_name = f"{prefix}_{key}"
            if isinstance(value, (int, float)):
                mlflow.log_metric(metric_name, value, step=step)

    def log_dataset(self, dataset_dir: str, dataset_name: str = "dataset", tags: dict[str, str] | None = None) -> None:
        if not self.enabled:
            return

        try:
            import numpy as np
            from mlflow.data import from_numpy  # type: ignore[import]

            arr = np.empty((0, 0))
            ds = from_numpy(arr, source=dataset_dir, name=dataset_name, digest="00000000")
            mlflow.log_input(ds, tags=tags)

            print(f"已记录数据集信息: {dataset_name}, 路径: {dataset_dir}")
        except Exception as e:
            print(f"警告: 记录数据集信息失败: {str(e)}")

    def log_pytorch_model(self, model, input, model_name: str = "model") -> None:
        if self.enabled:
            # mlflow.pytorch.log_model(model, model_name, input_example=input)
            # mlflow.pytorch.log_state_dict({"model_info": "x1"}, artifact_path="test_path")
            pass

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        if self.enabled and os.path.exists(local_path):
            # mlflow.log_artifact(local_path, artifact_path)
            pass

    def create_eval_run(self, eval_name: str) -> "MLflowManager":
        eval_manager = MLflowManager(self.mlflow_host, self.experiment_name, eval_name, self.enabled)
        eval_manager.start_run()
        return eval_manager

    def log_eval_metrics(
        self,
        task_name,
        params,
        action_dist_dict,
        metrics,
    ) -> None:
        if not self.enabled:
            return

        eval_mlflow = self.create_eval_run(f"{task_name}_evaluation")
        eval_mlflow.log_params(params)
        eval_mlflow.log_metrics(metrics)
        eval_mlflow.log_dict_items_as_metrics("action", action_dist_dict)


if __name__ == "__main__":
    mlflow_manager = MLflowManager("http://localhost:5000", "test_experiment", "test_run")
    mlflow_manager.log_params({"param1": 1, "param2": 2.5, "param3": "test"})
    mlflow_manager.log_metric("metric1", 0.95)
    mlflow_manager.log_dataset("path/to/dataset", "test_dataset")
    mlflow_manager.end_run()
