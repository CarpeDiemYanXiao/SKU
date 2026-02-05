from typing import Optional
import mlflow
import psutil
import os
import json
from abc import ABC, abstractmethod
import logging
import time

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, model, data) -> dict:
        pass


class OfflineEvaluator(BaseEvaluator):
    def evaluate(self, model, data) -> dict:
        logger.info("正在执行离线评估 (占位符)...")
        return {"offline_accuracy": 0.9, "offline_loss": 0.1}


class SimulationEvaluator(BaseEvaluator):
    def evaluate(self, model, data) -> dict:
        logger.info("正在执行仿真评估 (占位符)...")
        return {"sim_avg_reward": 100.0, "sim_episodes_completed": 10}


class MLOpsUtils:
    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None, run_name: Optional[str] = None):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.run_name = run_name
        self._active_run = None
        self._run_start_time = None

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            logger.info(f"Experiment '{self.experiment_name}' not found. Creating new experiment.")
            mlflow.create_experiment(self.experiment_name)
        mlflow.set_experiment(self.experiment_name)

        logger.info(f"MLflow experiment 设置为 '{self.experiment_name}'. Tracking URI: '{mlflow.get_tracking_uri()}'")

    def _get_system_metrics(self, prefix="system_") -> dict:
        metrics = {}
        try:
            metrics[f"{prefix}cpu_percent_current"] = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            metrics[f"{prefix}memory_percent_used"] = mem.percent
            metrics[f"{prefix}memory_gb_used"] = round(mem.used / (1024**3), 3)

            try:
                disk_path = os.getcwd()
                disk = psutil.disk_usage(disk_path)
                metrics[f"{prefix}disk_percent_used"] = disk.percent
                metrics[f"{prefix}disk_gb_used"] = round(disk.used / (1024**3), 3)
            except Exception as e:
                logger.warning(f"无法获取磁盘使用情况: {e}")

        except Exception as e:
            logger.error(f"收集系统指标时出错: {e}")
        return metrics

    def start_run(self, run_name: Optional[str] = None, nested: bool = False, tags: Optional[dict] = None):
        if self._active_run and not nested:
            logger.warning("已存在活动的 run。要启动新的 run，请先结束当前的 run 或使用 nested=True。")
            return self._active_run

        run_to_start_with = run_name or self.run_name
        self._active_run = mlflow.start_run(run_name=run_to_start_with, nested=nested, tags=tags)
        self._run_start_time = time.time()
        run_id = self._active_run.info.run_id
        actual_run_name = self._active_run.info.run_name
        logger.info(f"MLflow run '{actual_run_name}' (ID: {run_id}) 已启动。")

        initial_sys_metrics = self._get_system_metrics(prefix="system_initial_")
        if initial_sys_metrics:
            self.log_metrics(initial_sys_metrics)
        return self._active_run

    def end_run(self, status: str = "FINISHED"):
        if self._active_run:
            run_id = self._active_run.info.run_id
            run_name = self._active_run.info.run_name

            final_sys_metrics = self._get_system_metrics(prefix="system_final_")
            if self._run_start_time:
                run_duration_seconds = time.time() - self._run_start_time
                final_sys_metrics["system_run_duration_seconds"] = round(run_duration_seconds, 2)
            if final_sys_metrics:
                self.log_metrics(final_sys_metrics)

            mlflow.end_run(status=status)
            logger.info(f"MLflow run '{run_name}' (ID: {run_id}) 已结束，状态为 '{status}'.")
            self._active_run = None
            self._run_start_time = None
        else:
            logger.warning("没有活动的 MLflow run 可以结束。")

    def __enter__(self):
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "FAILED" if exc_type else "FINISHED"
        self.end_run(status=status)

    def _ensure_active_run(self, default_run_name_suffix: str = "generic_run"):
        if not self._active_run:
            fallback_run_name = self.run_name or f"auto_{default_run_name_suffix}"
            logger.warning(f"没有活动的 run。正在启动一个名为 '{fallback_run_name}' 的默认 run。")
            self.start_run(run_name=fallback_run_name)

    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        self._ensure_active_run("metrics_run")
        mlflow.log_metrics(metrics, step=step)
        logger.info(f"已记录指标: {metrics}" + (f" (step: {step})" if step is not None else ""))

    def log_model_metrics(self, metrics: dict, step: Optional[int] = None):
        self.log_metrics(metrics, step=step)

    def log_params(self, params: dict):
        self._ensure_active_run("params_run")
        mlflow.log_params(params)
        logger.info(f"已记录参数: {params}")

    def log_dataset_info(self, dataset_name: str, dataset_details: Optional[dict] = None):
        self._ensure_active_run("dataset_info_run")

        mlflow.set_tag("dataset_name", dataset_name)
        logger.info(f"已记录数据集名称为 tag: {dataset_name}")
        if dataset_details:
            mlflow.log_dict(dataset_details, "dataset_details.json")
            logger.info(f"已将数据集详细信息记录到 dataset_details.json")

    def log_model_info(self, model_info: dict, prompt_keys: Optional[list] = None):
        self._ensure_active_run("model_info_run")

        params_to_log = {}
        prompt_keys = prompt_keys or []

        for key, value in model_info.items():
            sanitized_key = "".join(c if c.isalnum() or c in ["_", "-", ".", "/"] else "_" for c in key)

            if key in prompt_keys:
                if isinstance(value, str):
                    artifact_file_name = "".join(c if c.isalnum() or c in ["_", "-"] else "_" for c in key)
                    mlflow.log_text(value, artifact_file=f"prompts/{artifact_file_name}.txt")
                    logger.info(f"已将 prompt '{key}' 记录到 artifacts/prompts/{artifact_file_name}.txt")
                else:
                    logger.warning(f"prompt 键 '{key}' 的值不是字符串。跳过其文本 artifact 记录。")
            elif isinstance(value, (str, int, float, bool)):
                if isinstance(value, str) and len(value) > 250:
                    logger.warning(
                        f"参数 '{sanitized_key}' 的值过长 ({len(value)} chars)。将作为 model_info.json 的一部分存储，而不是直接参数。"
                    )
                else:
                    params_to_log[sanitized_key] = value

        if params_to_log:
            self.log_params(params_to_log)

        mlflow.log_dict(model_info, "model_info.json")
        logger.info("已将完整的 model_info 记录到 model_info.json")

    def log_and_register_model(
        self, model, model_flavor: str, artifact_path: str, registered_model_name: Optional[str] = None, **kwargs
    ):
        self._ensure_active_run("model_log_run")

        log_model_module = getattr(mlflow, model_flavor, None)
        if not log_model_module or not hasattr(log_model_module, "log_model"):
            raise ValueError(f"不支持的模型 flavor: {model_flavor}。未找到 'mlflow.{model_flavor}.log_model'。")

        log_model_func = getattr(log_model_module, "log_model")

        model_info_obj = log_model_func(
            model, artifact_path=artifact_path, registered_model_name=registered_model_name, **kwargs
        )
        model_uri = model_info_obj.model_uri
        logger.info(f"已使用 flavor '{model_flavor}' 将模型记录到 '{model_uri}'。")
        if registered_model_name:
            logger.info(f"模型已注册为 '{registered_model_name}'。")
        return model_uri

    def run_evaluation(self, evaluator: BaseEvaluator, model, data, metrics_prefix: str = "eval_") -> dict:
        self._ensure_active_run("evaluation_run")

        logger.info(f"正在使用 {type(evaluator).__name__} 执行评估...")
        metrics = evaluator.evaluate(model, data)

        prefixed_metrics = {f"{metrics_prefix}{k}": v for k, v in metrics.items()}
        self.log_metrics(prefixed_metrics)
        logger.info(f"已记录评估指标: {prefixed_metrics}")
        return prefixed_metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    mlflow_tracking_uri = "http://127.0.0.1:5000"

    mlops = MLOpsUtils(
        experiment_name="ReplenishmentAlgoDev", tracking_uri=mlflow_tracking_uri, run_name="DailyTrainingRun"
    )

    with mlops:
        mlops.log_params({"algorithm_version": "1.2.3", "max_iterations": 1000})

        for epoch in range(5):
            loss = 1.0 / (epoch + 1)
            accuracy = 0.7 + epoch * 0.05
            mlops.log_model_metrics({"epoch_loss": loss, "epoch_accuracy": accuracy}, step=epoch)
            time.sleep(0.1)

        mlops.log_dataset_info(
            dataset_name="StoreSales_Jan2024",
            dataset_details={"source_table": "db.sales_raw", "filter_date": "2024-01-31", "num_rows": 1500000},
        )

        model_config = {
            "model_type": "GradientBoostingRegressor",
            "n_estimators": 200,
            "learning_rate": 0.05,
            "feature_set_version": "v3",
            "training_prompt": "Train a model to predict next week's sales for items, considering promotions and holidays.",
            "inference_prompt_template": "Predict sales for item {item_id} at store {store_id} for date {date}.",
        }
        mlops.log_model_info(model_config, prompt_keys=["training_prompt", "inference_prompt_template"])

        class MySklearnModel:
            def __init__(self):
                self.coef_ = [0.1, 0.2]

            def predict(self, X):
                return [sum(x_row) for x_row in X]

        trained_model = MySklearnModel()

        try:
            import sklearn.dummy

            dummy_regressor = sklearn.dummy.DummyRegressor()

            mlops.log_and_register_model(
                dummy_regressor,
                model_flavor="sklearn",
                artifact_path="sales_predictor_sklearn",
                registered_model_name="SalesPredictorSklearn",
            )
        except ImportError:
            logger.warning("scikit-learn 未安装。跳过 sklearn 模型记录示例。")
            with open("generic_model_placeholder.txt", "w") as f:
                f.write("This is a placeholder for a model when its flavor's library is not installed.")
            mlflow.log_artifact("generic_model_placeholder.txt", "generic_model_files")
            os.remove("generic_model_placeholder.txt")

        eval_data_offline = {"features": [[1, 1], [2, 2], [3, 3]], "labels": [2, 4, 6]}
        eval_data_sim = {"environment_config": "config_A.yaml", "num_episodes": 50}

        offline_eval = OfflineEvaluator()
        mlops.run_evaluation(offline_eval, trained_model, eval_data_offline, metrics_prefix="main_offline_")

        sim_eval = SimulationEvaluator()
        mlops.run_evaluation(sim_eval, trained_model, eval_data_sim, metrics_prefix="main_sim_")

    logger.info(f"示例脚本完成。请检查 MLflow UI 中的 experiment '{mlops.experiment_name}'。")
