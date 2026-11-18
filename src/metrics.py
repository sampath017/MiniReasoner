from contextlib import contextmanager
from time import time_ns

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Metric


def _config_params(config: dict, prefix: str = "") -> dict:
    params = {}
    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            params.update(_config_params(value, name))
        else:
            params[name] = value
    return params


class TrainingRun:
    def __init__(self, client: MlflowClient, run):
        self._client = client
        self.id = run.info.run_id
        self.last_step = int(run.data.metrics.get("step", -1))
        self._pending: list[Metric] = []

    def log(self, metrics: dict, *, step: int) -> None:
        """Log one optimizer step, skipping steps already recorded before a restart."""
        if step <= self.last_step:
            return
        timestamp = time_ns() // 1_000_000
        for key, value in metrics.items():
            if key != "step" and value is not None:
                self._pending.append(Metric(key, float(value), timestamp, step))
        self._pending.append(Metric("step", float(step), timestamp, step))
        self.last_step = step
        # Normally two metrics per step. Send validation points before their checkpoint.
        if len(self._pending) >= 64 or "val_loss" in metrics:
            self.flush()

    def flush(self) -> None:
        if self._pending:
            self._client.log_batch(self.id, metrics=self._pending, synchronous=True)
            self._pending.clear()

    def set_tag(self, key: str, value: str) -> None:
        self._client.set_tag(self.id, key, value, synchronous=True)


@contextmanager
def track_run(tracking_uri: str, experiment_name: str, config: dict, *, run_id: str | None = None):
    """Track training metrics by optimizer step. Model checkpoints are stored separately."""
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_id=run_id) as active:
        if run_id is None:
            mlflow.log_params(_config_params(config), synchronous=True)
        run = TrainingRun(MlflowClient(tracking_uri=tracking_uri), active)
        try:
            yield run
        finally:
            run.flush()
