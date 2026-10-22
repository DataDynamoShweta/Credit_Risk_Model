"""Minimal MLflow registry adapter; imports MLflow only when used."""
import os


class ModelRegistry:
    def __init__(self, tracking_uri: str | None = None):
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")

    def _client(self):
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
        except ImportError as exc:
            raise RuntimeError("Install the 'registry' extra to use MLflow") from exc
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        return MlflowClient()

    def latest_version(self, model_name: str, stage: str | None = None):
        client = self._client()
        versions = client.get_latest_versions(model_name, stages=[stage] if stage else None)
        if not versions:
            raise LookupError(f"No registered versions for {model_name}")
        return max(versions, key=lambda v: int(v.version))

    def transition(self, model_name: str, version: str, stage: str, *, archive_existing: bool = True):
        return self._client().transition_model_version_stage(
            name=model_name, version=version, stage=stage,
            archive_existing_versions=archive_existing)
