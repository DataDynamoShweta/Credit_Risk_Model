"""Minimal MLflow registry adapter using model version aliases (not deprecated stages).

MLflow is imported only when used. The tracking URI comes from the argument, then from
the ``MLFLOW_TRACKING_URI`` setting.
"""


class ModelRegistry:
    def __init__(self, tracking_uri: str | None = None, client=None):
        if tracking_uri is None:
            from credit_risk_model.config import get_settings
            tracking_uri = get_settings().mlflow_tracking_uri
        self.tracking_uri = tracking_uri
        self._injected_client = client

    def _client(self):
        if self._injected_client is not None:
            return self._injected_client
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
        except ImportError as exc:
            raise RuntimeError("Install the 'registry' extra to use MLflow") from exc
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        return MlflowClient()

    def version_for_alias(self, model_name: str, alias: str = "champion"):
        """Return the model version currently pointed to by ``alias``."""
        try:
            return self._client().get_model_version_by_alias(model_name, alias)
        except Exception as exc:
            raise LookupError(f"No version of {model_name} has alias {alias!r}") from exc

    def set_alias(self, model_name: str, version: str, alias: str = "champion"):
        """Point ``alias`` at ``version`` (replaces the deprecated stage transition)."""
        self._client().set_registered_model_alias(model_name, alias, str(version))

    def delete_alias(self, model_name: str, alias: str):
        self._client().delete_registered_model_alias(model_name, alias)
