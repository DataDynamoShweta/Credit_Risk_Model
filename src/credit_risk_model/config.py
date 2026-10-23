"""Centralised configuration loaded from environment variables."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8",
        extra="ignore", case_sensitive=False,
    )

    governance_dir: Path = Field(default=PROJECT_ROOT / "governance")
    inventory_path: Path = Field(
        default=PROJECT_ROOT / "governance" / "model_inventory.yaml")
    audit_trail_dir: Path = Field(
        default=PROJECT_ROOT / "governance" / "audit_trail")
    pma_log_path: Path = Field(
        default=PROJECT_ROOT / "governance" / "pma_log.csv")
    artifacts_dir: Path = Field(default=PROJECT_ROOT / "artifacts")
    reports_dir: Path = Field(default=PROJECT_ROOT / "reports")

    allow_unreviewed_model: bool = Field(default=False)
    environment: str = Field(default="local")

    psi_investigate: float = 0.10
    psi_alert: float = 0.25
    csi_investigate: float = 0.10
    csi_alert: float = 0.25
    decision_threshold: float = 0.20

    audit_retention_days: int = 90
    audit_max_page_size: int = 500

    mlflow_tracking_uri: str | None = None

    def ensure_dirs(self) -> None:
        for path in (
            self.governance_dir,
            self.audit_trail_dir,
            self.audit_trail_dir / "decisions",
            self.audit_trail_dir / "changes",
            self.audit_trail_dir / "monitoring",
            self.artifacts_dir,
            self.reports_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings


def reset_settings_cache() -> None:
    get_settings.cache_clear()
