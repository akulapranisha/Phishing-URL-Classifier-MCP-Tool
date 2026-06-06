"""Application configuration via environment variables."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven settings for training and serving."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_path: Path = Field(default=Path("data/urls.csv"))
    model_path: Path = Field(default=Path("models/phishing_model.joblib"))
    metrics_path: Path = Field(default=Path("data/metrics.json"))
    mlflow_tracking_uri: str = Field(default="file:./mlruns")
    mlflow_experiment_name: str = Field(default="phishing-url-classifier")

    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_reload: bool = Field(default=False)

    # Model
    phishing_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    random_seed: int = Field(default=42)

    # Monitoring seam (no-op by default; wire to Prometheus/Datadog when enabled)
    monitoring_enabled: bool = Field(default=False)

    # Logging
    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=False)

    def resolve_path(self, path: Path) -> Path:
        """Resolve a path relative to project root if not absolute."""
        if path.is_absolute():
            return path
        return self.project_root / path


settings = Settings()
