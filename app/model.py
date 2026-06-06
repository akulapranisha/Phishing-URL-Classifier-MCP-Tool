"""Model artifact loading, saving, and prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from app.features import FEATURE_NAMES, extract_features, features_to_vector


class FeatureTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer wrapping shared extract_features()."""

    def fit(self, X: Any, y: Any = None) -> FeatureTransformer:
        """No-op fit — feature schema is fixed."""
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Transform URLs or feature dicts into a numeric matrix."""
        rows: list[list[float]] = []
        for item in X:
            if isinstance(item, str):
                features = extract_features(item)
            elif isinstance(item, dict):
                features = item
            else:
                raise TypeError(f"Unsupported input type: {type(item)!r}")
            rows.append(features_to_vector(features))
        return np.asarray(rows, dtype=np.float64)


@dataclass(frozen=True)
class PredictionResult:
    """Structured prediction output."""

    label: Literal["phishing", "benign"]
    probability: float
    features_used: int


@dataclass
class PhishingModelArtifact:
    """In-memory representation of the saved joblib artifact."""

    model: Any
    transformer: FeatureTransformer
    feature_names: list[str]
    threshold: float
    version: str = "1.0.0"

    def predict(self, url: str) -> PredictionResult:
        """Score a single URL."""
        features = extract_features(url)
        vector = np.asarray([features_to_vector(features)], dtype=np.float64)
        proba = float(self.model.predict_proba(vector)[0][1])
        label: Literal["phishing", "benign"] = (
            "phishing" if proba >= self.threshold else "benign"
        )
        return PredictionResult(
            label=label,
            probability=round(proba, 6),
            features_used=len(self.feature_names),
        )


def save_artifact(
    path: Path,
    model: Any,
    transformer: FeatureTransformer,
    threshold: float = 0.5,
    version: str = "1.0.0",
) -> None:
    """Persist model and feature transformer as a single joblib artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "transformer": transformer,
        "feature_names": list(FEATURE_NAMES),
        "threshold": threshold,
        "version": version,
    }
    joblib.dump(artifact, path)


def load_artifact(path: Path) -> PhishingModelArtifact:
    """Load the joblib artifact from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")

    raw = joblib.load(path)
    return PhishingModelArtifact(
        model=raw["model"],
        transformer=raw["transformer"],
        feature_names=list(raw.get("feature_names", FEATURE_NAMES)),
        threshold=float(raw.get("threshold", 0.5)),
        version=str(raw.get("version", "1.0.0")),
    )
