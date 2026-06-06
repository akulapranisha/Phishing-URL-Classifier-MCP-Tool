"""Tests for model artifact load and predict."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from xgboost import XGBClassifier

from app.features import FEATURE_NAMES, extract_features
from app.model import FeatureTransformer, load_artifact, save_artifact


@pytest.fixture
def artifact_path(tmp_path: Path) -> Path:
    """Train a tiny model and save artifact for tests."""
    urls = [
        "https://www.google.com",
        "https://github.com/python",
        "https://www.python.org",
        "http://paypa1-login.com/verify",
        "http://192.168.0.1/bank-login",
        "https://secure-banking-update.xyz/verify",
    ]
    labels = np.array([0, 0, 0, 1, 1, 1])
    X = np.asarray(
        [[extract_features(u)[name] for name in FEATURE_NAMES] for u in urls],
        dtype=np.float64,
    )
    model = XGBClassifier(
        n_estimators=10,
        max_depth=3,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X, labels)

    path = tmp_path / "phishing_model.joblib"
    save_artifact(path, model, FeatureTransformer(), threshold=0.5)
    return path


def test_load_artifact(artifact_path: Path) -> None:
    artifact = load_artifact(artifact_path)
    assert artifact.model is not None
    assert len(artifact.feature_names) == 23


def test_predict_benign(artifact_path: Path) -> None:
    artifact = load_artifact(artifact_path)
    result = artifact.predict("https://www.google.com/search")
    assert result.label in {"phishing", "benign"}
    assert 0.0 <= result.probability <= 1.0
    assert result.features_used == 23


def test_predict_phishing_style(artifact_path: Path) -> None:
    artifact = load_artifact(artifact_path)
    result = artifact.predict("http://paypa1-secure-login.com/signin?account=suspended")
    assert result.label in {"phishing", "benign"}
    assert result.probability >= 0.0


def test_feature_transformer_from_urls() -> None:
    transformer = FeatureTransformer()
    matrix = transformer.transform(["https://example.com", "http://192.168.0.1"])
    assert matrix.shape == (2, 23)
