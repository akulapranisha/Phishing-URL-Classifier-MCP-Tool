"""Tests for FastAPI /predict endpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from xgboost import XGBClassifier

from app import api
from app.config import settings
from app.features import FEATURE_NAMES, extract_features
from app.model import FeatureTransformer, save_artifact


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Provide TestClient with a loaded in-memory model."""
    urls = [
        "https://www.google.com",
        "https://www.python.org",
        "http://paypa1-login.com/verify",
        "http://192.168.0.1/bank-login",
    ]
    labels = np.array([0, 0, 1, 1])
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

    model_path = tmp_path / "phishing_model.joblib"
    save_artifact(model_path, model, FeatureTransformer())

    monkeypatch.setattr(settings, "model_path", model_path)
    monkeypatch.setattr(settings, "project_root", tmp_path)

    from app.model import load_artifact

    api._model = load_artifact(model_path)
    return TestClient(api.app)


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={"url": "https://www.google.com/search?q=test"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in {"phishing", "benign"}
    assert 0.0 <= body["probability"] <= 1.0
    assert body["features_used"] == 23


def test_predict_phishing_url(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={"url": "http://paypa1-secure-login.com/signin"},
    )
    assert response.status_code == 200
    assert "probability" in response.json()
