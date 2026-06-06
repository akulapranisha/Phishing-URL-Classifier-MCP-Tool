"""Training CLI for the phishing URL classifier."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

from app.config import settings
from app.features import FEATURE_NAMES, extract_features
from app.logging_config import configure_logging, get_logger
from app.model import FeatureTransformer, save_artifact

logger = get_logger(__name__)

LABEL_MAP = {"benign": 0, "phishing": 1, "legitimate": 0, "safe": 0, "bad": 1, "malicious": 1}


def load_dataset(path: Path) -> pd.DataFrame:
    """Load labeled URL dataset from CSV."""
    df = pd.read_csv(path)
    required = {"url", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")

    df = df.dropna(subset=["url", "label"]).copy()
    df["url"] = df["url"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    def _map_label(raw: str) -> int:
        if raw in LABEL_MAP:
            return LABEL_MAP[raw]
        if raw in {"0", "1"}:
            return int(raw)
        raise ValueError(f"Unknown label: {raw!r}")

    df["target"] = df["label"].map(_map_label)
    if df["target"].isna().any():
        unknown = df.loc[df["target"].isna(), "label"].unique().tolist()
        raise ValueError(f"Unmapped labels: {unknown}")

    return df


def build_feature_matrix(urls: list[str]) -> np.ndarray:
    """Extract features for a list of URLs using the shared function."""
    rows = [extract_features(url) for url in urls]
    return np.asarray(
        [[row[name] for name in FEATURE_NAMES] for row in rows],
        dtype=np.float64,
    )


def evaluate_model(
    model: XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Compute classification metrics on held-out data."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }


def save_confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    """Save confusion matrix PNG."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = ["benign", "phishing"]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    thresh = cm.max() / 2.0 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train(dataset_path: Path | None = None) -> dict[str, float]:
    """Run the full training pipeline with MLflow tracking."""
    configure_logging()
    data_path = settings.resolve_path(dataset_path or settings.data_path)
    model_path = settings.resolve_path(settings.model_path)
    metrics_path = settings.resolve_path(settings.metrics_path)

    logger.info("training_start", dataset=str(data_path))
    df = load_dataset(data_path)

    X = build_feature_matrix(df["url"].tolist())
    y = df["target"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=settings.random_seed,
        stratify=y,
    )

    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=settings.random_seed,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=settings.random_seed)
    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [100, 200],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    best_model: XGBClassifier = search.best_estimator_

    cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring="roc_auc")
    test_metrics = evaluate_model(best_model, X_test, y_test)
    y_pred = best_model.predict(X_test)

    transformer = FeatureTransformer()
    save_artifact(
        model_path,
        best_model,
        transformer,
        threshold=settings.phishing_threshold,
    )

    cm_path = metrics_path.parent / "confusion_matrix.png"
    save_confusion_matrix_plot(y_test, y_pred, cm_path)

    metrics_payload = {
        **test_metrics,
        "cv_roc_auc_mean": float(cv_scores.mean()),
        "cv_roc_auc_std": float(cv_scores.std()),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "best_params": search.best_params_,
        "feature_count": len(FEATURE_NAMES),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name="phishing-xgb-train"):
        mlflow.log_params(search.best_params_)
        mlflow.log_param("random_seed", settings.random_seed)
        mlflow.log_param("feature_count", len(FEATURE_NAMES))
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        for name, value in test_metrics.items():
            mlflow.log_metric(name, value)
        mlflow.log_metric("cv_roc_auc_mean", float(cv_scores.mean()))
        mlflow.log_metric("cv_roc_auc_std", float(cv_scores.std()))
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
        mlflow.log_artifact(str(cm_path), artifact_path="plots")
        mlflow.sklearn.log_model(best_model, artifact_path="xgboost_estimator")

    logger.info("training_complete", metrics=test_metrics, model_path=str(model_path))
    return test_metrics


def main() -> None:
    """CLI entrypoint."""
    import argparse

    parser = argparse.ArgumentParser(description="Train phishing URL classifier")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to CSV dataset (default: settings.data_path)",
    )
    args = parser.parse_args()

    try:
        metrics = train(args.data)
        print(json.dumps(metrics, indent=2))
    except Exception as exc:
        logger.exception("training_failed", error=str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
