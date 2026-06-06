"""FastAPI REST API for phishing URL classification."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

import structlog
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.logging_config import configure_logging, get_logger
from app.model import PhishingModelArtifact, load_artifact
from app.monitoring import monitoring

configure_logging()
logger = get_logger(__name__)

_model: PhishingModelArtifact | None = None


class PredictRequest(BaseModel):
    """Request body for /predict."""

    url: str = Field(..., min_length=1, description="URL to classify")


class PredictResponse(BaseModel):
    """Response body for /predict."""

    label: Literal["phishing", "benign"]
    probability: float = Field(..., ge=0.0, le=1.0)
    features_used: int


class HealthResponse(BaseModel):
    """Response body for /health."""

    status: str
    model_loaded: bool
    model_version: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifact once at startup."""
    global _model
    model_path = settings.resolve_path(settings.model_path)
    logger.info("loading_model", path=str(model_path))
    try:
        _model = load_artifact(model_path)
        logger.info("model_loaded", version=_model.version, features=len(_model.feature_names))
    except FileNotFoundError as exc:
        logger.error("model_not_found", error=str(exc))
        _model = None
    yield
    _model = None


app = FastAPI(
    title="Phishing URL Classifier",
    description="Gradient-boosted lexical/host feature classifier for phishing detection.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok" if _model is not None else "degraded",
        model_loaded=_model is not None,
        model_version=_model.version if _model else None,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Classify a URL as phishing or benign."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    structlog.contextvars.bind_contextvars(url=request.url[:120])

    with monitoring.timer("predict_latency_ms"):
        try:
            result = _model.predict(request.url)
        except Exception as exc:
            logger.exception("predict_failed", error=str(exc))
            raise HTTPException(status_code=400, detail=f"Invalid URL or prediction error: {exc}") from exc

    monitoring.increment("predict_total", tags={"label": result.label})
    logger.info(
        "predict_complete",
        label=result.label,
        probability=result.probability,
    )

    return PredictResponse(
        label=result.label,
        probability=result.probability,
        features_used=result.features_used,
    )


def main() -> None:
    """Run uvicorn server."""
    import uvicorn

    uvicorn.run(
        "app.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )


if __name__ == "__main__":
    main()
