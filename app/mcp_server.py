"""MCP server exposing score_url for external LLM agents."""

from __future__ import annotations

import json
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from app.config import settings
from app.logging_config import configure_logging, get_logger
from app.model import PhishingModelArtifact, load_artifact
from app.monitoring import monitoring

configure_logging()
logger = get_logger(__name__)

mcp = FastMCP(
    name="phishing-url-classifier",
    instructions=(
        "Classifies URLs as phishing or benign using a gradient-boosted model "
        "trained on lexical and host features. Call score_url with a full or partial URL."
    ),
)

_model: PhishingModelArtifact | None = None


def _get_model() -> PhishingModelArtifact:
    """Lazy-load the shared joblib artifact."""
    global _model
    if _model is None:
        model_path = settings.resolve_path(settings.model_path)
        logger.info("mcp_loading_model", path=str(model_path))
        _model = load_artifact(model_path)
        logger.info("mcp_model_loaded", version=_model.version)
    return _model


@mcp.tool(
    name="score_url",
    description=(
        "Score a URL for phishing risk. Returns label ('phishing' or 'benign') "
        "and probability (0-1, higher means more likely phishing)."
    ),
)
def score_url(url: str) -> dict[str, Any]:
    """Classify a URL using the same artifact and features as the REST API.

    Args:
        url: The URL string to evaluate (e.g. 'https://example.com/login').

    Returns:
        JSON object with keys: label, probability.
    """
    if not url or not url.strip():
        raise ValueError("url must be a non-empty string")

    with monitoring.timer("mcp_score_url_latency_ms"):
        model = _get_model()
        result = model.predict(url.strip())

    monitoring.increment("mcp_score_url_total", tags={"label": result.label})
    logger.info("mcp_score_url", label=result.label, probability=result.probability)

    return {
        "label": result.label,
        "probability": result.probability,
    }


def main() -> None:
    """Run MCP server over stdio transport."""
    try:
        mcp.run(transport="stdio")
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
