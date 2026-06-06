"""Monitoring seam — no-op by default, extensible when MONITORING_ENABLED=true."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


class MonitoringClient:
    """Thin abstraction for metrics emission behind a feature flag."""

    def __init__(self, enabled: bool | None = None) -> None:
        self.enabled = settings.monitoring_enabled if enabled is None else enabled

    def increment(self, metric: str, tags: dict[str, str] | None = None) -> None:
        """Increment a counter metric."""
        if not self.enabled:
            return
        logger.info("metric_increment", metric=metric, tags=tags or {})

    def histogram(self, metric: str, value: float, tags: dict[str, str] | None = None) -> None:
        """Record a histogram/timing metric."""
        if not self.enabled:
            return
        logger.info("metric_histogram", metric=metric, value=value, tags=tags or {})

    @contextmanager
    def timer(self, metric: str, tags: dict[str, str] | None = None) -> Generator[None, None, None]:
        """Time an operation and emit a histogram when monitoring is enabled."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.histogram(metric, elapsed_ms, tags)


monitoring = MonitoringClient()
