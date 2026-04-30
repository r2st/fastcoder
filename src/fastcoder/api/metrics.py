"""Lightweight metrics collection system for Prometheus-compatible export."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger(__name__)


@dataclass
class HistogramBucket:
    """Single histogram bucket with count."""

    le: float  # Less than or equal to
    count: int = 0


class MetricsCollector:
    """Singleton metrics collector with Prometheus-compatible export."""

    _instance: Optional[MetricsCollector] = None

    def __new__(cls) -> MetricsCollector:
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize metrics collector."""
        if self._initialized:
            return

        self._initialized = True

        # Counters: name -> {labels_key -> value}
        self._counters: dict[str, dict[str, int]] = {
            "stories_submitted": {},
            "stories_completed": {},
            "stories_failed": {},
            "llm_calls_total": {},
            "llm_errors_total": {},
            "http_requests_total": {},
        }

        # Histograms: name -> {labels_key -> [buckets]}
        # Bucket boundaries for common metrics
        self._llm_latency_buckets = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        self._http_duration_buckets = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        self._story_processing_buckets = [1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0]

        self._histograms: dict[str, dict[str, dict[float, int]]] = {
            "llm_latency_seconds": {},
            "http_request_duration_seconds": {},
            "story_processing_duration_seconds": {},
        }

        # Gauges: name -> {labels_key -> value}
        self._gauges: dict[str, dict[str, float]] = {
            "active_stories": {},
            "approval_gates_pending": {},
        }

        logger.info("metrics_collector_initialized")

    def increment(self, name: str, labels: Optional[dict[str, str]] = None, value: int = 1) -> None:
        """Increment a counter metric.

        Args:
            name: Metric name (e.g., 'stories_submitted').
            labels: Optional dict of label key-values (e.g., {'provider': 'anthropic'}).
            value: Amount to increment by (default 1).
        """
        if name not in self._counters:
            logger.warning("unknown_metric", metric_name=name, metric_type="counter")
            return

        labels_key = self._make_labels_key(labels)
        if labels_key not in self._counters[name]:
            self._counters[name][labels_key] = 0

        self._counters[name][labels_key] += value

    def observe(self, name: str, value: float, labels: Optional[dict[str, str]] = None) -> None:
        """Observe a histogram value.

        Args:
            name: Histogram metric name (e.g., 'llm_latency_seconds').
            value: Value to observe.
            labels: Optional dict of label key-values.
        """
        if name not in self._histograms:
            logger.warning("unknown_metric", metric_name=name, metric_type="histogram")
            return

        labels_key = self._make_labels_key(labels)

        # Initialize histogram for this label combination if needed
        if labels_key not in self._histograms[name]:
            self._histograms[name][labels_key] = {}

        # Determine which buckets to use
        if name == "llm_latency_seconds":
            buckets = self._llm_latency_buckets
        elif name == "http_request_duration_seconds":
            buckets = self._http_duration_buckets
        elif name == "story_processing_duration_seconds":
            buckets = self._story_processing_buckets
        else:
            buckets = self._llm_latency_buckets  # Default

        # Initialize bucket counts if needed
        for bucket_le in buckets:
            if bucket_le not in self._histograms[name][labels_key]:
                self._histograms[name][labels_key][bucket_le] = 0

        # Increment all buckets where value <= bucket_le
        for bucket_le in buckets:
            if value <= bucket_le:
                self._histograms[name][labels_key][bucket_le] += 1

    def set_gauge(self, name: str, value: float, labels: Optional[dict[str, str]] = None) -> None:
        """Set a gauge metric to a specific value.

        Args:
            name: Gauge metric name (e.g., 'active_stories').
            value: Value to set.
            labels: Optional dict of label key-values.
        """
        if name not in self._gauges:
            logger.warning("unknown_metric", metric_name=name, metric_type="gauge")
            return

        labels_key = self._make_labels_key(labels)
        self._gauges[name][labels_key] = value

    def export_prometheus(self) -> str:
        """Export all metrics in Prometheus text exposition format.

        Returns:
            Prometheus text format metrics string.
        """
        lines = []

        # Export counters
        for name, label_values in self._counters.items():
            lines.append(f"# HELP {name} Counter metric")
            lines.append(f"# TYPE {name} counter")
            for labels_key, value in label_values.items():
                if labels_key:
                    lines.append(f'{name}{{{labels_key}}} {value}')
                else:
                    lines.append(f"{name} {value}")

        # Export histograms
        for name, label_values in self._histograms.items():
            lines.append(f"# HELP {name} Histogram metric")
            lines.append(f"# TYPE {name} histogram")

            for labels_key, buckets in label_values.items():
                # Sort bucket boundaries
                sorted_buckets = sorted(buckets.items())

                for bucket_le, count in sorted_buckets:
                    if labels_key:
                        lines.append(f'{name}_bucket{{le="{bucket_le}",{labels_key}}} {count}')
                    else:
                        lines.append(f'{name}_bucket{{le="{bucket_le}"}} {count}')

                # Sum of all observations
                if sorted_buckets:
                    total_count = sorted_buckets[-1][1]
                    if labels_key:
                        lines.append(f'{name}_sum{{{labels_key}}} 0')
                        lines.append(f'{name}_count{{{labels_key}}} {total_count}')
                    else:
                        lines.append(f'{name}_sum 0')
                        lines.append(f'{name}_count {total_count}')

        # Export gauges
        for name, label_values in self._gauges.items():
            lines.append(f"# HELP {name} Gauge metric")
            lines.append(f"# TYPE {name} gauge")
            for labels_key, value in label_values.items():
                if labels_key:
                    lines.append(f'{name}{{{labels_key}}} {value}')
                else:
                    lines.append(f"{name} {value}")

        # Add final blank line
        lines.append("")

        return "\n".join(lines)

    def export_json(self) -> dict[str, Any]:
        """Export all metrics as JSON.

        Returns:
            Dictionary with 'counters', 'histograms', and 'gauges' keys.
        """
        # Convert label keys back to dicts for JSON output
        def _parse_labels_key(key: str) -> dict[str, str]:
            """Parse labels_key string back to dict."""
            if not key:
                return {}
            labels = {}
            for pair in key.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    labels[k] = v
            return labels

        counters_json = {}
        for name, label_values in self._counters.items():
            counters_json[name] = {}
            for labels_key, value in label_values.items():
                labels = _parse_labels_key(labels_key)
                counters_json[name][labels_key or "no_labels"] = {"value": value, "labels": labels}

        histograms_json = {}
        for name, label_values in self._histograms.items():
            histograms_json[name] = {}
            for labels_key, buckets in label_values.items():
                labels = _parse_labels_key(labels_key)
                histograms_json[name][labels_key or "no_labels"] = {
                    "buckets": buckets,
                    "labels": labels,
                }

        gauges_json = {}
        for name, label_values in self._gauges.items():
            gauges_json[name] = {}
            for labels_key, value in label_values.items():
                labels = _parse_labels_key(labels_key)
                gauges_json[name][labels_key or "no_labels"] = {"value": value, "labels": labels}

        return {
            "counters": counters_json,
            "histograms": histograms_json,
            "gauges": gauges_json,
        }

    @staticmethod
    def _make_labels_key(labels: Optional[dict[str, str]] = None) -> str:
        """Convert labels dict to Prometheus-compatible key string.

        Args:
            labels: Dictionary of label key-values.

        Returns:
            Labels as a string like 'provider="anthropic",method="POST"' or empty string.
        """
        if not labels:
            return ""

        pairs = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return ",".join(pairs)


class MetricsMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for automatic HTTP metrics collection."""

    def __init__(self, app, metrics: MetricsCollector | None = None):
        """Initialize metrics middleware.

        Args:
            app: FastAPI application instance.
            metrics: MetricsCollector instance (uses singleton if None).
        """
        super().__init__(app)
        self.metrics = metrics or MetricsCollector()

    async def dispatch(self, request: Request, call_next) -> Any:
        """Intercept request/response to collect metrics.

        Args:
            request: The incoming request.
            call_next: Callable to process the request.

        Returns:
            Response from the application.
        """
        start_time = time.time()

        # Record start of request
        method = request.method
        path = request.url.path

        try:
            response = await call_next(request)
        except Exception as e:
            # Record error and re-raise
            self.metrics.increment(
                "http_requests_total",
                labels={"method": method, "path": path, "status": "error"},
            )
            raise

        # Record metrics after response
        duration_seconds = time.time() - start_time
        status_code = response.status_code

        self.metrics.increment(
            "http_requests_total",
            labels={"method": method, "path": path, "status": str(status_code)},
        )

        self.metrics.observe(
            "http_request_duration_seconds",
            duration_seconds,
            labels={"method": method, "path": path},
        )

        return response


def create_metrics_router(metrics: MetricsCollector | None = None) -> APIRouter:
    """Create FastAPI router with metrics endpoints.

    Args:
        metrics: MetricsCollector instance (uses singleton if None).

    Returns:
        Configured APIRouter for metrics endpoints.
    """
    router = APIRouter(tags=["metrics"])
    metrics_instance = metrics or MetricsCollector()

    @router.get("/metrics", response_class=PlainTextResponse)
    async def get_metrics_prometheus() -> str:
        """Export metrics in Prometheus text exposition format.

        Returns:
            Prometheus-format metrics string.
        """
        return metrics_instance.export_prometheus()

    @router.get("/metrics/json")
    async def get_metrics_json() -> dict[str, Any]:
        """Export metrics as JSON.

        Returns:
            JSON representation of all metrics.
        """
        return metrics_instance.export_json()

    logger.info("metrics_router_created")

    return router
