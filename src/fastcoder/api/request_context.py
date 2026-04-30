"""Request context management using contextvars for request ID and correlation ID.

This module provides:
- ContextVar for request_id (str)
- ContextVar for correlation_id (str)
- FastAPI middleware to manage these values
- Helper functions to retrieve context values
- Automatic binding to structlog context
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar
from typing import Callable, Optional

import structlog
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = structlog.get_logger(__name__)

# Context variables
_request_id_var: ContextVar[str] = ContextVar("request_id", default="")
_correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")


def get_request_id() -> str:
    """Retrieve the current request ID from context.

    Returns:
        The current request ID string, or empty string if not set.
    """
    return _request_id_var.get()


def get_correlation_id() -> str:
    """Retrieve the current correlation ID from context.

    Returns:
        The current correlation ID string, or empty string if not set.
    """
    return _correlation_id_var.get()


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware to manage request context (request_id and correlation_id).

    Sets up context variables for every request and binds them to structlog.
    Adds X-Request-ID and X-Correlation-ID to response headers.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and manage context.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware/route handler.

        Returns:
            HTTP response with request/correlation ID headers.
        """
        # Generate a new request ID
        request_id = str(uuid.uuid4())

        # Extract correlation ID from header or generate new one
        correlation_id = request.headers.get(
            "X-Correlation-ID", str(uuid.uuid4())
        )

        # Set context variables
        _request_id_var.set(request_id)
        _correlation_id_var.set(correlation_id)

        # Bind to structlog context so all logs include these IDs
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            correlation_id=correlation_id,
        )

        logger.debug(
            "request_context_initialized",
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if request.client else "unknown",
        )

        # Call next handler
        response = await call_next(request)

        # Add IDs to response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Correlation-ID"] = correlation_id

        logger.debug(
            "request_context_response_sent",
            status_code=response.status_code,
        )

        return response


def create_request_context_middleware(app: FastAPI) -> None:
    """Create and attach request context middleware to FastAPI app.

    This middleware:
    - Generates a UUID4 request_id for every request
    - Extracts X-Correlation-ID header or generates one
    - Sets both in context variables (accessible via get_request_id/get_correlation_id)
    - Binds both to structlog context (all logs will include them)
    - Adds X-Request-ID and X-Correlation-ID to response headers

    Args:
        app: FastAPI application instance.

    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> create_request_context_middleware(app)
        >>> # Now all requests have request/correlation IDs in logs and response headers
    """
    app.add_middleware(RequestContextMiddleware)
    logger.info("request_context_middleware_added")
