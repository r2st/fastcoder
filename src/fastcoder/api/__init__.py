"""FastAPI application factory for the Autonomous Software Development Agent."""

from __future__ import annotations

import asyncio
import hashlib
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Optional

import structlog
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles

from fastcoder.api.routes import create_router
from fastcoder.api.admin_routes import create_admin_router
from fastcoder.api.ops_routes import create_ops_router
from fastcoder.api.rate_limiter import RateLimitConfig, create_rate_limit_middleware
from fastcoder.api.request_context import create_request_context_middleware
from fastcoder.api.metrics import create_metrics_router, MetricsCollector
from fastcoder.auth.auth_routes import create_auth_router, _validate_session, SESSION_COOKIE, ADMIN_SESSION_COOKIE

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Admin server — started alongside the main app
# ---------------------------------------------------------------------------
_admin_app_holder: dict = {}      # populated by create_app or externally
_admin_server_task: Optional[asyncio.Task] = None


# ---------------------------------------------------------------------------
# Security: API key / token authentication
# ---------------------------------------------------------------------------

# The agent generates a random API token at startup if none is set via env var.
# Clients must send it as a Bearer token in the Authorization header.
_API_TOKEN: Optional[str] = None


def _get_api_token() -> str:
    """Get or generate the API token for this session."""
    global _API_TOKEN
    if _API_TOKEN is None:
        _API_TOKEN = os.environ.get("AGENT_API_TOKEN") or secrets.token_urlsafe(32)
        logger.info(
            "api_token_configured",
            source="env" if os.environ.get("AGENT_API_TOKEN") else "auto-generated",
        )
        if not os.environ.get("AGENT_API_TOKEN"):
            # Print to stderr so operator can grab it
            import sys
            print(f"\n  API Token: {_API_TOKEN}\n", file=sys.stderr)
    return _API_TOKEN


async def _verify_token(request: Request) -> None:
    """Dependency that verifies the Bearer token on protected routes.

    Skips auth for health checks, static assets, and the root page.
    """
    path = request.url.path

    # Public endpoints that don't require authentication
    public_paths = {"/health", "/favicon.ico", "/", "/admin", "/workspace", "/login", "/auth/token"}
    if path in public_paths or path.startswith((
        "/admin-static", "/workspace-static", "/login-static",
        "/auth/register", "/auth/login", "/auth/logout",
        "/auth/providers", "/auth/me", "/auth/oauth/",
    )):
        return

    # Check session cookies (for browser-based auth)
    # Try both workspace and admin cookies — either grants access
    for cookie_name in (SESSION_COOKIE, ADMIN_SESSION_COOKIE):
        session_id = request.cookies.get(cookie_name)
        if session_id:
            session = _validate_session(session_id)
            if session:
                return  # Valid session — allow through

    # Check Authorization header (Bearer token for API access)
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header. Use: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = auth_header[7:]  # Strip "Bearer "
    expected = _get_api_token()

    # Constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(token, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def create_app(
    orchestrator,
    story_store: dict,
    config_holder: dict | None = None,
    activity_log: list | None = None,
    approval_manager=None,
    admin_app: FastAPI | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Sets up:
    - CORS middleware (restricted origins)
    - Bearer token authentication on all API routes
    - Request logging middleware
    - Health endpoint
    - API routes from routes.py
    - Admin config routes from admin_routes.py
    - Operational routes from ops_routes.py
    - Static file serving for admin and workspace UIs
    - Safe exception handlers (no internal detail leakage)
    - Lifespan that starts the admin server on its own port

    Args:
        orchestrator: The Orchestrator instance.
        story_store: Dict to store Story objects.
        config_holder: Dict with "config" key holding current AgentConfig.
        activity_log: List to accumulate activity entries.
        approval_manager: ApprovalGateManager instance.
        admin_app: Optional pre-built admin FastAPI app to run on AGENT_ADMIN_PORT.

    Returns:
        Configured FastAPI application.
    """
    # Ensure the API token is initialised early
    _get_api_token()

    # Store admin app for the lifespan to start
    if admin_app is not None:
        _admin_app_holder["app"] = admin_app

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        """Start the admin server as a background task alongside the main app."""
        global _admin_server_task
        a_app = _admin_app_holder.get("app")
        if a_app is not None:
            admin_port = int(os.environ.get("AGENT_ADMIN_PORT", "3001"))
            cfg = uvicorn.Config(
                a_app,
                host="0.0.0.0",
                port=admin_port,
                log_level="warning",   # quieter than main
            )
            server = uvicorn.Server(cfg)
            _admin_server_task = asyncio.create_task(server.serve())
            logger.info(f"Admin server starting on port {admin_port}")
        yield
        # Shutdown admin server when main app stops
        if _admin_server_task and not _admin_server_task.done():
            _admin_server_task.cancel()
            try:
                await _admin_server_task
            except asyncio.CancelledError:
                pass

    app = FastAPI(
        title="Autonomous Software Development Agent",
        description="API for autonomous code generation and development",
        version="3.2.0",
        lifespan=_lifespan,
    )

    # ------------------------------------------------------------------
    # CORS middleware — restricted to localhost by default.
    # Set AGENT_CORS_ORIGINS env var to a comma-separated list of origins
    # to allow cross-origin access from specific domains.
    # ------------------------------------------------------------------
    cors_origins_raw = os.environ.get("AGENT_CORS_ORIGINS", "")
    if cors_origins_raw.strip():
        allowed_origins = [o.strip() for o in cors_origins_raw.split(",") if o.strip()]
    else:
        admin_port = os.environ.get("AGENT_ADMIN_PORT", "3001")
        allowed_origins = [
            "http://localhost:3000",
            "http://localhost:8000",
            "http://localhost:8080",
            f"http://localhost:{admin_port}",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8000",
            "http://127.0.0.1:8080",
            f"http://127.0.0.1:{admin_port}",
        ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,  # Needed for session cookies
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Accept"],
    )

    # ------------------------------------------------------------------
    # Request context middleware — adds request_id and correlation_id
    # to every request and binds them to structlog context.
    # ------------------------------------------------------------------
    create_request_context_middleware(app)

    # ------------------------------------------------------------------
    # Rate limiting middleware — per-IP token bucket
    # ------------------------------------------------------------------
    rate_limit_config = RateLimitConfig(requests_per_minute=60, burst_size=20)
    create_rate_limit_middleware(app, rate_limit_config)

    # ------------------------------------------------------------------
    # Request logging middleware
    # ------------------------------------------------------------------
    @app.middleware("http")
    async def log_requests(request: Request, call_next: Callable) -> object:
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        logger.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_s=round(duration, 3),
        )
        return response

    # ------------------------------------------------------------------
    # Request size limiter — reject bodies > 1 MB
    # ------------------------------------------------------------------
    MAX_BODY_BYTES = 1_048_576  # 1 MB

    @app.middleware("http")
    async def limit_request_body(request: Request, call_next: Callable) -> object:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={"error": "Request body too large", "max_bytes": MAX_BODY_BYTES},
            )
        return await call_next(request)

    # ------------------------------------------------------------------
    # Favicon
    # ------------------------------------------------------------------
    favicon_path = Path(__file__).parent.parent.parent.parent / "favicon.ico"
    if favicon_path.exists():

        @app.get("/favicon.ico", include_in_schema=False)
        async def favicon():
            return FileResponse(
                favicon_path,
                media_type="image/x-icon",
                headers={"Cache-Control": "public, max-age=86400"},
            )

    # ------------------------------------------------------------------
    # Health check — public, no auth required
    # ------------------------------------------------------------------
    @app.get("/health")
    async def health_check() -> dict:
        return {"status": "healthy", "timestamp": time.time()}

    # ------------------------------------------------------------------
    # Prometheus-compatible metrics endpoint (public for scraping)
    # ------------------------------------------------------------------
    metrics_router = create_metrics_router()
    app.include_router(metrics_router)

    # ------------------------------------------------------------------
    # Token endpoint — serves API token to same-origin UI pages only.
    # The built-in workspace and admin UIs fetch this on load so they
    # can attach the Bearer token to subsequent API requests.
    # ------------------------------------------------------------------
    @app.get("/auth/token", include_in_schema=False)
    async def get_auth_token(request: Request) -> dict:
        referer = request.headers.get("referer", "")
        host = request.headers.get("host", "")
        # Only serve the token when the request originates from the
        # same host (i.e. the built-in UI pages).
        if host and referer:
            from urllib.parse import urlparse
            parsed = urlparse(referer)
            referer_host = parsed.netloc
            if referer_host == host:
                return {"token": _get_api_token()}
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Token endpoint is only accessible from the built-in UI",
        )

    # ------------------------------------------------------------------
    # Auth routes (public — handles its own auth checks)
    # ------------------------------------------------------------------
    auth_router = create_auth_router()
    app.include_router(auth_router, dependencies=[Depends(_verify_token)])

    # ------------------------------------------------------------------
    # API routes (protected by auth dependency)
    # ------------------------------------------------------------------
    router = create_router(orchestrator, story_store)
    app.include_router(router, dependencies=[Depends(_verify_token)])

    if config_holder is not None:
        admin_router = create_admin_router(config_holder)
        app.include_router(admin_router, dependencies=[Depends(_verify_token)])

    if activity_log is None:
        activity_log = []
    ops_router = create_ops_router(
        orchestrator, story_store, activity_log, approval_manager
    )
    app.include_router(ops_router, dependencies=[Depends(_verify_token)])

    # ------------------------------------------------------------------
    # Static file serving for UIs
    # ------------------------------------------------------------------
    package_root = Path(__file__).parent.parent.parent.parent

    # Admin panel is served on its own port — see create_admin_app()
    # We keep a redirect so direct hits to /admin go to the admin server.
    @app.get("/admin")
    async def admin_redirect(request: Request):
        """Redirect to admin panel on its dedicated port."""
        host = request.headers.get("host", "localhost:3000")
        hostname = host.split(":")[0]
        admin_port = os.environ.get("AGENT_ADMIN_PORT", "3001")
        from fastapi.responses import RedirectResponse as RR
        return RR(url=f"http://{hostname}:{admin_port}/")

    workspace_dir = package_root / "workspace-ui"
    if workspace_dir.exists():

        @app.get("/")
        async def workspace_home():
            return FileResponse(workspace_dir / "index.html")

        @app.get("/workspace")
        async def workspace_alias():
            return FileResponse(workspace_dir / "index.html")

        app.mount(
            "/workspace-static",
            StaticFiles(directory=str(workspace_dir)),
            name="workspace-static",
        )

    # ------------------------------------------------------------------
    # Login page
    # ------------------------------------------------------------------
    login_ui_dir = package_root / "login-ui"
    if login_ui_dir.exists():

        @app.get("/login")
        async def login_page():
            return FileResponse(login_ui_dir / "index.html")

        app.mount(
            "/login-static",
            StaticFiles(directory=str(login_ui_dir)),
            name="login-static",
        )

    # ------------------------------------------------------------------
    # Exception handlers — safe: no internal details leak to client
    # ------------------------------------------------------------------
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        # Log full detail server-side, return generic message to client
        logger.warning("value_error", path=request.url.path, detail=str(exc))
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid request", "request_id": str(uuid.uuid4())},
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        # Log full traceback server-side for debugging
        request_id = str(uuid.uuid4())
        logger.error(
            f"Unhandled exception [{request_id}]: {exc}",
            exc_info=True,
        )
        # Never expose internal details to the client
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "request_id": request_id,
            },
        )

    return app


# ======================================================================
# Admin-only application — runs on a separate port
# ======================================================================

def create_admin_app(
    config_holder: dict | None = None,
    main_port: int = 3000,
) -> FastAPI:
    """Create a lightweight FastAPI app that serves *only* the admin panel.

    It runs on its own port (default 3001) so admin and workspace sessions
    are fully isolated.  It mounts:
    - Admin panel UI (HTML)
    - Login page (for admin login)
    - Auth routes (login / logout / me / providers / admin/* endpoints)
    - Admin config routes (project, LLM, safety, etc.)
    - Health check

    API calls from the admin UI that need the main orchestrator
    (e.g. dashboard metrics, stories) are proxied by the admin UI
    to the main port via ``API_BASE`` — but most admin-panel
    sections only need the config endpoints which live here.
    """
    _get_api_token()

    admin_app = FastAPI(
        title="Auto-Dev Agent – Admin",
        version="3.2.0",
    )

    # CORS — allow the main server's origin so cross-port fetch works
    admin_port_str = os.environ.get("AGENT_ADMIN_PORT", "3001")
    admin_origins = [
        f"http://localhost:{main_port}",
        f"http://127.0.0.1:{main_port}",
        f"http://localhost:{admin_port_str}",
        f"http://127.0.0.1:{admin_port_str}",
    ]
    admin_app.add_middleware(
        CORSMiddleware,
        allow_origins=admin_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Accept"],
    )

    # Middleware — request logging
    @admin_app.middleware("http")
    async def admin_log_requests(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        logger.info(
            f"[admin] {request.method} {request.url.path} "
            f"status={response.status_code} duration={time.time()-start:.3f}s"
        )
        return response

    # Health
    @admin_app.get("/health")
    async def admin_health():
        return {"status": "healthy", "service": "admin", "timestamp": time.time()}

    # Token endpoint (same-origin only)
    @admin_app.get("/auth/token", include_in_schema=False)
    async def admin_get_auth_token(request: Request):
        referer = request.headers.get("referer", "")
        host = request.headers.get("host", "")
        if host and referer:
            from urllib.parse import urlparse
            if urlparse(referer).netloc == host:
                return {"token": _get_api_token()}
        raise HTTPException(403, "Token endpoint is only accessible from the built-in UI")

    # Auth routes — shared in-memory state (users, sessions, oauth)
    auth_router = create_auth_router()
    admin_app.include_router(auth_router, dependencies=[Depends(_verify_token)])

    # Admin config routes (project settings, LLM, safety, quality, etc.)
    if config_holder is not None:
        admin_router_cfg = create_admin_router(config_holder)
        admin_app.include_router(admin_router_cfg, dependencies=[Depends(_verify_token)])

    # Proxy endpoint — forward dashboard / stories / ops requests to main server
    @admin_app.api_route(
        "/api/v1/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE"],
        include_in_schema=False,
    )
    async def proxy_to_main(path: str, request: Request):
        """Proxy API calls to the main server for routes not on the admin app."""
        import httpx
        target = f"http://127.0.0.1:{main_port}/api/v1/{path}"
        headers = dict(request.headers)
        headers.pop("host", None)
        body = await request.body()

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.request(
                method=request.method,
                url=target,
                headers=headers,
                content=body,
                cookies=request.cookies,
            )

        from fastapi.responses import Response as RawResponse
        return RawResponse(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers),
        )

    # Favicon
    package_root = Path(__file__).parent.parent.parent.parent
    favicon_path = package_root / "favicon.ico"
    if favicon_path.exists():
        @admin_app.get("/favicon.ico", include_in_schema=False)
        async def admin_favicon():
            return FileResponse(
                favicon_path,
                media_type="image/x-icon",
                headers={"Cache-Control": "public, max-age=86400"},
            )

    # Admin panel UI — served at root since the admin app runs on its own port
    admin_panel_dir = package_root / "admin-panel"
    if admin_panel_dir.exists():
        @admin_app.get("/")
        async def admin_root():
            return FileResponse(admin_panel_dir / "index.html")

        @admin_app.get("/admin")
        async def admin_legacy_alias():
            """Backward-compat alias — redirect to root."""
            from fastapi.responses import RedirectResponse as RR
            return RR(url="/")

        admin_app.mount(
            "/admin-static",
            StaticFiles(directory=str(admin_panel_dir)),
            name="admin-static",
        )

    # Login page
    login_ui_dir = package_root / "login-ui"
    if login_ui_dir.exists():
        @admin_app.get("/login")
        async def admin_login_page():
            return FileResponse(login_ui_dir / "index.html")

        admin_app.mount(
            "/login-static",
            StaticFiles(directory=str(login_ui_dir)),
            name="login-static",
        )

    # Exception handlers
    @admin_app.exception_handler(ValueError)
    async def admin_value_error(request: Request, exc: ValueError):
        logger.warning("admin_value_error", path=request.url.path, detail=str(exc))
        return JSONResponse(status_code=400, content={"error": "Invalid request"})

    @admin_app.exception_handler(Exception)
    async def admin_general_error(request: Request, exc: Exception):
        rid = str(uuid.uuid4())
        logger.error(f"[admin] Unhandled [{rid}]: {exc}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Internal server error", "request_id": rid})

    return admin_app
