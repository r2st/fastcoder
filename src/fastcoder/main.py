"""Main entry point for the Autonomous Software Development Agent."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Optional

import structlog
import uvicorn
from structlog.processors import (
    JSONRenderer,
    KeyValueRenderer,
    TimeStamper,
)
from structlog.stdlib import ProcessorFormatter

from fastcoder.api import create_app, create_admin_app
from fastcoder.analyzer import StoryAnalyzer
from fastcoder.codebase import CodebaseIntelligence
from fastcoder.config import load_config, validate_config
from fastcoder.context import ContextManager
from fastcoder.errors.classifier import ErrorClassifier
from fastcoder.errors.recovery import RecoveryManager
from fastcoder.generator import CodeGenerator
from fastcoder.llm.router import ModelRouter
from fastcoder.memory import MemoryStore
from fastcoder.orchestrator import Orchestrator
from fastcoder.planner import Planner
from fastcoder.reviewer import CodeReviewer
from fastcoder.tester import TestGenerator
from fastcoder.tools import ToolLayer
from fastcoder.deployer import Deployer
from fastcoder.verifier import Verifier
from fastcoder.security import SecurityScanner
from fastcoder.quality import QualityGateEngine
from fastcoder.evidence import EvidenceBundleGenerator
from fastcoder.learning import PostMortemEngine
from fastcoder.codebase.ownership_map import OwnershipMap
from fastcoder.codebase.cross_repo_index import CrossRepoIndex
from fastcoder.auth import SSOProvider, SCIMProvider, AuthMiddleware
from fastcoder.auth.types import SAMLConfig, OIDCConfig
from fastcoder.auth.scim_routes import create_scim_router, set_scim_bearer_token
from fastcoder.tools.resource_limiter import ResourceLimiter, ResourceLimits
from fastcoder.types.config import AgentConfig
from fastcoder.orchestrator.adapters import wrap_components
from fastcoder.types.story import Story

logger = structlog.get_logger(__name__)


def _initialize_components(
    config: AgentConfig, memory_store: MemoryStore
) -> dict:
    """Initialize all agent components with error handling.

    Builds components in order with try/except for graceful degradation.
    Returns dict with keys: llm_router, tool_layer, codebase_intelligence,
    context_manager, analyzer, planner, generator, test_generator, reviewer,
    error_classifier, recovery_manager.

    Args:
        config: Loaded and validated agent configuration
        memory_store: Initialized memory store

    Returns:
        Dictionary of initialized components (None values indicate failures)
    """
    components = {}

    # 1. Initialize LLM Providers and Router
    try:
        logger.info("initializing_llm_router")
        llm_router = ModelRouter(
            provider_configs=config.llm.providers,
            routing_config=config.llm.routing,
        )
        components["llm_router"] = llm_router
        logger.info("llm_router_initialized")
    except Exception as e:
        logger.warning("llm_router_initialization_failed", error=str(e))
        components["llm_router"] = None

    # 2. Initialize Tool Layer
    try:
        logger.info("initializing_tool_layer")
        tool_layer = ToolLayer(
            project_dir=config.project.project_dir,
            policies=config.tools.policies if hasattr(config.tools, "policies") else None,
            command_allowlist=config.tools.command_allowlist
            if hasattr(config.tools, "command_allowlist")
            else None,
        )
        components["tool_layer"] = tool_layer
        logger.info("tool_layer_initialized")
    except Exception as e:
        logger.warning("tool_layer_initialization_failed", error=str(e))
        components["tool_layer"] = None

    # 3. Initialize Codebase Intelligence
    try:
        logger.info("initializing_codebase_intelligence")
        codebase_intelligence = CodebaseIntelligence()
        components["codebase_intelligence"] = codebase_intelligence
        logger.info("codebase_intelligence_initialized")
    except Exception as e:
        logger.warning("codebase_intelligence_initialization_failed", error=str(e))
        components["codebase_intelligence"] = None

    # 4. Initialize Context Manager
    try:
        logger.info("initializing_context_manager")
        context_manager = ContextManager(
            graphify_config=config.graphify,
            project_dir=config.project.project_dir,
        )
        components["context_manager"] = context_manager
        logger.info(
            "context_manager_initialized",
            graphify_enabled=config.graphify.enabled,
        )
    except Exception as e:
        logger.warning("context_manager_initialization_failed", error=str(e))
        components["context_manager"] = None

    # 5. Initialize Analysis Components (Analyzer, Planner, Generator, etc)
    # These all need an llm_complete callable from the router
    llm_router = components["llm_router"]
    codebase_intelligence = components["codebase_intelligence"]

    # 5a. StoryAnalyzer
    try:
        logger.info("initializing_story_analyzer")
        analyzer = StoryAnalyzer(llm_complete=llm_router.complete)
        components["analyzer"] = analyzer
        logger.info("story_analyzer_initialized")
    except Exception as e:
        logger.warning("story_analyzer_initialization_failed", error=str(e))
        components["analyzer"] = None

    # 5b. Planner
    try:
        logger.info("initializing_planner")
        codebase_query = (
            codebase_intelligence.search if codebase_intelligence else None
        )
        planner = Planner(
            llm_complete=llm_router.complete,
            codebase_query=codebase_query,
        )
        components["planner"] = planner
        logger.info("planner_initialized")
    except Exception as e:
        logger.warning("planner_initialization_failed", error=str(e))
        components["planner"] = None

    # 5c. CodeGenerator
    try:
        logger.info("initializing_code_generator")
        generator = CodeGenerator(llm_complete=llm_router.complete)
        components["generator"] = generator
        logger.info("code_generator_initialized")
    except Exception as e:
        logger.warning("code_generator_initialization_failed", error=str(e))
        components["generator"] = None

    # 5d. TestGenerator
    try:
        logger.info("initializing_test_generator")
        test_generator = TestGenerator(llm_complete=llm_router.complete)
        components["test_generator"] = test_generator
        logger.info("test_generator_initialized")
    except Exception as e:
        logger.warning("test_generator_initialization_failed", error=str(e))
        components["test_generator"] = None

    # 5e. CodeReviewer
    try:
        logger.info("initializing_code_reviewer")
        reviewer = CodeReviewer(llm_complete=llm_router.complete)
        components["reviewer"] = reviewer
        logger.info("code_reviewer_initialized")
    except Exception as e:
        logger.warning("code_reviewer_initialization_failed", error=str(e))
        components["reviewer"] = None

    # 6. Initialize Error Handling Components
    try:
        logger.info("initializing_error_classifier")
        error_classifier = ErrorClassifier()
        components["error_classifier"] = error_classifier
        logger.info("error_classifier_initialized")
    except Exception as e:
        logger.warning("error_classifier_initialization_failed", error=str(e))
        components["error_classifier"] = None

    try:
        logger.info("initializing_recovery_manager")
        recovery_manager = RecoveryManager()
        components["recovery_manager"] = recovery_manager
        logger.info("recovery_manager_initialized")
    except Exception as e:
        logger.warning("recovery_manager_initialization_failed", error=str(e))
        components["recovery_manager"] = None

    # 7. Initialize Deployer
    tool_layer = components.get("tool_layer")
    try:
        logger.info("initializing_deployer")
        deployer = Deployer(
            git_client=tool_layer.git if tool_layer else None,
            build_runner=tool_layer.build_runner if tool_layer else None,
            shell_executor=tool_layer.shell if tool_layer else None,
            config=config.project,
        )
        components["deployer"] = deployer
        logger.info("deployer_initialized")
    except Exception as e:
        logger.warning("deployer_initialization_failed", error=str(e))
        components["deployer"] = None

    # 8. Initialize Verifier
    try:
        logger.info("initializing_verifier")
        verifier = Verifier(
            test_runner=tool_layer.test_runner if tool_layer else None,
            build_runner=tool_layer.build_runner if tool_layer else None,
            shell_executor=tool_layer.shell if tool_layer else None,
            config=config,
        )
        components["verifier"] = verifier
        logger.info("verifier_initialized")
    except Exception as e:
        logger.warning("verifier_initialization_failed", error=str(e))
        components["verifier"] = None

    # 9. Initialize Security Scanner
    try:
        logger.info("initializing_security_scanner")
        security_scanner = SecurityScanner(
            project_dir=config.project.project_dir,
        )
        components["security_scanner"] = security_scanner
        logger.info("security_scanner_initialized")
    except Exception as e:
        logger.warning("security_scanner_initialization_failed", error=str(e))
        components["security_scanner"] = None

    # 10. Initialize Quality Gate Engine
    try:
        logger.info("initializing_quality_gate_engine")
        quality_gate_engine = QualityGateEngine(
            project_dir=config.project.project_dir,
        )
        components["quality_gate_engine"] = quality_gate_engine
        logger.info("quality_gate_engine_initialized")
    except Exception as e:
        logger.warning("quality_gate_engine_initialization_failed", error=str(e))
        components["quality_gate_engine"] = None

    # 11. Initialize Evidence Bundle Generator
    try:
        logger.info("initializing_evidence_bundle_generator")
        dep_graph = (
            codebase_intelligence.dependency_graph
            if codebase_intelligence and hasattr(codebase_intelligence, "dependency_graph")
            else None
        )
        evidence_generator = EvidenceBundleGenerator(
            project_dir=config.project.project_dir,
            dependency_graph=dep_graph,
        )
        components["evidence_generator"] = evidence_generator
        logger.info("evidence_bundle_generator_initialized")
    except Exception as e:
        logger.warning("evidence_bundle_generator_initialization_failed", error=str(e))
        components["evidence_generator"] = None

    # 12. Initialize Post-Mortem Learning Engine
    try:
        logger.info("initializing_post_mortem_engine")
        post_mortem_engine = PostMortemEngine(
            memory_store=memory_store,
            project_id=config.project.project_id,
        )
        # Load persisted learnings if available
        learning_file = Path(config.project.project_dir) / ".agent_learnings.json"
        if learning_file.exists():
            post_mortem_engine.load(str(learning_file))
            logger.info("learnings_loaded", path=str(learning_file))
        components["post_mortem_engine"] = post_mortem_engine
        logger.info("post_mortem_engine_initialized")
    except Exception as e:
        logger.warning("post_mortem_engine_initialization_failed", error=str(e))
        components["post_mortem_engine"] = None

    # 13. Initialize Ownership Map
    try:
        logger.info("initializing_ownership_map")
        ownership_map = OwnershipMap(
            project_dir=config.project.project_dir,
        )
        components["ownership_map"] = ownership_map
        logger.info("ownership_map_initialized")
    except Exception as e:
        logger.warning("ownership_map_initialization_failed", error=str(e))
        components["ownership_map"] = None

    # 14. Initialize Cross-Repo Index
    try:
        logger.info("initializing_cross_repo_index")
        cross_repo_index = CrossRepoIndex()
        # Load persisted index if available
        index_file = Path(config.project.project_dir) / ".agent_cross_repo_index.json"
        if index_file.exists():
            cross_repo_index.load(str(index_file))
            logger.info("cross_repo_index_loaded", path=str(index_file))
        components["cross_repo_index"] = cross_repo_index
        logger.info("cross_repo_index_initialized")
    except Exception as e:
        logger.warning("cross_repo_index_initialization_failed", error=str(e))
        components["cross_repo_index"] = None

    # 15. Initialize Resource Limiter
    try:
        logger.info("initializing_resource_limiter")
        standard_limits = ResourceLimiter.from_template("standard")
        resource_limiter = ResourceLimiter(limits=standard_limits)
        components["resource_limiter"] = resource_limiter
        logger.info("resource_limiter_initialized")
    except Exception as e:
        logger.warning("resource_limiter_initialization_failed", error=str(e))
        components["resource_limiter"] = None

    # 16. Initialize SSO/SCIM Authentication
    try:
        logger.info("initializing_auth")
        sso_provider = SSOProvider()
        scim_provider = SCIMProvider()
        # Load persisted SCIM data if available
        scim_file = Path(config.project.project_dir) / ".agent_scim_data"
        if scim_file.exists():
            scim_provider.load(str(scim_file))
            logger.info("scim_data_loaded", path=str(scim_file))
        auth_middleware = AuthMiddleware(
            sso_provider=sso_provider,
            scim_provider=scim_provider,
        )
        components["sso_provider"] = sso_provider
        components["scim_provider"] = scim_provider
        components["auth_middleware"] = auth_middleware
        logger.info("auth_initialized")
    except Exception as e:
        logger.warning("auth_initialization_failed", error=str(e))
        components["sso_provider"] = None
        components["scim_provider"] = None
        components["auth_middleware"] = None

    logger.info(
        "components_initialization_complete",
        initialized_count=sum(1 for v in components.values() if v is not None),
        total_components=len(components),
    )

    return components


async def start_agent(config_overrides: Optional[dict] = None) -> None:
    """Start the Autonomous Software Development Agent.

    Initializes all components in order:
    1. Structlog logger with Rich console
    2. MemoryStore (with persistence loading)
    3. LLM providers + ModelRouter
    4. ToolLayer (FileSystem, Shell, Git, PackageManager, TestRunner, BuildRunner)
    5. CodebaseIntelligence
    6. ContextManager
    7. StoryAnalyzer, Planner, CodeGenerator, TestGenerator, CodeReviewer
    8. ErrorClassifier, RecoveryManager
    9. Orchestrator (wires everything)
    10. FastAPI app with routes

    Starts uvicorn server on configurable port (default 3000).
    Handles graceful shutdown (SIGINT, SIGTERM).
    Logs startup banner with version, providers, project dir.

    Args:
        config_overrides: Optional dict to override config values.
    """
    config_overrides = config_overrides or {}

    # 1. Initialize logger
    _setup_logging()
    logger.info("agent_startup_begin", version="3.1.0")

    # 2. Load and validate config
    config = load_config(config_overrides)
    errors = validate_config(config)
    if errors:
        for error in errors:
            logger.error("config_validation_error", error=error)
        logger.error("agent_startup_failed", reason="Invalid configuration")
        return

    logger.info(
        "config_loaded",
        project_id=config.project.project_id,
        project_dir=config.project.project_dir,
        language=config.project.language,
    )

    # 3. Initialize MemoryStore
    memory_store = MemoryStore(max_entries_per_tier=500)

    # Load persisted memory if available
    memory_file = Path(config.project.project_dir) / ".agent_memory.json"
    if memory_file.exists():
        memory_store.load(str(memory_file))
        logger.info("memory_loaded", path=str(memory_file))

    # 4-8. Initialize all components using helper
    components = _initialize_components(config, memory_store)

    # 9. Wrap components with adapters to bridge Protocol interfaces
    adapted = wrap_components(components, memory_store)

    # 10. Initialize Orchestrator with adapted components
    orchestrator = Orchestrator(
        config=config,
        analyzer=adapted.get("analyzer"),
        planner=adapted.get("planner"),
        generator=adapted.get("generator"),
        reviewer=adapted.get("reviewer"),
        test_generator=adapted.get("test_generator"),
        build_runner=adapted.get("build_runner"),
        test_runner=adapted.get("test_runner"),
        deployer=adapted.get("deployer"),
        context_manager=adapted.get("context_manager"),
        memory_store=adapted.get("memory_store"),
        error_classifier=adapted.get("error_classifier"),
        recovery_manager=adapted.get("recovery_manager"),
        llm_router=adapted.get("llm_router"),
    )

    # Store verifier on orchestrator for _run_verification
    orchestrator._verifier = adapted.get("verifier")

    # Attach new Phase 2+ components to orchestrator
    orchestrator._security_scanner = components.get("security_scanner")
    orchestrator._quality_gate_engine = components.get("quality_gate_engine")
    orchestrator._evidence_generator = components.get("evidence_generator")
    orchestrator._post_mortem_engine = components.get("post_mortem_engine")
    orchestrator._ownership_map = components.get("ownership_map")
    orchestrator._cross_repo_index = components.get("cross_repo_index")
    orchestrator._resource_limiter = components.get("resource_limiter")
    orchestrator._auth_middleware = components.get("auth_middleware")

    # 11. Create FastAPI app
    story_store: dict[str, Story] = {}
    config_holder: dict = {"config": config}
    activity_log: list = []

    # Initialize approval gate manager
    from fastcoder.orchestrator.approval_gate import ApprovalGateManager

    approval_manager = ApprovalGateManager(config.safety)

    # 12. Create Admin app (runs on separate port via main app lifespan)
    main_port = config_overrides.get("port", 3000)
    admin_port = int(os.environ.get("AGENT_ADMIN_PORT", config_overrides.get("admin_port", 3001)))

    admin_app = create_admin_app(
        config_holder=config_holder,
        main_port=main_port,
    )

    app = create_app(
        orchestrator,
        story_store,
        config_holder,
        activity_log=activity_log,
        approval_manager=approval_manager,
        admin_app=admin_app,
    )

    # Mount SCIM 2.0 routes if SCIM provider is available
    scim_provider = components.get("scim_provider")
    if scim_provider:
        # Reuse the agent's API token for SCIM bearer auth
        from fastcoder.api import _get_api_token
        set_scim_bearer_token(_get_api_token())
        scim_router = create_scim_router(scim_provider)
        app.include_router(scim_router)
        logger.info("scim_routes_mounted")

    # Print startup banner
    _print_startup_banner(config, main_port=main_port, admin_port=admin_port)

    # Setup graceful shutdown
    shutdown_event = asyncio.Event()

    def signal_handler(signum, frame):
        logger.info("shutdown_signal_received", signal=signum)
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start main server (admin server starts via lifespan)
    log_level = config.observability.log_level.lower()

    server_config = uvicorn.Config(
        app, host="0.0.0.0", port=main_port, log_level=log_level,
    )
    server = uvicorn.Server(server_config)

    logger.info("starting_servers", main_port=main_port, admin_port=admin_port)

    server_task = asyncio.create_task(server.serve())

    try:
        # Wait for shutdown signal
        await shutdown_event.wait()
    except KeyboardInterrupt:
        logger.info("keyboard_interrupt_received")

    # Graceful shutdown
    logger.info("shutting_down")
    server.should_exit = True

    # Save memory before exit
    memory_store.save(str(memory_file))
    logger.info("memory_saved", path=str(memory_file))

    # Save learnings
    post_mortem = components.get("post_mortem_engine")
    if post_mortem:
        learning_file = Path(config.project.project_dir) / ".agent_learnings.json"
        post_mortem.save(str(learning_file))
        logger.info("learnings_saved", path=str(learning_file))

    # Save cross-repo index
    cross_repo = components.get("cross_repo_index")
    if cross_repo:
        index_file = Path(config.project.project_dir) / ".agent_cross_repo_index.json"
        cross_repo.save(str(index_file))
        logger.info("cross_repo_index_saved", path=str(index_file))

    # Save SCIM data
    scim = components.get("scim_provider")
    if scim:
        scim_file = Path(config.project.project_dir) / ".agent_scim_data"
        scim.save(str(scim_file))
        logger.info("scim_data_saved", path=str(scim_file))

    # Wait for server to finish (admin server shuts down via lifespan)
    try:
        await asyncio.wait_for(server_task, timeout=5.0)
    except asyncio.TimeoutError:
        logger.warning("server_shutdown_timeout")

    logger.info("agent_shutdown_complete", version="3.1.0")


def _setup_logging() -> None:
    """Configure structlog with Rich console output and JSON fallback."""
    timestamper = TimeStamper(fmt="iso")

    shared_processors = [
        timestamper,
        structlog.stdlib.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=shared_processors
        + [
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _print_startup_banner(config: AgentConfig, main_port: int = 3000, admin_port: int = 3001) -> None:
    """Print startup banner with key configuration.

    Args:
        config: AgentConfig instance.
        main_port: Port for the workspace/API server.
        admin_port: Port for the admin panel server.
    """
    enabled_providers = [
        p.name for p in config.llm.providers if p.enabled
    ]

    providers_display = ', '.join(enabled_providers) if enabled_providers else "(none — add keys via Admin Panel)"

    banner = f"""
╔════════════════════════════════════════════════════════════════╗
║   Autonomous Software Development Agent v3.1.0                ║
╠════════════════════════════════════════════════════════════════╣
║ Project:           {config.project.project_id:<35} ║
║ Language:          {config.project.language:<35} ║
║ Providers:         {providers_display:<35} ║
║ Max Iterations:    {str(config.safety.max_iterations_per_story):<35} ║
║ Quality Coverage:  {str(config.quality.min_test_coverage) + '%':<35} ║
║ Cost Limit/Story:  ${config.llm.cost.max_cost_per_story_usd:<34} ║
╠════════════════════════════════════════════════════════════════╣
║ Workspace:         http://localhost:{main_port:<29}║
║ Admin Panel:       http://localhost:{admin_port:<29}║
╚════════════════════════════════════════════════════════════════╝
"""
    print(banner)

    if not enabled_providers:
        logger.info(
            "no_llm_providers_configured",
            hint=f"Go to http://localhost:{admin_port}/admin to add provider API keys. Keys are stored in the admin DB — no environment variables needed.",
        )

    logger.info("startup_banner_printed")


def create_uvicorn_app():
    """Factory function for uvicorn --factory mode.

    Creates the FastAPI app with full initialization (config, orchestrator,
    approval manager) without starting the async event loop or uvicorn server.
    Used by run.sh and direct uvicorn invocations.

    Returns:
        Configured FastAPI application.
    """
    _setup_logging()
    logger.info("app_factory_start", version="3.1.0")

    config = load_config()
    errors = validate_config(config)
    if errors:
        for error in errors:
            logger.error("config_validation_error", error=error)

    memory_store = MemoryStore(max_entries_per_tier=500)

    memory_file = Path(config.project.project_dir) / ".agent_memory.json"
    if memory_file.exists():
        memory_store.load(str(memory_file))

    # Initialize all components using helper
    components = _initialize_components(config, memory_store)

    # Wrap with adapters
    adapted = wrap_components(components, memory_store)

    orchestrator = Orchestrator(
        config=config,
        analyzer=adapted.get("analyzer"),
        planner=adapted.get("planner"),
        generator=adapted.get("generator"),
        reviewer=adapted.get("reviewer"),
        test_generator=adapted.get("test_generator"),
        build_runner=adapted.get("build_runner"),
        test_runner=adapted.get("test_runner"),
        deployer=adapted.get("deployer"),
        context_manager=adapted.get("context_manager"),
        memory_store=adapted.get("memory_store"),
        error_classifier=adapted.get("error_classifier"),
        recovery_manager=adapted.get("recovery_manager"),
        llm_router=adapted.get("llm_router"),
    )

    # Store verifier on orchestrator for _run_verification
    orchestrator._verifier = adapted.get("verifier")

    # Attach new Phase 2+ components to orchestrator
    orchestrator._security_scanner = components.get("security_scanner")
    orchestrator._quality_gate_engine = components.get("quality_gate_engine")
    orchestrator._evidence_generator = components.get("evidence_generator")
    orchestrator._post_mortem_engine = components.get("post_mortem_engine")
    orchestrator._ownership_map = components.get("ownership_map")
    orchestrator._cross_repo_index = components.get("cross_repo_index")
    orchestrator._resource_limiter = components.get("resource_limiter")
    orchestrator._auth_middleware = components.get("auth_middleware")

    story_store: dict[str, Story] = {}
    config_holder: dict = {"config": config}
    activity_log: list = []

    from fastcoder.orchestrator.approval_gate import ApprovalGateManager

    approval_manager = ApprovalGateManager(config.safety)

    # Build admin app first so we can pass it to create_app
    main_port = int(os.environ.get("AGENT_PORT", "3000"))
    admin_port = int(os.environ.get("AGENT_ADMIN_PORT", "3001"))

    admin_app = create_admin_app(
        config_holder=config_holder,
        main_port=main_port,
    )

    app = create_app(
        orchestrator,
        story_store,
        config_holder,
        activity_log=activity_log,
        approval_manager=approval_manager,
        admin_app=admin_app,
    )

    # Mount SCIM 2.0 routes if SCIM provider is available
    scim_provider = components.get("scim_provider")
    if scim_provider:
        from fastcoder.api import _get_api_token
        set_scim_bearer_token(_get_api_token())
        scim_router = create_scim_router(scim_provider)
        app.include_router(scim_router)
        logger.info("scim_routes_mounted")

    _print_startup_banner(config, main_port=main_port, admin_port=admin_port)
    return app


def create_uvicorn_admin_app():
    """Factory function for the admin panel uvicorn instance.

    Returns a standalone FastAPI app that serves the admin panel,
    admin config API, and auth routes.  Run it via:

        uvicorn fastcoder.main:create_uvicorn_admin_app --factory --port 3001
    """
    _setup_logging()
    config = load_config()
    config_holder: dict = {"config": config}
    main_port = int(os.environ.get("AGENT_PORT", "3000"))

    admin_app = create_admin_app(
        config_holder=config_holder,
        main_port=main_port,
    )
    return admin_app


def main() -> None:
    """Synchronous entry point that calls asyncio.run(start_agent())."""
    asyncio.run(start_agent())


if __name__ == "__main__":
    main()
