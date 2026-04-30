#!/usr/bin/env python3
"""
run.py — Cross-platform startup script for the Autonomous Software Development Agent.

Usage:
    python run.py                    # Start with defaults (port 3000)
    python run.py --port 8080        # Custom port
    python run.py --dev              # Dev mode (auto-reload, debug logging)
    python run.py --check            # Check dependencies only
    python run.py --help             # Show help

Works on macOS, Linux, and Windows without bash.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path


# ── Colors (ANSI, disabled on non-tty) ──
USE_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def green(t: str) -> str:
    return _c("32", t)


def red(t: str) -> str:
    return _c("31", t)


def yellow(t: str) -> str:
    return _c("33", t)


def cyan(t: str) -> str:
    return _c("36", t)


def bold(t: str) -> str:
    return _c("1", t)


BANNER = r"""
╔══════════════════════════════════════════════════════════╗
║   Autonomous Software Development Agent  v3.1.0         ║
╚══════════════════════════════════════════════════════════╝
"""


def check_python_version() -> bool:
    """Verify Python 3.11+."""
    v = sys.version_info
    ok = v.major >= 3 and v.minor >= 11
    ver = f"{v.major}.{v.minor}.{v.micro}"
    if ok:
        print(green(f"  ✓ Python {ver}"))
    else:
        print(red(f"  ✗ Python 3.11+ required, found {ver}"))
    return ok


def check_dependencies() -> list[str]:
    """Check required Python packages, return list of missing ones."""
    required = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn[standard]",
        "pydantic": "pydantic",
        "anthropic": "anthropic",
        "openai": "openai",
        "google.genai": "google-genai",
        "ollama": "ollama",
        "aiofiles": "aiofiles",
        "httpx": "httpx",
        "structlog": "structlog",
        "rich": "rich",
    }

    missing = []
    for module_name, pip_name in required.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(pip_name)

    # GitPython package check must not import `git`, because that triggers
    # git executable refresh and can fail on macOS if Xcode license is pending.
    try:
        importlib.metadata.version("gitpython")
    except importlib.metadata.PackageNotFoundError:
        missing.append("gitpython")

    if missing:
        print(yellow(f"  ⚠ Missing: {', '.join(missing)}"))
    else:
        print(green("  ✓ All dependencies present"))

    return missing


def install_deps(missing: list[str]) -> bool:
    """Attempt to install missing dependencies."""
    print(cyan(f"  Installing: {' '.join(missing)}"))
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *missing, "-q"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(green("  ✓ Installed successfully"))
        return True
    except subprocess.CalledProcessError:
        print(red("  ✗ Installation failed. Run manually:"))
        print(f"    pip install {' '.join(missing)}")
        return False


def check_package() -> bool:
    """Ensure fastcoder is importable."""
    src_dir = str(Path(__file__).parent / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    os.environ["PYTHONPATH"] = src_dir + os.pathsep + os.environ.get("PYTHONPATH", "")

    try:
        importlib.import_module("fastcoder")
        print(green("  ✓ Package accessible"))
        return True
    except ImportError as e:
        print(red(f"  ✗ Cannot import fastcoder: {e}"))
        return False


def check_providers() -> int:
    """Check LLM provider configuration. Returns count of configured providers."""
    count = 0
    db_path = os.environ.get("AGENT_ADMIN_DB_PATH", str(Path(__file__).parent / ".agent_admin.db"))
    if Path(db_path).exists():
        print(green(f"  ✓ Admin DB found: {db_path}"))
        count += 1
    else:
        print(yellow(f"  ○ Admin DB not found yet: {db_path}"))
        print(yellow("    It will be created automatically on first key/config write from Admin UI."))

    try:
        from fastcoder.config import load_config

        cfg = load_config()
        provider = next((p for p in cfg.llm.providers if p.name == "ollama"), None)
        if provider and provider.enabled:
            print(green(f"  ✓ Ollama — enabled in admin config ({provider.base_url})"))
            count += 1
        else:
            print(yellow("  ○ Ollama — disabled in admin config"))
    except Exception as exc:
        print(yellow(f"  ○ Ollama — unable to read config ({exc})"))

    return count


def check_frontends() -> None:
    """Check frontend HTML files exist."""
    root = Path(__file__).parent
    ws = root / "workspace-ui" / "index.html"
    admin = root / "admin-panel" / "index.html"

    if ws.exists():
        print(green("  ✓ Workspace UI ready"))
    else:
        print(red("  ✗ workspace-ui/index.html not found"))

    if admin.exists():
        print(green("  ✓ Admin Panel ready"))
    else:
        print(red("  ✗ admin-panel/index.html not found"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous Software Development Agent — Startup Script"
    )
    parser.add_argument("--port", type=int, default=3000, help="Server port (default: 3000)")
    parser.add_argument("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--dev", action="store_true", help="Development mode (auto-reload, debug)")
    parser.add_argument("--workers", type=int, default=1, help="Uvicorn workers (default: 1)")
    parser.add_argument(
        "--log-level",
        default=os.environ.get("AGENT_LOG_LEVEL", "info"),
        choices=["debug", "info", "warning", "error"],
        help="Log level",
    )
    parser.add_argument("--check", action="store_true", help="Check dependencies and exit")

    args = parser.parse_args()

    print(bold(cyan(BANNER)))

    # ── Step 1: Python ──
    print(bold("[1/5] Checking Python..."))
    if not check_python_version():
        sys.exit(1)

    # ── Step 2: Dependencies ──
    print(bold("[2/5] Checking dependencies..."))
    missing = check_dependencies()
    if missing:
        if not install_deps(missing):
            sys.exit(1)

    # ── Step 3: Package ──
    print(bold("[3/5] Checking package..."))
    if not check_package():
        sys.exit(1)

    # ── Step 4: Providers ──
    print(bold("[4/5] Checking LLM providers..."))
    check_providers()

    # ── Step 5: Frontends ──
    print(bold("[5/5] Checking frontend files..."))
    check_frontends()

    if args.check:
        print()
        print(green(bold("All checks passed.")))
        sys.exit(0)

    # ── Dev mode overrides ──
    log_level = args.log_level
    if args.dev:
        log_level = "debug"
        print(yellow("\n⚡ Development mode: auto-reload ON, log level DEBUG"))

    # ── Launch ──
    print()
    print(bold("━" * 60))
    print(green(bold(f"  Starting Autonomous Dev Agent on port {args.port}")))
    print(bold("━" * 60))
    print()
    print(f"  {cyan('Workspace UI:')}   http://localhost:{args.port}/")
    print(f"  {cyan('Admin Panel:')}    http://localhost:{args.port}/admin")
    print(f"  {cyan('API Docs:')}       http://localhost:{args.port}/docs")
    print(f"  {cyan('Health Check:')}   http://localhost:{args.port}/health")
    print()
    print(f"  {yellow('Press Ctrl+C to stop')}")
    print()

    os.environ["AGENT_LOG_LEVEL"] = log_level

    import uvicorn

    uvicorn.run(
        "fastcoder.main:create_uvicorn_app",
        factory=True,
        host=args.host,
        port=args.port,
        log_level=log_level,
        workers=args.workers,
        reload=args.dev,
        reload_dirs=["src"] if args.dev else None,
    )


if __name__ == "__main__":
    main()
