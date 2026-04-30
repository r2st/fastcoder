#!/usr/bin/env bash
#
# run.sh — Startup script for the Autonomous Software Development Agent
#
# Usage:
#   ./run.sh                     # Start with defaults (port 3000)
#   ./run.sh --port 8080         # Custom port
#   ./run.sh --dev               # Dev mode (auto-reload, debug logging)
#   ./run.sh --help              # Show help
#
# Environment Variables:
#   AGENT_ADMIN_DB_PATH — Admin DB path (stores provider keys/config)
#   AGENT_PROJECT_DIR   — Target project directory (default: .)
#   AGENT_CONFIG_FILE   — Path to config file (default: .agent.json)
#   AGENT_LOG_LEVEL     — Log level: debug, info, warning, error (default: info)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ── Defaults ──
PORT=3000
HOST="0.0.0.0"
DEV_MODE=false
LOG_LEVEL="${AGENT_LOG_LEVEL:-info}"
WORKERS=1

# ── Parse Arguments ──
show_help() {
    echo -e "${BOLD}Autonomous Software Development Agent v3.1.0${NC}"
    echo ""
    echo -e "Usage: ${CYAN}./run.sh [OPTIONS]${NC}"
    echo ""
    echo "Options:"
    echo "  --port PORT       Server port (default: 3000)"
    echo "  --host HOST       Server host (default: 0.0.0.0)"
    echo "  --dev             Development mode (auto-reload, debug logging)"
    echo "  --workers N       Number of uvicorn workers (default: 1)"
    echo "  --log-level LVL   Log level: debug, info, warning, error"
    echo "  --check           Check dependencies and config, then exit"
    echo "  --help            Show this help message"
    echo ""
    echo "Frontends (served automatically):"
    echo "  http://localhost:PORT/          Workspace UI (submit instructions, track stories)"
    echo "  http://localhost:PORT/admin     Admin Panel (manage configuration)"
    echo ""
    echo "API Docs:"
    echo "  http://localhost:PORT/docs      Swagger UI"
    echo "  http://localhost:PORT/redoc     ReDoc"
    echo ""
    echo "Environment Variables:"
    echo "  AGENT_ADMIN_DB_PATH  Admin DB path (provider keys + config)"
    echo "  AGENT_PROJECT_DIR    Target project directory"
    echo "  AGENT_CONFIG_FILE    Config file path (default: .agent.json)"
    echo "  AGENT_LOG_LEVEL      Log level (default: info)"
    exit 0
}

CHECK_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --port)   PORT="$2"; shift 2 ;;
        --host)   HOST="$2"; shift 2 ;;
        --dev)    DEV_MODE=true; shift ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --log-level) LOG_LEVEL="$2"; shift 2 ;;
        --check)  CHECK_ONLY=true; shift ;;
        --help)   show_help ;;
        *)        echo -e "${RED}Unknown option: $1${NC}"; show_help ;;
    esac
done

# ── Banner ──
echo -e "${BOLD}${CYAN}"
cat << 'BANNER'
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     █████╗ ██╗   ██╗████████╗ ██████╗                        ║
║    ██╔══██╗██║   ██║╚══██╔══╝██╔═══██╗                       ║
║    ███████║██║   ██║   ██║   ██║   ██║                       ║
║    ██╔══██║██║   ██║   ██║   ██║   ██║                       ║
║    ██║  ██║╚██████╔╝   ██║   ╚██████╔╝                       ║
║    ╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝                       ║
║                                                              ║
║    ██████╗ ███████╗██╗   ██╗     █████╗  ██████╗ ████████╗   ║
║    ██╔══██╗██╔════╝██║   ██║    ██╔══██╗██╔════╝ ╚══██╔══╝   ║
║    ██║  ██║█████╗  ██║   ██║    ███████║██║  ███╗   ██║      ║
║    ██║  ██║██╔══╝  ╚██╗ ██╔╝    ██╔══██║██║   ██║   ██║      ║
║    ██████╔╝███████╗ ╚████╔╝     ██║  ██║╚██████╔╝   ██║      ║
║    ╚═════╝ ╚══════╝  ╚═══╝      ╚═╝  ╚═╝ ╚═════╝    ╚═╝      ║
║                                                              ║
║    Autonomous Software Development Agent  v3.1.0             ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
BANNER
echo -e "${NC}"

# ── Check Python ──
echo -e "${BOLD}[1/5] Checking Python...${NC}"
if ! command -v python3 &>/dev/null; then
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.11+${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [[ "$PYTHON_MAJOR" -lt 3 ]] || { [[ "$PYTHON_MAJOR" -eq 3 ]] && [[ "$PYTHON_MINOR" -lt 11 ]]; }; then
    echo -e "${RED}✗ Python 3.11+ required, found $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"

# ── Check Dependencies ──
echo -e "${BOLD}[2/5] Checking dependencies...${NC}"
MISSING_DEPS=()

check_module() {
    python3 -c "import $1" 2>/dev/null || MISSING_DEPS+=("$2")
}

check_distribution() {
    python3 - <<PY 2>/dev/null || MISSING_DEPS+=("$1")
import importlib.metadata
importlib.metadata.version("$1")
PY
}

check_module "fastapi" "fastapi"
check_module "uvicorn" "uvicorn[standard]"
check_module "pydantic" "pydantic"
check_module "anthropic" "anthropic"
check_module "openai" "openai"
check_module "google.genai" "google-genai"
check_module "ollama" "ollama"
check_distribution "gitpython"
check_module "aiofiles" "aiofiles"
check_module "httpx" "httpx"
check_module "structlog" "structlog"
check_module "rich" "rich"

if [[ ${#MISSING_DEPS[@]} -gt 0 ]]; then
    echo -e "${YELLOW}⚠ Missing packages: ${MISSING_DEPS[*]}${NC}"
    echo -e "${CYAN}Installing...${NC}"
    pip install "${MISSING_DEPS[@]}" --break-system-packages -q 2>/dev/null || \
    pip install "${MISSING_DEPS[@]}" -q 2>/dev/null || {
        echo -e "${RED}✗ Failed to install dependencies. Run manually:${NC}"
        echo "  pip install ${MISSING_DEPS[*]}"
        exit 1
    }
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${GREEN}✓ All dependencies present${NC}"
fi

# ── Check Package Install ──
echo -e "${BOLD}[3/5] Checking package...${NC}"
if ! python3 -c "import fastcoder" 2>/dev/null; then
    # Try with PYTHONPATH
    export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}"
    if ! python3 -c "import fastcoder" 2>/dev/null; then
        echo -e "${YELLOW}⚠ Package not installed. Installing in development mode...${NC}"
        pip install -e . --break-system-packages -q 2>/dev/null || \
        pip install -e . -q 2>/dev/null || {
            echo -e "${YELLOW}⚠ pip install -e failed, using PYTHONPATH fallback${NC}"
        }
    fi
fi
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}"
echo -e "${GREEN}✓ Package accessible${NC}"

# ── Check LLM Providers ──
echo -e "${BOLD}[4/5] Checking LLM providers...${NC}"
DB_PATH="${AGENT_ADMIN_DB_PATH:-${SCRIPT_DIR}/.agent_admin.db}"
if [[ -f "${DB_PATH}" ]]; then
    echo -e "${GREEN}  ✓ Admin DB found: ${DB_PATH}${NC}"
else
    echo -e "${YELLOW}  ○ Admin DB not found yet: ${DB_PATH}${NC}"
    echo -e "${YELLOW}    It will be created automatically on first key/config write from Admin UI.${NC}"
fi

OLLAMA_INFO="$(python3 - <<'PY'
from fastcoder.config import load_config

cfg = load_config()
provider = next((p for p in cfg.llm.providers if p.name == "ollama"), None)
if not provider:
    print("missing")
else:
    print(("enabled" if provider.enabled else "disabled") + "|" + (provider.base_url or "http://localhost:11434"))
PY
)"

if [[ "${OLLAMA_INFO}" == "missing" ]]; then
    echo -e "${YELLOW}  ○ Ollama provider missing from config${NC}"
else
    OLLAMA_STATE="${OLLAMA_INFO%%|*}"
    OLLAMA_URL="${OLLAMA_INFO#*|}"
    if [[ "${OLLAMA_STATE}" == "enabled" ]]; then
        if curl -s --connect-timeout 2 "${OLLAMA_URL}/api/tags" &>/dev/null; then
            echo -e "${GREEN}  ✓ Ollama — enabled and reachable at ${OLLAMA_URL}${NC}"
        else
            echo -e "${YELLOW}  ⚠ Ollama — enabled in admin config but not reachable at ${OLLAMA_URL}${NC}"
        fi
    else
        echo -e "${YELLOW}  ○ Ollama — disabled in admin config${NC}"
    fi
fi

# ── Check Frontend Files ──
echo -e "${BOLD}[5/5] Checking frontend files...${NC}"

if [[ -f "${SCRIPT_DIR}/workspace-ui/index.html" ]]; then
    echo -e "${GREEN}  ✓ Workspace UI ready${NC}"
else
    echo -e "${RED}  ✗ Workspace UI not found at workspace-ui/index.html${NC}"
fi

if [[ -f "${SCRIPT_DIR}/admin-panel/index.html" ]]; then
    echo -e "${GREEN}  ✓ Admin Panel ready${NC}"
else
    echo -e "${RED}  ✗ Admin Panel not found at admin-panel/index.html${NC}"
fi

# ── Check Only Mode ──
if [[ "$CHECK_ONLY" == "true" ]]; then
    echo ""
    echo -e "${GREEN}${BOLD}All checks passed.${NC}"
    exit 0
fi

# ── Set Log Level ──
export AGENT_LOG_LEVEL="$LOG_LEVEL"

# ── Dev Mode Adjustments ──
RELOAD_FLAG=""
if [[ "$DEV_MODE" == "true" ]]; then
    RELOAD_FLAG="--reload --reload-dir src"
    LOG_LEVEL="debug"
    export AGENT_LOG_LEVEL="debug"
    echo -e "${YELLOW}⚡ Development mode: auto-reload ON, log level DEBUG${NC}"
fi

# ── Ports ──
ADMIN_PORT="${AGENT_ADMIN_PORT:-3001}"

# ── Start Server ──
echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}${BOLD}  Starting Autonomous Dev Agent${NC}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  ${CYAN}Workspace:${NC}      http://localhost:${PORT}/"
echo -e "  ${CYAN}Admin Panel:${NC}    http://localhost:${ADMIN_PORT}/admin"
echo -e "  ${CYAN}API Docs:${NC}       http://localhost:${PORT}/docs"
echo -e "  ${CYAN}Health Check:${NC}   http://localhost:${PORT}/health"
echo ""
echo -e "  ${YELLOW}Press Ctrl+C to stop${NC}"
echo ""

exec python3 -m uvicorn \
    "fastcoder.main:create_uvicorn_app" \
    --factory \
    --host "$HOST" \
    --port "$PORT" \
    --log-level "$LOG_LEVEL" \
    --workers "$WORKERS" \
    $RELOAD_FLAG
