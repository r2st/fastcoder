#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

require_python
ensure_pythonpath
ensure_run_dir

echo "[preflight] checking python version..."
python3 - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11+ is required")
print(f"Python OK: {sys.version.split()[0]}")
PY

echo "[preflight] checking package import..."
python3 - <<'PY'
import fastcoder  # noqa: F401
print("fastcoder import OK")
PY

echo "[preflight] checking uvicorn import..."
python3 - <<'PY'
import uvicorn  # noqa: F401
print("uvicorn import OK")
PY

echo "[preflight] checking frontend artifacts..."
[[ -f "${ROOT_DIR}/workspace-ui/index.html" ]] || { echo "workspace-ui/index.html missing"; exit 1; }
[[ -f "${ROOT_DIR}/admin-panel/index.html" ]] || { echo "admin-panel/index.html missing"; exit 1; }
echo "frontend artifacts OK"

echo "[preflight] checking port availability..."
if lsof -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "port ${PORT} is already in use"
  exit 1
fi
echo "port ${PORT} available"

echo "[preflight] complete"
