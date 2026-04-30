#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

ensure_run_dir
require_python
ensure_pythonpath

if is_running; then
  echo "agent already running (pid=$(cat "${PID_FILE}"))"
  exit 0
fi

echo "[start] launching fastcoder..."
print_runtime

(
  cd "${ROOT_DIR}"
  nohup python3 -m uvicorn \
    "fastcoder.main:create_uvicorn_app" \
    --factory \
    --host "${HOST}" \
    --port "${PORT}" \
    --workers "${WORKERS}" \
    --log-level "${LOG_LEVEL}" \
    >>"${LOG_FILE}" 2>&1 &
  echo $! > "${PID_FILE}"
)

sleep 1
if is_running; then
  echo "[start] started (pid=$(cat "${PID_FILE}"))"
  echo "[start] logs: ${LOG_FILE}"
else
  echo "[start] failed to start. check logs: ${LOG_FILE}" >&2
  exit 1
fi
