#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="${ROOT_DIR}/.run"
PID_FILE="${RUN_DIR}/fastcoder.pid"
LOG_FILE="${RUN_DIR}/fastcoder.log"

PORT="${PORT:-3000}"
HOST="${HOST:-0.0.0.0}"
WORKERS="${WORKERS:-1}"
LOG_LEVEL="${LOG_LEVEL:-info}"

ensure_run_dir() {
  mkdir -p "${RUN_DIR}"
}

is_running() {
  if [[ ! -f "${PID_FILE}" ]]; then
    return 1
  fi
  local pid
  pid="$(cat "${PID_FILE}")"
  if [[ -z "${pid}" ]]; then
    return 1
  fi
  kill -0 "${pid}" 2>/dev/null
}

require_python() {
  if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required but not found" >&2
    exit 1
  fi
}

ensure_pythonpath() {
  export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
}

print_runtime() {
  echo "ROOT_DIR=${ROOT_DIR}"
  echo "HOST=${HOST}"
  echo "PORT=${PORT}"
  echo "WORKERS=${WORKERS}"
  echo "LOG_LEVEL=${LOG_LEVEL}"
  echo "PID_FILE=${PID_FILE}"
  echo "LOG_FILE=${LOG_FILE}"
}
