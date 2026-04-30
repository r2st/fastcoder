#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

if ! is_running; then
  echo "agent is not running"
  rm -f "${PID_FILE}"
  exit 0
fi

pid="$(cat "${PID_FILE}")"
echo "[stop] stopping pid=${pid}"
kill "${pid}" || true

for _ in {1..20}; do
  if kill -0 "${pid}" 2>/dev/null; then
    sleep 0.25
  else
    break
  fi
done

if kill -0 "${pid}" 2>/dev/null; then
  echo "[stop] force killing pid=${pid}"
  kill -9 "${pid}" || true
fi

rm -f "${PID_FILE}"
echo "[stop] stopped"
