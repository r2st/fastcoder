#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

if is_running; then
  pid="$(cat "${PID_FILE}")"
  echo "status: running"
  echo "pid: ${pid}"
  echo "port: ${PORT}"
  echo "log: ${LOG_FILE}"
  exit 0
fi

echo "status: stopped"
echo "pid_file: ${PID_FILE}"
exit 1
