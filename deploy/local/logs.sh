#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

ensure_run_dir
touch "${LOG_FILE}"
tail -n 100 -f "${LOG_FILE}"
