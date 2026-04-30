#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

url="http://127.0.0.1:${PORT}/health"
echo "[health] ${url}"
curl -fsS "${url}" && echo
