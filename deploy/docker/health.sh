#!/usr/bin/env bash
set -euo pipefail

url="http://127.0.0.1:3000/health"
echo "[health] ${url}"
curl -fsS "${url}" && echo
