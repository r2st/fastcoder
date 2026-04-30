#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.prod.yml"

if [[ ! -f "${SCRIPT_DIR}/.env.prod" ]]; then
  echo ".env.prod not found. creating from .env.prod.example"
  cp "${SCRIPT_DIR}/.env.prod.example" "${SCRIPT_DIR}/.env.prod"
fi

docker compose -f "${COMPOSE_FILE}" --env-file "${SCRIPT_DIR}/.env.prod" up -d --build
echo "docker deployment started"
