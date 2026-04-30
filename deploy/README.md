# Deployment Scripts

This folder provides a complete deployment workflow for the Auto Dev Agent.

## Quick start (local process)

```bash
cd /path/to/fastcoder
./deploy/local/preflight.sh
./deploy/local/start.sh
./deploy/local/status.sh
./deploy/local/health.sh
```

Stop:

```bash
./deploy/local/stop.sh
```

## Quick start (Docker)

```bash
cd /path/to/fastcoder
cp deploy/docker/.env.prod.example deploy/docker/.env.prod
./deploy/docker/up.sh
./deploy/docker/status.sh
./deploy/docker/health.sh
```

Stop:

```bash
./deploy/docker/down.sh
```

## Notes

- Local scripts write runtime files to `.run/`:
  - PID: `.run/fastcoder.pid`
  - Logs: `.run/fastcoder.log`
- All configuration (project settings, LLM providers, routing, cost budgets, safety, quality, observability) is managed through the Admin Panel UI and persisted to `.agent.json`.
- LLM API keys are stored securely in the admin database (`.agent_admin.db` by default).
- No `.env` file is required. Docker deployments may optionally use `deploy/docker/.env.prod` for container-level overrides (API token, CORS, encryption key).
