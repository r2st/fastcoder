# fastcoder

> **Autonomous Software Development Agent** — accepts user stories and autonomously plans, writes, tests, reviews, and deploys code.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

`fastcoder` is a self-hostable, multi-LLM autonomous development agent. Submit a user story, approve at gates, and the agent plans the work, writes the code, runs tests, opens a PR, and ships it.

---

## Features

- **Multi-LLM routing** — Anthropic, OpenAI, Gemini, and Ollama, with circuit-breakers and a tiered router
- **Story-driven workflow** — `Story → Plan → Iteration → Review → Deploy`, fully traceable
- **Configurable approval gates** — 7 human-in-the-loop checkpoints (plan, code, tests, security, deploy, …)
- **Durable orchestration** — write-ahead log, checkpoints, retries, convergence detection
- **Codebase-aware** — AST indexer, symbol table, dependency graph, semantic search, ownership map
- **Built-in auth** — local users, SSO (OIDC/GitHub), SCIM provisioning, RBAC middleware
- **First-class admin UI** — workspace, admin panel, and login UIs ship in-repo
- **Observable** — structured logs, Prometheus metrics, health/readiness, ops routes
- **Self-improving** — learning module captures error/recovery patterns across runs

## Quickstart

```bash
# 1. Clone & install
git clone https://github.com/r2st/fastcoder.git
cd fastcoder
pip install -e ".[dev]"

# 2. Run
./run.sh                 # starts on http://localhost:3000
# or:
fastcoder           # if installed as a package
```

On first start the agent generates super-admin credentials and writes them to `.agent_super_admin.json` (gitignored). Open the admin panel at `http://localhost:3000/admin` to add LLM provider keys and configure the project.

### Docker

```bash
cp deploy/docker/.env.prod.example deploy/docker/.env.prod   # edit values
cd deploy/docker
./up.sh
```

## Configuration

All runtime configuration lives in the **admin panel UI** (encrypted at rest in `.agent_admin.db`). Optional initial defaults can be supplied via env vars — see [`deploy/docker/.env.prod.example`](deploy/docker/.env.prod.example).

| Env var                    | Purpose                                              |
| -------------------------- | ---------------------------------------------------- |
| `AGENT_API_TOKEN`          | Pre-set headless token (auto-generated if omitted)   |
| `AGENT_CORS_ORIGINS`       | Comma-separated allowed origins                      |
| `AGENT_KEY_ENCRYPTION_KEY` | Master key for at-rest encryption (recommended prod) |
| `AGENT_PROJECT_DIR`        | Target project directory (default `.`)               |
| `AGENT_LOG_LEVEL`          | `debug` \| `info` \| `warning` \| `error`            |

## Architecture

```
┌──────────┐    ┌─────────┐    ┌───────────┐    ┌──────────┐    ┌─────────┐
│ Analyzer │ -> │ Planner │ -> │ Generator │ -> │ Reviewer │ -> │ Deployer│
└──────────┘    └─────────┘    └───────────┘    └──────────┘    └─────────┘
       \             |               |               |              /
        \____________|_______________|_______________|_____________/
                                     |
                            ┌────────▼─────────┐
                            │   Orchestrator   │  state machine, WAL,
                            │   + Approval     │  checkpoints, retries
                            │     Gates        │
                            └──────────────────┘
```

Core abstractions: `Story`, `StorySpec`, `PlanTask`, `Iteration`, `ErrorContext`. See [`docs/project-specification.html`](docs/project-specification.html) for the full design.

## Development

```bash
make install       # install deps + pre-commit hooks
make test          # pytest with coverage (fail-under 70%)
make lint          # ruff + mypy
make format        # ruff format
```

## Project layout

```
src/fastcoder/
├── analyzer/       story → spec
├── planner/        spec → plan
├── generator/      plan → code changes
├── reviewer/       static + LLM code review
├── tester/         test execution
├── verifier/       quality gates
├── deployer/       branch / PR / release flow
├── orchestrator/   state machine, WAL, gates, retries
├── llm/            multi-provider router + circuit breaker
├── codebase/       AST indexer, symbol table, dep graph
├── memory/         persistent agent memory
├── learning/       failure/recovery pattern store
├── auth/           local + SSO + SCIM + middleware
├── api/            FastAPI routes (admin, ops, health)
├── tools/          git, shell, pkg manager, build/test runners
└── types/          typed domain models
```

## Contributing

Contributions are very welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md) before opening a PR.

## Security

If you find a security issue, **please do not open a public issue**. See [SECURITY.md](SECURITY.md) for responsible disclosure.

## License

[MIT](LICENSE) © Suman
