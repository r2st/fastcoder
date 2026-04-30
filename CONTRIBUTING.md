# Contributing to fastcoder

Thanks for your interest in contributing! This document explains how to get a working development environment, the conventions used in this repo, and how to submit changes.

## Code of Conduct

By participating in this project, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md). Be kind. Assume good intent.

## Ways to contribute

- **Bug reports** — open an issue with reproduction steps, expected vs. actual behaviour, and your environment.
- **Feature requests** — open an issue describing the use case before opening a large PR.
- **Pull requests** — for non-trivial changes, please discuss in an issue first.
- **Documentation** — typo fixes, clarifications, and missing examples are always appreciated.

## Development setup

```bash
git clone https://github.com/r2st/fastcoder.git
cd fastcoder
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

Run the agent:

```bash
./run.sh --dev
```

## Running checks

The same checks that run in CI:

```bash
make lint     # ruff + mypy (strict)
make test     # pytest with coverage (fail-under 70%)
make format   # ruff format
```

## Conventions

- **Python ≥ 3.10**, type-annotated, `mypy --strict` clean.
- **Lint**: `ruff` with rules `E, F, I, N, W, UP`. Line length 100.
- **Tests**: live in `tests/`, mirror the `src/fastcoder/` layout. Use `pytest-asyncio`.
- **Coverage**: PRs should not drop coverage below 70 %.
- **Commits**: imperative mood, short subject (≤ 72 chars), longer body if needed.
  Example: `auth: add SCIM group provisioning`.
- **Branch names**: `feat/...`, `fix/...`, `docs/...`, `refactor/...`.

## Domain model

Before touching the orchestration core, skim the central abstractions:

- `Story` — user request unit
- `StorySpec` — analyzer output
- `PlanTask` — planner output
- `Iteration` — a single generate→review→test cycle
- `ErrorContext` / `ErrorClassification` — error tracking and recovery

See [`docs/project-specification.html`](docs/project-specification.html) for the full design.

## Pull request checklist

- [ ] Tests added or updated
- [ ] `make lint && make test` passes locally
- [ ] Public APIs documented (docstrings + README if user-facing)
- [ ] No new secrets, hardcoded keys, or runtime files committed
- [ ] CHANGELOG updated for user-facing changes (if applicable)

## Reporting security issues

Please **do not** open a public issue for security vulnerabilities. See [SECURITY.md](SECURITY.md).

## Licensing

By contributing, you agree your contributions will be licensed under the [MIT License](LICENSE).
