.PHONY: install install-dev lint format typecheck test test-cov docker-build docker-up docker-down clean all help

help:
	@echo "Available targets:"
	@echo "  install         Install the package in production mode"
	@echo "  install-dev     Install the package with development dependencies"
	@echo "  lint            Run ruff linter"
	@echo "  format          Format code with ruff"
	@echo "  typecheck       Run mypy type checker"
	@echo "  test            Run pytest tests"
	@echo "  test-cov        Run pytest with coverage report"
	@echo "  docker-build    Build Docker image"
	@echo "  docker-up       Start Docker container"
	@echo "  docker-down     Stop Docker container"
	@echo "  clean           Clean up generated files and caches"
	@echo "  all             Run lint, typecheck, and test"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

lint:
	ruff check .

format:
	ruff format .
	ruff check . --fix

typecheck:
	mypy src/fastcoder

test:
	pytest tests/

test-cov:
	pytest \
		--cov=fastcoder \
		--cov-report=html \
		--cov-report=term-missing \
		tests/
	@echo "Coverage report generated in htmlcov/index.html"

docker-build:
	docker build -f deploy/docker/Dockerfile -t fastcoder:latest .

docker-up:
	docker run -d \
		--name fastcoder \
		-p 3000:3000 \
		fastcoder:latest

docker-down:
	docker stop fastcoder || true
	docker rm fastcoder || true

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.egg-info" -delete
	rm -rf build/ dist/ htmlcov/ .coverage
	@echo "Cleaned up generated files and caches"

all: lint typecheck test
	@echo "All checks passed!"
