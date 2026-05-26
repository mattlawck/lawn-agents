# lawn-agents local development shortcuts.
#
# `make check` runs the same gates CI runs — use it before committing
# so you catch failures locally instead of after pushing.
#
# Resolves tools via the project venv so `uv` isn't a hard requirement
# locally. Anyone who prefers uv can override on the command line:
#
#     make check PY="uv run"

VENV ?= .venv
PY ?= $(VENV)/bin

.PHONY: help install dev check lint format format-check type test cov clean

help: ## Show this help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install runtime + dev dependencies (requires uv).
	uv sync --all-extras --dev

dev: install ## Install deps + activate pre-commit hooks.
	$(PY)/pre-commit install

check: lint format-check type test ## Full CI parity: lint, format, types, tests.

lint: ## Ruff lint.
	$(PY)/ruff check .

format: ## Auto-fix formatting.
	$(PY)/ruff format .

format-check: ## Fail if formatting is wrong (CI uses this).
	$(PY)/ruff format --check .

type: ## Mypy strict.
	$(PY)/mypy

test: ## Pytest with coverage.
	$(PY)/pytest

cov: ## Pytest with HTML coverage report.
	$(PY)/pytest --cov-report=html

clean: ## Remove caches and coverage artifacts.
	rm -rf .mypy_cache .ruff_cache .pytest_cache .coverage coverage.xml htmlcov
