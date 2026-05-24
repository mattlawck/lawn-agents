"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return REPO_ROOT


@pytest.fixture
def config_yaml_path(repo_root: Path) -> Path:
    """Path to the repo's committed example config.

    The real `config.yaml` is gitignored and per-user. Tests run against
    `config.example.yaml`, which is committed and uses placeholder values.
    """
    return repo_root / "config.example.yaml"


@pytest.fixture(autouse=True)
def _isolate_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Isolate each test from the developer's real `.env`.

    Tests run in a temporary cwd so pydantic-settings can't shadow our
    dummy env vars with the real `.env` file in the project root.
    """
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))
    monkeypatch.setenv("GEMINI_API_KEY", "test-dummy")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy")
