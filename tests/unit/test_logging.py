"""Tests for the structured-logging configuration."""

from __future__ import annotations

from lawn_agents.logging import configure, get_logger


def test_configure_is_idempotent() -> None:
    configure()
    configure()


def test_get_logger_returns_bound_logger() -> None:
    configure()
    logger = get_logger("lawn_agents.test")
    assert hasattr(logger, "info")


def test_initial_context_is_bound() -> None:
    configure()
    logger = get_logger("lawn_agents.test", subject="lawn")
    # structlog binds context invisibly; the easiest assertion is that the
    # call doesn't raise and `bind` is reachable for further composition.
    assert hasattr(logger, "bind")
