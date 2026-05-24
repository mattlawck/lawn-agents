"""Structured logging configured once at startup.

Pretty console output when stdout is a tty; JSON one-line-per-event when
it isn't (e.g., under launchd). Use `get_logger(__name__)` everywhere.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.typing import Processor


def configure(level: int = logging.INFO) -> None:
    """Configure structlog + stdlib logging.

    Call this once at program start. Idempotent.

    Args:
        level: stdlib logging level for the root logger.
    """
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    is_tty = sys.stdout.isatty()

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: Processor = (
        structlog.dev.ConsoleRenderer(colors=True)
        if is_tty
        else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(format="%(message)s", level=level, stream=sys.stdout)


def get_logger(name: str | None = None, **initial_context: Any) -> structlog.stdlib.BoundLogger:
    """Get a bound logger with optional initial context fields.

    Args:
        name: Logger name, typically `__name__`.
        **initial_context: Key/value pairs to bind into every event.

    Returns:
        A structlog BoundLogger.
    """
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    if initial_context:
        return logger.bind(**initial_context)
    return logger
