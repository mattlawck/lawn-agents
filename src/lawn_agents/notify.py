"""Output sinks — Phase 1 renders to the console; Phase 3 adds email/SMS.

The `Sink` Protocol keeps the orchestrator decoupled from any specific
output medium. Adding a new sink is a single class plus an entry in
`config.yaml > notify.sinks`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from lawn_agents.config import AppConfig
    from lawn_agents.models import Recommendation


class Sink(Protocol):
    """Anything that can render a `Recommendation` to some medium."""

    def emit(self, recommendation: Recommendation) -> None:
        """Render the recommendation. Must never raise."""
        ...


def build_sinks(config: AppConfig) -> list[Sink]:
    """Construct the list of sinks named in `config.notify.sinks`.

    Args:
        config: Application configuration.

    Returns:
        Ordered list of `Sink` instances. Phase 1 returns a single
        console sink.
    """
    raise NotImplementedError
