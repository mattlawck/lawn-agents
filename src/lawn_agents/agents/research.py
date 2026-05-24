"""Self-extending RAG: allowlisted web research subagent.

When `knowledge.retrieve` returns weak results, the orchestrator may
invoke `search_and_ingest` to web-search the query, fetch results from
domains on the allowlist, chunk + embed them, and store in LanceDB
with `requires_review=True`. See ADR 0005.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lawn_agents.config import AppConfig
    from lawn_agents.models import Passage


def search_and_ingest(query: str, config: AppConfig) -> list[Passage]:
    """Search the allowlist, fetch new content, chunk + embed + store.

    Args:
        query: The user's question or topic.
        config: Application configuration (uses `research.domain_allowlist`).

    Returns:
        The newly ingested passages. May be empty if no allowlisted
        results were found.
    """
    raise NotImplementedError
