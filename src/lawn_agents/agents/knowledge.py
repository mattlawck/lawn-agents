"""Local RAG: LanceDB-backed corpus with hybrid retrieval.

Embeds chunks with `BAAI/bge-small-en-v1.5` (see ADR 0002), stores in
LanceDB at `knowledge.index_dir`, and serves queries via hybrid
(vector + BM25) retrieval. Returns `Passage` objects with full
provenance for citation.

The ingester (`scripts/ingest_corpus.py`) consumes `data/corpus/*.pdf`
plus `seed_urls:` from `config.yaml`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lawn_agents.config import AppConfig
    from lawn_agents.models import Passage


def retrieve(query: str, config: AppConfig, *, top_k: int | None = None) -> list[Passage]:
    """Retrieve top-k passages relevant to `query`.

    Args:
        query: Natural-language question or topic phrase.
        config: Application configuration (used for retrieval knobs).
        top_k: Override `knowledge.retrieval.rerank_top_k` if non-None.

    Returns:
        Passages sorted by descending score. Empty list if the index is
        empty or no passages clear the relevance floor.
    """
    raise NotImplementedError


def is_weak(passages: list[Passage], config: AppConfig) -> bool:
    """Decide whether retrieval failed to find a strong-enough match.

    Used by the orchestrator to gate the research subagent.

    Args:
        passages: Output of `retrieve`.
        config: Application configuration (used for `weak_score_threshold`).

    Returns:
        True if the top score is below the configured threshold or the
        list is empty.
    """
    if not passages:
        return True
    return passages[0].score < config.knowledge.retrieval.weak_score_threshold
