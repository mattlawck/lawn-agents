"""Self-extending RAG: allowlisted web research subagent.

When `knowledge.retrieve` returns weak results, the orchestrator may
invoke `search_and_ingest` to web-search the query, fetch results from
domains on `config.research.domain_allowlist`, chunk + embed them, and
store in LanceDB with `requires_review=True`. See ADR 0005.

Search uses DuckDuckGo via `ddgs` — no API key required. Allowlist
matching is enforced client-side: even if the search engine returns
something off-allowlist (which it can — `site:` is a hint, not a
filter), we drop it before fetching.

The synthesizer is allowed to use auto-ingested passages but they
arrive flagged with `requires_review=True`. `notify.ConsoleSink` and
the synthesizer prompt's `_format_sources` both label them
`[unreviewed]` so a human can review the new sources via
`lawn-agents review-additions` (separate task).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import httpx

from lawn_agents.agents import knowledge
from lawn_agents.agents.knowledge import Chunk
from lawn_agents.ingest import (
    CHARS_PER_TOKEN,
    chunk_text,
    fetch_url_text,
    make_url_sources,
)
from lawn_agents.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from lawn_agents.agents.knowledge import Embeddings, VectorStore
    from lawn_agents.config import AppConfig
    from lawn_agents.ingest import IngestSource
    from lawn_agents.models import Passage

log = get_logger(__name__)

DEFAULT_MAX_RESULTS = 8

SearchFn = "Callable[[str, list[str], int], list[str]]"


def search_and_ingest(
    query: str,
    config: AppConfig,
    *,
    embeddings: Embeddings | None = None,
    store: VectorStore | None = None,
    http_client: httpx.Client | None = None,
    search_fn: Callable[[str, list[str], int], list[str]] | None = None,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> list[Passage]:
    """Search the allowlist, ingest new content, return refreshed passages.

    Args:
        query: The query that came back weak from the local RAG.
        config: Application configuration.
        embeddings: Override embedder (tests inject `FakeEmbeddings`).
        store: Override vector store (tests inject a tmp-path store).
        http_client: Override HTTP client for URL fetches (tests inject
            `httpx.MockTransport`).
        search_fn: Override the search function (tests inject a function
            returning a canned URL list). Defaults to a DuckDuckGo
            search via `ddgs`.
        max_results: Cap on how many search results to fetch.

    Returns:
        The result of re-running `knowledge.retrieve` after ingestion.
        Empty list if research is disabled, the allowlist is empty,
        the search returned nothing relevant, or everything failed.
    """
    if not config.research.enabled or not config.research.domain_allowlist:
        log.info("research.disabled_or_empty_allowlist")
        return []

    embed = embeddings or knowledge.BgeSmall(model_name=config.knowledge.embedding_model)
    vec_store = store or knowledge.LanceDBStore(index_dir=config.knowledge.index_dir)
    own_client = http_client is None
    client = http_client or httpx.Client(
        timeout=config.http.timeout_seconds,
        headers={"User-Agent": _build_user_agent(config)},
    )
    do_search = search_fn or _ddg_search

    try:
        urls = _try_search(do_search, query, config.research.domain_allowlist, max_results)
        log.info("research.search_done", query=query, urls=len(urls))
        if not urls:
            return []

        sources = make_url_sources(urls)
        new_chunks = _fetch_and_chunk(sources, client, config)
        if not new_chunks:
            return []

        marked = [_mark_auto_ingested(c) for c in new_chunks]
        vectors = embed.embed_documents([c.content for c in marked])
        vec_store.add(marked, vectors)
        log.info("research.ingested", chunks=len(marked))
    finally:
        if own_client:
            client.close()

    # Re-run retrieval against the now-larger corpus. We call the
    # public knowledge.retrieve so the caller gets the freshly-scored
    # mix (auto-ingested + curated).
    return knowledge.retrieve(query, config)


# --- internals -------------------------------------------------------------


def _try_search(
    do_search: Callable[[str, list[str], int], list[str]],
    query: str,
    allowlist: list[str],
    max_results: int,
) -> list[str]:
    try:
        return do_search(query, allowlist, max_results)
    except Exception as exc:
        # Boundary swallow: research never raises past `search_and_ingest`.
        log.warning("research.search_failed", error=str(exc))
        return []


def _fetch_and_chunk(
    sources: list[IngestSource], client: httpx.Client, config: AppConfig
) -> list[Chunk]:
    fetched_at = datetime.now(UTC)
    chunk_size_chars = config.knowledge.chunk_size_tokens * CHARS_PER_TOKEN
    chunk_overlap_chars = config.knowledge.chunk_overlap_tokens * CHARS_PER_TOKEN
    chunks: list[Chunk] = []
    for source in sources:
        try:
            text = fetch_url_text(source, client)
        except Exception as exc:
            log.warning("research.fetch_failed", source=source.source_id, error=str(exc))
            continue
        if not text:
            continue
        chunks.extend(
            chunk_text(
                text,
                source_id=source.source_id,
                source_title=source.source_title,
                chunk_size_chars=chunk_size_chars,
                chunk_overlap_chars=chunk_overlap_chars,
                url=source.location,
                fetched_at=fetched_at,
            )
        )
    return chunks


def _mark_auto_ingested(chunk: Chunk) -> Chunk:
    """Return a copy of `chunk` with the auto-ingested flags flipped on."""
    return Chunk(
        id=chunk.id,
        content=chunk.content,
        source_id=chunk.source_id,
        source_title=chunk.source_title,
        url=chunk.url,
        page=chunk.page,
        fetched_at=chunk.fetched_at,
        auto_ingested=True,
        requires_review=True,
    )


def _ddg_search(query: str, allowlist: list[str], max_results: int) -> list[str]:
    """Search DuckDuckGo for `query` restricted to allowlisted domains."""
    from ddgs import DDGS

    if not allowlist:
        return []

    site_filter = " OR ".join(f"site:{d}" for d in allowlist)
    full_query = f"{query} ({site_filter})"

    urls: list[str] = []
    with DDGS() as ddgs:
        results = ddgs.text(full_query, max_results=max_results)
        for result in results:
            url = result.get("href")
            if not isinstance(url, str):
                continue
            if not _matches_allowlist(url, allowlist):
                continue
            urls.append(url)
    return urls


def _matches_allowlist(url: str, allowlist: list[str]) -> bool:
    """True if `url`'s host matches any entry in `allowlist`."""
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if not host:
        return False
    for entry in allowlist:
        entry_norm = entry.lower().lstrip("*.")
        if host == entry_norm or host.endswith("." + entry_norm):
            return True
    return False


def _build_user_agent(config: AppConfig) -> str:
    return (
        f"lawn-agents-research (+https://github.com/mattlawck/lawn-agents, "
        f"{config.http.contact_email})"
    )
