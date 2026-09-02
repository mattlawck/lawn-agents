"""Local RAG: bge-small embeddings + LanceDB vector store.

Phase-1 minimum: pure vector retrieval. Hybrid (BM25 + reranker) and
the self-extending research subagent (ADR 0005) land in follow-up PRs.

Design (per ADR 0002):
- `Embeddings` Protocol abstracts the embedder. `BgeSmall` is the
  default; tests inject a deterministic fake.
- `VectorStore` Protocol abstracts the index. `LanceDBStore` is the
  default; the corpus lives in `config.knowledge.index_dir`.
- Top-level `retrieve()` and `is_weak()` are the only orchestrator-
  facing surface; the rest is internal.
- The ingester (`scripts/ingest_corpus.py`, PR B) calls
  `LanceDBStore.add()` to populate the index. PR A is read-only +
  exposes `add()` so tests can seed corpora.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from lawn_agents.logging import get_logger
from lawn_agents.models import Passage, SourceTier

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from datetime import datetime

    from lawn_agents.config import AppConfig, SourceTiersConfig

log = get_logger(__name__)

# bge-small-en-v1.5 produces 384-dim vectors.
DEFAULT_VECTOR_DIM = 384
TABLE_NAME = "chunks"


@dataclass(slots=True, frozen=True)
class Chunk:
    """A single document chunk ready to be embedded and stored.

    The ingester (PR B) builds these; the store persists them. PR A
    only needs the schema in scope to type the `add()` signature.
    """

    id: str
    content: str
    source_id: str
    source_title: str
    url: str | None = None
    page: int | None = None
    fetched_at: datetime | None = None
    auto_ingested: bool = False
    requires_review: bool = False


def chunk_id(source_id: str, page: int | None, chunk_idx: int, content: str) -> str:
    """Deterministic SHA-256 chunk ID so re-ingestion dedupes naturally."""
    h = hashlib.sha256()
    h.update(source_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(str(page if page is not None else -1).encode("utf-8"))
    h.update(b"\x00")
    h.update(str(chunk_idx).encode("utf-8"))
    h.update(b"\x00")
    h.update(content.encode("utf-8"))
    return h.hexdigest()


# --- Embeddings ------------------------------------------------------------


class Embeddings(Protocol):
    """A text embedder."""

    @property
    def dimension(self) -> int:
        """Output vector dimensionality."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query (instruction-tuned variant if applicable)."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed many corpus passages in one call."""


class BgeSmall:
    """`Embeddings` impl backed by `fastembed` BAAI/bge-small-en-v1.5.

    Uses the ONNX runtime instead of torch. Same 384-dim BGE-small
    embedding model; smaller install (~67 MB ONNX vs ~150 MB+ torch),
    faster cold start, no `torch.jit.script` exposure. Loads lazily on
    first embed call so imports stay cheap for CLI startup and for
    tests that inject a fake.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        """Store the model name; defer the heavy load until first use."""
        self._model_name = model_name
        self._model: Any = None

    @property
    def dimension(self) -> int:
        """See `Embeddings.dimension`."""
        return DEFAULT_VECTOR_DIM

    def _ensure_model(self) -> Any:
        if self._model is None:
            from fastembed import TextEmbedding

            log.info("knowledge.bge.loading_model", name=self._model_name)
            # fastembed's BGE-small ONNX export bakes in L2 normalization
            # and applies the search-query instruction internally for
            # query_embed/passage_embed calls, matching sentence-
            # transformers' bge defaults.
            self._model = TextEmbedding(self._model_name)
        return self._model

    def embed_query(self, text: str) -> list[float]:
        """See `Embeddings.embed_query`."""
        model = self._ensure_model()
        # Apply BGE's recommended search-query instruction prefix
        # explicitly so vectors match what sentence-transformers
        # produced for the existing LanceDB index (no re-ingest
        # required). fastembed's `query_embed` is a no-op on bge-small
        # in this version — same output as `embed` — so we control the
        # instruction here.
        instructed = f"Represent this sentence for searching relevant passages: {text}"
        vecs = list(model.embed([instructed]))
        return [float(x) for x in vecs[0].tolist()]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """See `Embeddings.embed_documents`."""
        model = self._ensure_model()
        # No instruction prefix on the corpus side — matches BGE
        # convention and the prior sentence-transformers behavior.
        vecs = list(model.embed(list(texts)))
        return [[float(x) for x in v.tolist()] for v in vecs]


# --- VectorStore ----------------------------------------------------------


class VectorStore(Protocol):
    """A vector store over `Chunk`s."""

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> None:
        """Insert (or upsert by `id`) the given chunks with their vectors."""

    def search(self, query_vector: list[float], *, k: int) -> list[Passage]:
        """Return the top-k nearest chunks as `Passage`s, sorted by score desc."""

    def count(self) -> int:
        """Total chunks currently stored."""


class LanceDBStore:
    """`VectorStore` impl backed by LanceDB.

    File-backed; no server. The DB lives at `index_dir`; the table is
    `chunks`. Idempotent: re-adding a chunk with the same `id` upserts.
    """

    def __init__(self, index_dir: Path, vector_dim: int = DEFAULT_VECTOR_DIM) -> None:
        """Defer LanceDB connect + table open until the first method call."""
        self._index_dir = Path(index_dir)
        self._vector_dim = vector_dim
        self._table: Any = None

    def _ensure_table(self) -> Any:
        if self._table is not None:
            return self._table

        import lancedb
        from lancedb.pydantic import LanceModel, Vector

        self._index_dir.mkdir(parents=True, exist_ok=True)
        db = lancedb.connect(str(self._index_dir))

        vector_dim = self._vector_dim

        class ChunkRow(LanceModel):  # type: ignore[misc, no-any-unimported]
            id: str
            content: str
            vector: Vector(vector_dim)  # type: ignore[valid-type]
            source_id: str
            source_title: str
            url: str = ""
            page: int = 0
            fetched_at: str = ""
            auto_ingested: bool = False
            requires_review: bool = False

        # Open-first; `list_tables()` was unreliable across LanceDB versions.
        try:
            self._table = db.open_table(TABLE_NAME)
        except (FileNotFoundError, ValueError):
            self._table = db.create_table(TABLE_NAME, schema=ChunkRow)
        return self._table

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> None:
        """See `VectorStore.add`. Upserts by `Chunk.id` so re-ingest dedupes."""
        if len(chunks) != len(vectors):
            msg = f"chunks ({len(chunks)}) and vectors ({len(vectors)}) must be the same length"
            raise ValueError(msg)
        if not chunks:
            return

        rows = [
            {
                "id": c.id,
                "content": c.content,
                "vector": list(v),
                "source_id": c.source_id,
                "source_title": c.source_title,
                "url": c.url or "",
                "page": c.page if c.page is not None else 0,
                "fetched_at": c.fetched_at.isoformat() if c.fetched_at else "",
                "auto_ingested": c.auto_ingested,
                "requires_review": c.requires_review,
            }
            for c, v in zip(chunks, vectors, strict=True)
        ]
        table = self._ensure_table()
        table.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(
            rows
        )

    def search(self, query_vector: list[float], *, k: int) -> list[Passage]:
        """See `VectorStore.search`. Maps LanceDB distance to a [0, 1] score."""
        from datetime import datetime

        table = self._ensure_table()
        results = table.search(query_vector).limit(k).to_list()
        passages: list[Passage] = []
        for row in results:
            # LanceDB returns `_distance` (lower = closer). For unit
            # vectors with cosine distance, range is [0, 2]; clamp to a
            # similarity score in [0, 1].
            distance = float(row.get("_distance", 1.0))
            score = max(0.0, min(1.0, 1.0 - distance / 2.0))
            page_raw = row.get("page", 0)
            page_val: int | None = int(page_raw) if page_raw else None
            fetched_raw = row.get("fetched_at", "")
            fetched_val: datetime | None = (
                datetime.fromisoformat(fetched_raw) if fetched_raw else None
            )
            url_raw = row.get("url") or None
            passages.append(
                Passage(
                    content=str(row["content"]),
                    score=score,
                    source_id=str(row["source_id"]),
                    source_title=str(row["source_title"]),
                    url=str(url_raw) if url_raw else None,
                    page=page_val,
                    fetched_at=fetched_val,
                    auto_ingested=bool(row.get("auto_ingested", False)),
                    requires_review=bool(row.get("requires_review", False)),
                )
            )
        return passages

    def count(self) -> int:
        """See `VectorStore.count`."""
        table = self._ensure_table()
        return int(table.count_rows())


# --- public API -----------------------------------------------------------


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
    embeddings = _build_embeddings(config)
    store = _build_store(config)

    query_vec = embeddings.embed_query(query)
    k = top_k if top_k is not None else config.knowledge.retrieval.rerank_top_k
    return store.search(query_vec, k=k)


def is_weak(
    passages: list[Passage],
    config: AppConfig,
    *,
    query: str | None = None,
    extra_terms: Sequence[str] = (),
    relevance_gate: Callable[[str, Passage], bool] | None = None,
) -> bool:
    """Tiered relevance assessment of retrieval output.

    Three confidence bands keyed off `retrieval.weak_score_threshold`
    (default 0.55) and `retrieval.strong_score_threshold` (default 0.70):

    - ``score >= strong``      → trust the top passage; skip checks.
    - ``weak <= score < strong`` → run a lexical-overlap check between
      the query (+ `extra_terms` — the weed aliases and brand active
      ingredients the bridges would add) and the top passage. On miss,
      optionally escalate to a `relevance_gate` LLM check. Either
      failure marks the result weak.
    - ``score < weak``         → mark weak (research subagent fires).

    The tiered structure pays compute only for the ambiguous middle.
    Most queries fall in the strong or weak bands and skip the checks
    entirely. The Japanese-clover→sedge probe in PR #32 was at
    score 0.696 — squarely medium — and would have been caught here
    via lexical miss → gate.

    Args:
        passages: Output of `retrieve`.
        config: Application configuration (used for thresholds).
        query: User question text; required for lexical / gate checks.
        extra_terms: Additional terms to count as query vocabulary for
            the lexical overlap check — weed aliases (ADR 0008) and brand
            active ingredients (ADR 0007). The corpus discusses chemistry
            and scientific names; users ask by brand and common name, so
            without these the check compares two disjoint vocabularies.
        relevance_gate: Optional callable
            ``(query, top_passage) -> is_relevant``. Invoked only when
            lexical overlap fails in the medium band. Pass None to
            skip gate escalation (treats lexical miss as weak).

    Returns:
        True if the result is weak by any of the three signals.
    """
    if not passages:
        return True

    top = passages[0]
    weak_threshold = config.knowledge.retrieval.weak_score_threshold
    strong_threshold = config.knowledge.retrieval.strong_score_threshold

    if top.score < weak_threshold:
        return True
    if top.score >= strong_threshold:
        return False

    # Medium band.
    if query is None:
        # Fall back to single-threshold behavior when caller didn't
        # provide a query — keeps legacy callers (tests, etc.) working
        # without forcing them to pass through query/aliases/gate.
        return False

    if _has_lexical_overlap(query, top, extra_terms):
        return False

    log.info(
        "knowledge.is_weak.medium_lexical_miss",
        score=top.score,
        source=top.source_id,
    )
    if relevance_gate is None:
        return True

    try:
        relevant = relevance_gate(query, top)
    except Exception as exc:
        # Gate failures shouldn't block the pipeline; conservatively
        # treat as weak so the research subagent can run.
        log.warning("knowledge.is_weak.gate_failed", error=str(exc))
        return True
    log.info("knowledge.is_weak.gate_verdict", relevant=relevant)
    return not relevant


def classify_source(passage: Passage, tiers: SourceTiersConfig) -> SourceTier:
    """Classify a passage's provenance into a `SourceTier` (ADR 0009).

    Matches configured substrings against the passage's URL,
    `source_id`, and `source_title` combined, so both web sources and
    local corpus PDFs (which have no URL) can be tiered.

    Precedence is `label` → `extension` → `vendor`: trust travels with
    the document, not the host. A manufacturer label mirrored on a
    retailer's domain is still a label, and matching the label patterns
    first is what keeps it one.
    """
    haystack = " ".join(
        part for part in (passage.url, passage.source_id, passage.source_title) if part
    ).lower()
    for tier, patterns in (
        (SourceTier.LABEL, tiers.label),
        (SourceTier.EXTENSION, tiers.extension),
        (SourceTier.VENDOR, tiers.vendor),
    ):
        if any(pattern.lower() in haystack for pattern in patterns):
            return tier
    return SourceTier.UNKNOWN


def format_sources(passages: list[Passage], tiers: SourceTiersConfig) -> str:
    """Render retrieved passages as the `<sources>` block for a prompt.

    Shared by the orchestrator and the planner — they had byte-identical
    private copies before tiering gave them a reason to diverge, and one
    of them would have been updated without the other.

    Each entry carries a `tier=` marker so the synthesizer can weigh
    extension guidance against vendor marketing instead of treating
    every cited passage as equally authoritative.
    """
    if not passages:
        return "(no relevant passages retrieved)"
    parts: list[str] = []
    for i, p in enumerate(passages, start=1):
        review_flag = " [unreviewed]" if p.requires_review else ""
        page_str = f", page {p.page}" if p.page is not None else ""
        url_str = f", url {p.url}" if p.url else ""
        tier = classify_source(p, tiers)
        parts.append(
            f"[{i}] source_id={p.source_id!r} title={p.source_title!r}"
            f"{page_str}{url_str} tier={tier.value}{review_flag} score={p.score:.3f}\n"
            f"    {p.content}"
        )
    return "\n\n".join(parts)


_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "do",
        "for",
        "from",
        "have",
        "i",
        "in",
        "is",
        "it",
        "its",
        "my",
        "of",
        "on",
        "or",
        "should",
        "so",
        "the",
        "this",
        "to",
        "use",
        "want",
        "was",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "would",
        "you",
        "your",
        "how",
        "can",
        "could",
        "does",
        "did",
        "had",
        "has",
        "may",
        "might",
        "will",
        "yard",
        "lawn",
        "if",
    }
)


def _tokenize(text: str) -> set[str]:
    """Tokenize free text into lowercase content words."""
    import re

    raw = re.findall(r"[A-Za-z][A-Za-z0-9'-]+", text.lower())
    return {w for w in raw if len(w) > 2 and w not in _STOPWORDS}


def _has_lexical_overlap(query: str, passage: Passage, extra_terms: Sequence[str] = ()) -> bool:
    """At least one query (or alias) content word must appear in the passage."""
    query_terms = _tokenize(query)
    for term in extra_terms:
        query_terms |= _tokenize(term)
    if not query_terms:
        return False
    passage_terms = _tokenize(passage.content)
    return bool(query_terms & passage_terms)


# --- factory hooks (overridable in tests) ---------------------------------


def _build_embeddings(config: AppConfig) -> Embeddings:
    return BgeSmall(model_name=config.knowledge.embedding_model)


def _build_store(config: AppConfig) -> VectorStore:
    return LanceDBStore(index_dir=config.knowledge.index_dir, vector_dim=DEFAULT_VECTOR_DIM)
