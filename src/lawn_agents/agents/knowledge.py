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
from lawn_agents.models import Passage

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from lawn_agents.config import AppConfig

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
    """`Embeddings` impl backed by `sentence-transformers` BAAI/bge-small-en-v1.5.

    The 700MB+ torch dependency loads lazily on first embed call so
    imports stay cheap for CLI startup and for tests that inject a fake.
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
            from sentence_transformers import SentenceTransformer

            log.info("knowledge.bge.loading_model", name=self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_query(self, text: str) -> list[float]:
        """See `Embeddings.embed_query`."""
        model = self._ensure_model()
        # bge-small recommends a search-query instruction prefix.
        instructed = f"Represent this sentence for searching relevant passages: {text}"
        vec = model.encode(instructed, normalize_embeddings=True)
        return [float(x) for x in vec.tolist()]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """See `Embeddings.embed_documents`."""
        model = self._ensure_model()
        vecs = model.encode(list(texts), normalize_embeddings=True, batch_size=32)
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


def is_weak(passages: list[Passage], config: AppConfig) -> bool:
    """Decide whether retrieval failed to find a strong-enough match.

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


# --- factory hooks (overridable in tests) ---------------------------------


def _build_embeddings(config: AppConfig) -> Embeddings:
    return BgeSmall(model_name=config.knowledge.embedding_model)


def _build_store(config: AppConfig) -> VectorStore:
    return LanceDBStore(index_dir=config.knowledge.index_dir, vector_dim=DEFAULT_VECTOR_DIM)
