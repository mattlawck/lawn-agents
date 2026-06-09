"""Corpus ingester: PDFs from `data/corpus/` + URLs from `config.seed_urls`.

Discovers sources, fetches text, chunks them, embeds via the
`Embeddings` Protocol, and writes to the `VectorStore`. Safe to re-run:
`Chunk.id` is content-addressed (SHA-256 over source_id, page, chunk
index, content) so unchanged inputs upsert identically and the store
stays the same size.

PDFs are parsed page-by-page via pypdf. URLs are fetched with httpx and
the main content is extracted with trafilatura (filters navigation,
footers, ads — much higher signal than raw HTML-to-text).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlparse

import httpx

from lawn_agents.agents import knowledge
from lawn_agents.agents.knowledge import Chunk, chunk_id
from lawn_agents.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from lawn_agents.agents.knowledge import Embeddings, VectorStore
    from lawn_agents.config import AppConfig

log = get_logger(__name__)

# Rough English token-to-character ratio. Good enough for chunk sizing
# without dragging in tiktoken / model-specific tokenizers.
CHARS_PER_TOKEN = 4


@dataclass(slots=True, frozen=True)
class IngestSource:
    """A single input to the ingester: a PDF on disk, an HTML URL, or a PDF URL."""

    kind: Literal["pdf", "url", "pdf_url"]
    location: str  # absolute path for pdf; full URL for url and pdf_url
    source_id: str
    source_title: str


@dataclass(slots=True)
class IngestReport:
    """Outcome of one ingest cycle, surfaced by the CLI."""

    sources_seen: int = 0
    sources_ok: int = 0
    sources_failed: list[tuple[str, str]] = field(default_factory=list)
    chunks_added: int = 0
    chunks_total_before: int = 0
    chunks_total_after: int = 0


# --- discovery ------------------------------------------------------------


def discover_pdfs(corpus_dir: Path) -> list[IngestSource]:
    """Find PDFs in `corpus_dir` and wrap them as `IngestSource`s."""
    if not corpus_dir.exists():
        return []
    sources: list[IngestSource] = []
    for pdf in sorted(corpus_dir.glob("*.pdf")):
        sources.append(
            IngestSource(
                kind="pdf",
                location=str(pdf.resolve()),
                source_id=f"pdf:{pdf.name}",
                source_title=pdf.stem,
            )
        )
    return sources


def make_url_sources(seed_urls: Iterable[str]) -> list[IngestSource]:
    """Wrap each URL in an `IngestSource` with a readable title.

    URLs whose path ends in `.pdf` (case-insensitive, query/fragment
    ignored) are tagged `kind="pdf_url"` so the dispatcher routes them
    through pypdf rather than the HTML extractor. Trafilatura on a PDF
    byte stream returns empty and the source gets silently dropped —
    the bug fix here.
    """
    sources: list[IngestSource] = []
    for url in seed_urls:
        parsed = urlparse(url)
        domain = parsed.netloc
        path = parsed.path.rstrip("/") or "/"
        title = f"{domain}{path}"
        kind: Literal["url", "pdf_url"] = (
            "pdf_url" if parsed.path.lower().endswith(".pdf") else "url"
        )
        sources.append(
            IngestSource(
                kind=kind,
                location=url,
                source_id=f"url:{url}",
                source_title=title,
            )
        )
    return sources


# --- fetching -------------------------------------------------------------


def fetch_pdf_pages(source: IngestSource) -> list[tuple[int, str]]:
    """Return `[(page_number_1_indexed, text)]` from a PDF, skipping empty pages."""
    from pypdf import PdfReader

    reader = PdfReader(source.location)
    pages: list[tuple[int, str]] = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            # pypdf raises a variety of exception classes for malformed
            # pages; we never want one bad page to abort the whole PDF.
            log.warning(
                "ingest.pdf_page_extract_failed",
                source=source.source_id,
                page=idx,
                error=str(exc),
            )
            continue
        text = text.strip()
        if text:
            pages.append((idx, text))
    return pages


def fetch_pdf_url_pages(source: IngestSource, client: httpx.Client) -> list[tuple[int, str]]:
    """Download a PDF over HTTP and extract pages via pypdf.

    Mirrors `fetch_pdf_pages` but reads from an `httpx`-fetched byte
    stream wrapped in `io.BytesIO`, so no temp file is needed. PdfReader
    accepts any binary file-like; the BytesIO is closed implicitly when
    it goes out of scope.
    """
    from io import BytesIO

    from pypdf import PdfReader

    response = client.get(source.location, follow_redirects=True)
    response.raise_for_status()
    reader = PdfReader(BytesIO(response.content))
    pages: list[tuple[int, str]] = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            log.warning(
                "ingest.pdf_url_page_extract_failed",
                source=source.source_id,
                page=idx,
                error=str(exc),
            )
            continue
        text = text.strip()
        if text:
            pages.append((idx, text))
    return pages


def fetch_url_text(source: IngestSource, client: httpx.Client) -> str:
    """Fetch a URL and return its main content as plain text (trafilatura)."""
    import trafilatura

    response = client.get(source.location, follow_redirects=True)
    response.raise_for_status()
    extracted = trafilatura.extract(
        response.text,
        url=source.location,
        favor_recall=False,
        include_comments=False,
        include_tables=True,
    )
    return (extracted or "").strip()


# --- chunking -------------------------------------------------------------


def chunk_text(
    text: str,
    *,
    source_id: str,
    source_title: str,
    chunk_size_chars: int,
    chunk_overlap_chars: int,
    url: str | None = None,
    page: int | None = None,
    fetched_at: datetime | None = None,
) -> list[Chunk]:
    """Pack `text` into chunks of ~chunk_size_chars with paragraph-aware splits.

    Each chunk carries `chunk_overlap_chars` from the tail of the prior
    chunk so retrieval queries hitting a chunk boundary still see the
    surrounding context.
    """
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[Chunk] = []
    current = ""
    chunk_idx = 0
    for para in paragraphs:
        joined_len = len(current) + len(para) + 2
        if not current or joined_len <= chunk_size_chars:
            current = f"{current}\n\n{para}" if current else para
            continue
        chunks.append(
            _make_chunk(
                current.strip(),
                source_id,
                source_title,
                url,
                page,
                fetched_at,
                chunk_idx,
            )
        )
        chunk_idx += 1
        overlap = current[-chunk_overlap_chars:] if chunk_overlap_chars > 0 else ""
        current = f"{overlap}\n\n{para}" if overlap else para
    if current.strip():
        chunks.append(
            _make_chunk(
                current.strip(),
                source_id,
                source_title,
                url,
                page,
                fetched_at,
                chunk_idx,
            )
        )
    return chunks


def _make_chunk(
    content: str,
    source_id: str,
    source_title: str,
    url: str | None,
    page: int | None,
    fetched_at: datetime | None,
    chunk_idx: int,
) -> Chunk:
    return Chunk(
        id=chunk_id(source_id, page, chunk_idx, content),
        content=content,
        source_id=source_id,
        source_title=source_title,
        url=url,
        page=page,
        fetched_at=fetched_at,
        auto_ingested=False,
        requires_review=False,
    )


# --- orchestration --------------------------------------------------------


def ingest(
    config: AppConfig,
    *,
    embeddings: Embeddings | None = None,
    store: VectorStore | None = None,
    http_client: httpx.Client | None = None,
) -> IngestReport:
    """Run a full ingest cycle and return a report.

    Args:
        config: Application configuration.
        embeddings: Override for the default `BgeSmall` embedder. Tests
            inject `FakeEmbeddings`.
        store: Override for the default `LanceDBStore`. Tests inject a
            tmp-path-backed store.
        http_client: Override for URL fetches. Tests inject an
            `httpx.MockTransport`-backed client.

    Returns:
        An `IngestReport` summarizing sources processed and chunks added.
    """
    embeddings = embeddings or knowledge.BgeSmall(model_name=config.knowledge.embedding_model)
    store = store or knowledge.LanceDBStore(index_dir=config.knowledge.index_dir)
    own_client = http_client is None
    client = http_client or httpx.Client(
        timeout=config.http.timeout_seconds,
        headers={"User-Agent": _build_user_agent(config)},
    )

    report = IngestReport(chunks_total_before=store.count())

    sources = discover_pdfs(config.knowledge.corpus_dir) + make_url_sources(config.seed_urls)
    report.sources_seen = len(sources)

    chunk_size_chars = config.knowledge.chunk_size_tokens * CHARS_PER_TOKEN
    chunk_overlap_chars = config.knowledge.chunk_overlap_tokens * CHARS_PER_TOKEN
    fetched_at = datetime.now(UTC)

    try:
        for source in sources:
            _process_source(
                source,
                client=client,
                embeddings=embeddings,
                store=store,
                chunk_size_chars=chunk_size_chars,
                chunk_overlap_chars=chunk_overlap_chars,
                fetched_at=fetched_at,
                report=report,
            )
    finally:
        if own_client:
            client.close()

    report.chunks_total_after = store.count()
    return report


def _process_source(
    source: IngestSource,
    *,
    client: httpx.Client,
    embeddings: Embeddings,
    store: VectorStore,
    chunk_size_chars: int,
    chunk_overlap_chars: int,
    fetched_at: datetime,
    report: IngestReport,
) -> None:
    try:
        chunks = _source_to_chunks(
            source,
            client=client,
            chunk_size_chars=chunk_size_chars,
            chunk_overlap_chars=chunk_overlap_chars,
            fetched_at=fetched_at,
        )
    except Exception as exc:
        # Per-source boundary swallow: one bad PDF or 500-ing URL must
        # not abort the whole ingest run. The failure lands in the report.
        log.warning("ingest.source_failed", source=source.source_id, error=str(exc))
        report.sources_failed.append((source.source_id, f"{type(exc).__name__}: {exc}"))
        return

    if not chunks:
        log.info("ingest.source_empty", source=source.source_id)
        report.sources_ok += 1
        return

    vectors = embeddings.embed_documents([c.content for c in chunks])
    store.add(chunks, vectors)
    report.sources_ok += 1
    report.chunks_added += len(chunks)
    log.info("ingest.source_ok", source=source.source_id, chunks=len(chunks))


def _source_to_chunks(
    source: IngestSource,
    *,
    client: httpx.Client,
    chunk_size_chars: int,
    chunk_overlap_chars: int,
    fetched_at: datetime,
) -> list[Chunk]:
    if source.kind in ("pdf", "pdf_url"):
        pages = (
            fetch_pdf_pages(source) if source.kind == "pdf" else fetch_pdf_url_pages(source, client)
        )
        chunks: list[Chunk] = []
        url = source.location if source.kind == "pdf_url" else None
        for page_num, page_text in pages:
            chunks.extend(
                chunk_text(
                    page_text,
                    source_id=source.source_id,
                    source_title=source.source_title,
                    chunk_size_chars=chunk_size_chars,
                    chunk_overlap_chars=chunk_overlap_chars,
                    url=url,
                    page=page_num,
                    fetched_at=fetched_at,
                )
            )
        return chunks
    if source.kind == "url":
        text = fetch_url_text(source, client)
        return chunk_text(
            text,
            source_id=source.source_id,
            source_title=source.source_title,
            chunk_size_chars=chunk_size_chars,
            chunk_overlap_chars=chunk_overlap_chars,
            url=source.location,
            fetched_at=fetched_at,
        )
    # Defensive — `IngestSource.kind` is a Literal so all cases above
    # are covered; this guards against future additions to the Literal.
    msg = f"unknown source kind: {source.kind!r}"
    raise ValueError(msg)


def _build_user_agent(config: AppConfig) -> str:
    return (
        f"lawn-agents-ingester (+https://github.com/mattlawck/lawn-agents, "
        f"{config.http.contact_email})"
    )
