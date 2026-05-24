# ADR 0002 — LanceDB + local `bge-small` embeddings for the RAG layer

- **Status**: Accepted — 2026-05-22
- **Deciders**: Matt

## Context

The RAG layer needs to (1) ingest a small mixed corpus (~50–200 PDFs and
public web pages: Clemson HGIC, Super-Sod Lawn Academy, the Turfgrass
Group, paid PDFs the user subscribes to), (2) embed and store chunks with
provenance, and (3) serve sub-second queries from a CLI on a MacBook.
Hosting a server (Pinecone, Weaviate, Qdrant) is out of scope.

Embedding choices considered:

- **Voyage AI `voyage-3`** — Anthropic-recommended, top-of-leaderboard
  English RAG quality. Hosted, ~$0.06/M tokens. Adds a second API key.
- **OpenAI `text-embedding-3-small`** — hosted, $0.02/M tokens. Adds an
  OpenAI key.
- **`BAAI/bge-small-en-v1.5`** via `sentence-transformers` — local, free,
  runs on CPU or Mac MPS. Slightly behind Voyage on benchmarks but
  excellent for a small domain-specific corpus.

Vector store choices considered:

- **LanceDB** — embedded, file-backed (Lance columnar format), no server,
  hybrid retrieval supported, easy to inspect with `lancedb` or `polars`.
- **Chroma** — embedded, popular, larger dependency footprint.
- **FAISS + SQLite** — hand-rolled, fastest but the most glue code.

## Decision

- **Vector store**: LanceDB.
- **Embeddings (Phase 1)**: local `BAAI/bge-small-en-v1.5` via
  `sentence-transformers`.
- Both live behind narrow Protocols (`VectorStore`, `Embeddings`) so a
  later swap to Voyage or another store is a one-class change.

## Rationale

- LanceDB is the lowest-friction embedded option with first-class hybrid
  retrieval. Files in `data/index/` are inspectable and easy to back up.
- Local embeddings keep the repo single-vendor (Anthropic only) — one
  fewer API key in the README, one fewer outage to worry about, and no
  per-query embedding cost. For a corpus this small the quality
  difference vs. Voyage is real but small.
- The Protocols make the choice cheap to revisit if retrieval quality
  becomes a bottleneck.

## Consequences

- First-time setup downloads the bge-small model (~130 MB) into the
  HuggingFace cache. CI uses the same cache.
- Embedding throughput is CPU/MPS-limited, which is fine for a
  ~50-PDF corpus but won't scale to thousands of documents without
  switching to a hosted model.
- We accept slightly lower retrieval quality vs. Voyage for the
  simplicity and zero-cost properties. Promote to Voyage if specific
  recall failures surface in evaluation.
