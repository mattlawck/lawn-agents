# ADR 0011 — Hybrid retrieval and self-identifying chunks

- **Status**: Accepted — 2026-09-15
- **Deciders**: Matt

## Context

ADR 0010 closed the "cited but unsupported" gap and named the half it
could not fix: retrieval. The passages that would have answered a
question existed in the corpus and never reached the synthesizer, so
the guardrail refused — correctly, and uselessly.

On 2026-09-14 that became concrete. Matt had nutsedge to spot spray and
both products on hand. The system refused. Chasing it produced four
distinct causes, each of which looked like the answer until it wasn't:

**1. The labels weren't in the corpus.** Ingested both. Still refused —
correctly, since a citation needs a source.

**2. The brand bridge never reached retrieval.** ADR 0007 taught the
*synthesizer* that "Sedge Ender" means sulfentrazone, and since
2026-09-02 taught the *relevance check* too. Retrieval was never
wired — so a question naming a product never searched for its
chemistry, the same asymmetry ADR 0008 had already fixed for weeds.
Wired. Still refused.

**3. Dense retrieval cannot find a product by name.** Probed against
the live corpus, the query

> "Sedge Ender sulfentrazone application rate per 1000 sq ft Zoysia matrella"

returned **zero Sedge Ender chunks** in its top eight. It returned the
rate tables of Fusilade II, Recognition and Celsius. BGE-small treats
"application rate per 1000 sq ft" as the dominant signal and every
label matches it, while a brand name is a rare token carrying almost no
semantic weight. This is not a phrasing problem; it is what dense
retrieval is bad at.

**4. The rate tables do not contain the product's name.** This was the
actual cause, and it explains why none of the above worked. EPA labels
refer to their subject as "this product" throughout. Of the twelve
chunks in the Bonide Sedge Ender label, **exactly one** contained the
words "Sedge Ender" — and it was neither of the two holding the
application-rate tables. Those chunks open "SPECIFIC USE INSTRUCTIONS
IN TURFGRASS" and "Grass Type / Warm Season Grasses".

No retrieval algorithm can find text that isn't there.

## Decision

Two changes, addressing causes 3 and 4.

### Hybrid retrieval

`knowledge.retrieve` now fuses vector search with full-text (BM25)
search, using LanceDB's native FTS — no new dependency. The two halves
fail in opposite directions:

- **Vector** finds passages that *mean* the same thing. It is how "my
  grass is yellowing in patches" reaches a disease factsheet. It cannot
  find a product by name.
- **Full-text** finds exact terms. `"Sedge Ender"` puts the Sedge Ender
  label at rank 0, score 10.85. It cannot bridge vocabulary at all — it
  will never connect "Japanese clover" to "annual lespedeza".

Scores are not comparable (cosine similarity vs. unbounded BM25), so
they fuse by Reciprocal Rank Fusion — rank-based and therefore
scale-invariant, the same reasoning as ADR 0008's multi-query fusion.
BM25-only hits carry a neutral score in the medium confidence band
rather than a fabricated similarity, because the weak/strong thresholds
are calibrated against cosine distance and nothing else.

`top_k_vector` and `top_k_bm25` return to `RetrievalConfig`. They were
removed in the September audit as config advertising a capability the
code lacked; they are back because it now exists.

### Self-identifying chunks

Every chunk is stored with a `[Source: <title>]` header. A chunk is
otherwise anonymous, and anonymity is fatal for exactly the documents
that matter most here: a label's rate table is the chunk a user needs
and the chunk least likely to name its own product.

Chunk ids stay content-addressed over the *original* body, so
re-ingesting unchanged sources still dedupes.

### Reserved slots per bridge query

RRF rewards consensus across queries. That is right for general
relevance and wrong for a question naming two products: a label chunk
scores on the single query naming its product and loses to chunks
scoring moderately on four. So each bridge query is guaranteed to
contribute its top two hits before fusion fills the rest, and the
result budget is computed *after* reserving rather than before.

This was tried once under vector-only retrieval and reverted, because
it reserved the wrong documents. It only became correct once full-text
search made each brand query's top hits reliably the right product —
worth recording, because the same change was both wrong and right
depending on what sat underneath it.

## Consequences

**Good.** The motivating question now answers. The system recommends
SedgeHammer with rates cited to EPA label pages 3 and 4, factors in D2
drought, and states plainly that it cannot give Sedge Ender rates —
a complete answer for the product it can ground, and transparency about
the one it cannot, instead of a blanket refusal.

Prompt caching benefits too: the `[Source: ...]` header makes chunk
provenance visible to the synthesizer inside the passage text rather
than only in the `<sources>` scaffolding.

**Bad / accepted.**

- **`source_title` is a URL for most labels**, so headers read
  `[Source: docs.diypestcontrol.com/SPEC/LABELS/sedgeender.pdf]`.
  Tokenised, "sedgeender" is one token and does not match "Sedge
  Ender". It worked here because the brand query also matches the body
  text, but letting `seed_urls` carry a friendly title would be
  sharper. Not done yet.
- **Changing chunk text invalidates every id**, so adopting this
  required a full re-index. Cheap at a few hundred chunks; it would not
  be at a few hundred thousand.
- **The FTS index is rebuilt wholesale on each ingest.** Fine for this
  corpus size. A stale text index silently returns nothing for new
  sources, which presents as "the corpus doesn't cover that" — the one
  conclusion this system must never reach by accident — so correctness
  beats incrementality here.
- BM25 brings its own failure mode: a rare token in an irrelevant
  document now ranks. RRF dilutes this, and the ADR 0010 grounding
  check remains the backstop.

## Alternatives considered

**A dedicated BM25 library (`rank_bm25`).** Rejected: LanceDB already
indexes the corpus, native FTS keeps one store rather than two that can
drift, and it avoids a dependency for something the database does.

**Better query phrasing.** Tried and reverted. A rate-seeking variant
("application rate per 1000 sq ft <species>") embeds closer to *every*
label's rate table than to any particular one, so it retrieved other
products' numbers — actively worse than retrieving nothing.

**Re-chunking labels by section.** Would help, but does not address the
root problem: a section titled "Recommended Use Rates" still never says
which product's rates they are. Headers fix that at any chunk size.

**An LLM reranker over a wider candidate set.** Rejected for the usual
reason in this project — it puts a model on the safety-critical path to
compensate for a mechanical problem that has a mechanical fix.
