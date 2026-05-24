# ADR 0005 — Self-extending RAG via an allowlisted research subagent

- **Status**: Accepted — 2026-05-22
- **Deciders**: Matt

## Context

The corpus the user ingests up front will not cover every question.
Asking the synthesizer to "do its best" when the retrieval comes back
weak is exactly the failure mode ADR 0003 forbids. We need a way to
extend the corpus when there's a gap, without manual ingestion for
every blind spot.

## Decision

When the retrieval reranker's top score falls below
`knowledge.retrieval.weak_score_threshold` (default 0.55), the
orchestrator dispatches a **research subagent**:

1. Web-search the query restricted to a configured **domain allowlist**
   (`research.domain_allowlist` in `config.yaml`; defaults to `.edu`,
   `.gov`, and named turf-industry sites — Super-Sod, The Turfgrass
   Group, Sod Solutions, etc.). The general web is off by default.
2. Fetch each candidate page, extract main content, chunk, embed.
3. Store in LanceDB with `provenance` including:
   - `source_url`
   - `fetched_at` (UTC ISO-8601)
   - `auto_ingested: true`
   - `requires_review: true`
4. Re-run retrieval. If results are still weak, refuse per ADR 0003.

A CLI command `lawn-agents review-additions` lists pending auto-ingested
passages so the user can promote (`requires_review: false`) or delete
them. The synthesizer is allowed to use unreviewed passages but must
label their citations with an `"auto-researched on YYYY-MM-DD, unreviewed"`
suffix so the user can spot them in output.

## Rationale

- The corpus improves monotonically with use — the second run on a
  topic is smarter than the first.
- The allowlist contains the cost: the model can't pull in random
  forum advice or AI-generated SEO pages.
- The review gate keeps the user in control of what graduates to
  "trusted" status while not blocking the first useful answer.
- Citations remain real (a URL the user can verify), so ADR 0003's
  contract holds.

## Consequences

- Web fetches add latency to "first time we see this question" runs.
  We log the extra cost and surface it in `--scheduled` reports so the
  user understands when it ran.
- The allowlist must be curated. Adding a domain is intentionally a
  config-file edit (i.e., a code review on the public repo) — not
  a CLI command — so it sticks.
- Promoting/deleting auto-ingested passages is manual labor. Acceptable
  cost; the alternative is either an unbounded corpus or a brittle
  one.
- If the research subagent grows into a multi-step loop, revisit
  ADR 0001 (Anthropic SDK vs. Claude Agent SDK).
