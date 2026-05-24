# ADR 0003 — Three-layer "never guess" guardrail on chemical recommendations

- **Status**: Accepted — 2026-05-22
- **Deciders**: Matt

## Context

Lawn-agents recommends real chemical applications (herbicides,
fertilizers, insecticides, fungicides) for a real lawn next to the
South Carolina coastal marsh. Wrong product, rate, or timing can:

- kill the turf,
- contaminate sensitive waterways,
- violate federal label law (FIFRA — applying a pesticide inconsistent
  with its label is illegal).

LLMs hallucinate confident, plausible product names and rates when asked
about a topic outside their training distribution. The entire reason the
RAG layer exists is to ground every chemical recommendation in a cited,
authoritative source.

## Decision

Three concentric defenses, all required:

1. **Prompt** — The synthesizer's system prompt (`prompts/synthesizer.md`)
   contains a non-negotiable rule: any product name, application rate, or
   chemical timing must quote and cite a passage from the provided
   `<sources>` block. If no source supports the claim, the recommendation
   must say so explicitly and refuse rather than estimate.
2. **Schema** — Synthesis output is a Pydantic `Recommendation` model.
   `CalendarItem`s in chemical categories (`fertilizer`, `herbicide`,
   `insecticide`, `fungicide`) declare `citations: list[Citation]` with
   `min_length=1`. A Pydantic validator rejects the output if the
   constraint is violated. The orchestrator re-prompts once, then
   surfaces an explicit refusal.
3. **Tests** — `tests/test_guardrails.py` feeds the synthesizer empty
   `<sources>` and asserts that, for every chemical category, the output
   refuses rather than fabricates. Runs on every CI push.

General agronomic facts that are widely documented and configured (e.g.,
"Zoysia greens up around 65°F at 4-inch soil depth") may be stated
without per-call citation because they live in `config.yaml` as
configured thresholds. The guardrail applies to *products and rates*,
not to all of agronomy.

## Rationale

- Prompt-only guardrails fail open under distribution shift; schema
  validation catches what the model slips past the prompt.
- Schema-only enforcement is too late if the prompt didn't make
  refusal an option; the prompt gives the model a graceful exit.
- Tests prevent silent regression. The most expensive failure mode is
  a future contributor "loosening up" the prompt or schema for
  expedience.

## Consequences

- Some questions ("when should I fertilize?") will refuse on first run
  if the corpus is empty. Documented in the README as a feature, not a
  bug — `scripts/ingest_corpus.py` and `lawn-agents review-additions`
  are the path to making the system useful.
- The Opus synthesis call may consume an extra round-trip on the
  re-prompt path. Budget tracked in logs.
- We accept slightly less helpful surface behavior in exchange for
  hard correctness on chemical specifics.
