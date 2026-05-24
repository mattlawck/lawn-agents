# ADR 0001 — Use the direct provider SDK, not the Claude Agent SDK

- **Status**: Accepted — 2026-05-22 — partially superseded by
  [ADR 0006](./0006-provider-abstraction.md) (2026-05-23): the LLM call
  now goes through the `ChatModel` Protocol, not the Anthropic SDK
  directly. The reasoning below about *agent loop vs. direct calls*
  still applies; the reasoning about *which* SDK does not.
- **Deciders**: Matt

## Context

Anthropic ships two complementary tools for building with Claude:

- The **Anthropic Python SDK** (`anthropic`) — a low-level client over the
  Messages API. The caller controls every turn explicitly.
- The **Claude Agent SDK** — adds an agent loop, tool registration, and
  conventions for multi-step autonomous behavior.

Lawn-agents has a small number of well-defined LLM call sites and a mostly
deterministic pipeline (HTTP fetches and a vector search dominate the
work). The hard part of this system is the data-source contracts, the
"never guess" guardrail, and the citation pipeline — not the agent loop.

## Decision

Use the **direct Anthropic SDK** for the two LLM calls in Phase 1:

1. Sonnet 4.6 intent router (one call, classification).
2. Opus 4.7 synthesizer (one or two calls — generation, then optional
   re-prompt if the citation guardrail rejects the first draft).

Prompt caching is enabled on the synthesizer's static system prompt and
the source-block prefix.

## Rationale

- Two call sites do not benefit from an agent loop.
- Direct control over the messages array makes prompt caching and
  schema-validated outputs straightforward.
- The "never guess" guardrail wants tight integration between the model
  call, the Pydantic validator, and a single re-prompt — easier to
  express in plain code than as agent-loop hooks.
- One fewer abstraction layer to reason about when debugging.

## Consequences

- The "research subagent" (ADR 0005), if it grows into a multi-step web
  research loop, may be a better fit for the Agent SDK. Revisit this
  decision when implementing it.
- We give up some convenience around tool registration and built-in
  retry/budget machinery. We compensate with `tenacity` for retries and
  explicit budget tracking.
