# 2026-05-23 — Switching the default LLM provider to Gemini

We chose Anthropic on Day 1 mostly by inertia — I was building this
inside Claude Code, so Claude felt like the natural pick. Then I
actually looked at the cost: a year of weekly checks + ad-hoc
questions on Opus 4.7 would run somewhere in the $50–$100 range. Not
huge, but real money for what's essentially a personal hobby tool.

I already have ~$25 of credit sitting in Google AI Studio. Gemini 2.5
Pro is roughly 10× cheaper than Opus on the prompt shapes we'd use,
and the free-tier rate limits on Flash + Pro are plenty for our
workload anyway. That credit could carry the entire project for two
or three years.

## The right way to switch

The wrong way: rip out Anthropic, paste in Gemini, move on. That's
how you end up locked to the *next* vendor instead.

The right way: introduce a Protocol — a thin contract for what the
orchestrator actually needs from the model layer — and put each
provider behind it as an adapter. Two methods, six lines:

```python
class ChatModel(Protocol):
    def complete_text(self, *, system: str, user: str) -> str: ...
    def complete_structured(self, *, system: str, user: str, response_model: type[T]) -> T: ...
```

That's the whole contract. The router calls `complete_text` and gets
back a one-word intent. The synthesizer calls `complete_structured`
with a Pydantic model, and either gets a validated instance back or a
`ValidationError` it can re-prompt on. Provider-specific noise —
Gemini's `response_schema`, Anthropic's tool-use convention — stays
inside the adapter.

## What this actually cost

About 90 minutes of work:

- `src/lawn_agents/llm.py` (210 lines) — Protocol + `GeminiChat` +
  `AnthropicChat` + `build_chat_model` factory + `parse_router_intent`
  helper.
- `config.py` — added `provider: Literal["gemini", "anthropic"]`,
  made both API keys optional, added a model validator that requires
  the right key for the selected provider.
- `config.yaml` — flipped default to `provider: gemini`,
  `gemini-2.5-flash` (router), `gemini-2.5-pro` (synthesizer). Left
  the Anthropic IDs in a comment so swapping is a copy-paste.
- `pyproject.toml` — added `google-genai`. Both SDKs ride along.
- `ADR 0006` documenting the protocol decision; amended `ADR 0001` so
  the historical record stays coherent.
- 8 new tests for the factory + intent parser. All 66 tests still
  green; coverage 85%.

The never-guess guardrail (ADR 0003) didn't change at all — it lives
at the schema/Pydantic layer, *after* the adapter returns. Whichever
provider generates the JSON, the validator decides whether to accept
it. This is exactly the kind of thing you get for free when the
guardrail is structural, not prompt-only.

## Open question: prompt caching

Both providers have prompt caching but in different shapes —
Anthropic uses `cache_control` blocks, Gemini has explicit context
caching with its own API. Neither is in the Protocol yet because
each would force vendor specifics out into the orchestrator. Holding
off; the synthesizer's system prompt is maybe 500 tokens and we run
it ~once a week, so caching is an optimization we don't need until we
do.

## What I'd flag for someone reading this in a year

- The Protocol has two methods, not three. Resist adding
  `complete_streaming` or `complete_with_tools` until there's a
  concrete in-tree caller. The whole point is that this is the
  *narrowest* useful interface.
- The factory is keyed on `Literal["gemini", "anthropic"]`. When
  adding a third provider, also extend the `assert_never(provider)`
  branch — mypy will tell you.
- Don't be precious about which provider is "default." It's one line
  of config. The interesting decisions are at the prompt and schema
  layer.
