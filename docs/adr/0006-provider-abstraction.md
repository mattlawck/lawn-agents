# ADR 0006 — Provider abstraction (`ChatModel` Protocol) with Gemini default

- **Status**: Accepted — 2026-05-23
- **Supersedes**: in part, [ADR 0001 — Direct Anthropic SDK](./0001-direct-anthropic-sdk.md)
- **Deciders**: Matt

## Context

ADR 0001 chose the direct Anthropic SDK on the assumption that the
project would be Anthropic-only. The user already has a funded balance
in Google AI Studio (Gemini) and wants to use that instead. Locking the
code to a single provider would force a rewrite at every model-vendor
shift, and prevents low-cost A/B comparisons.

## Decision

Introduce a narrow `ChatModel` Protocol in
`src/lawn_agents/llm.py` with two methods:

- `complete_text(*, system, user) -> str` — used by the router.
- `complete_structured(*, system, user, response_model) -> T` — used by
  the synthesizer; output validates against a Pydantic model.

Ship two concrete adapters:

- `GeminiChat` — uses `google-genai`. Default provider.
- `AnthropicChat` — uses `anthropic`. Kept for A/B and fallback.

Provider selection is one line in `config.yaml`:

```yaml
models:
  provider: "gemini"       # or "anthropic"
  router: "gemini-2.5-flash"
  synthesizer: "gemini-2.5-pro"
```

The corresponding API key must be in `.env`. A model validator on
`Settings` enforces this at startup with an explicit error.

## Rationale

- Cost: user has existing AI Studio credit; Gemini 2.5 Pro is roughly
  10× cheaper than Opus 4.7 on synthesis-heavy workloads.
- Portability: the Protocol is the contract; adapters are leaves. Swap
  providers in a config edit.
- Quality flexibility: keep Anthropic as a one-config-line fallback so
  we can A/B if a particular synthesis case underperforms on Gemini.
- The never-guess guardrail (ADR 0003) is unchanged — schema validation
  happens *after* the adapter returns, regardless of provider.

## Consequences

- `google-genai` is added as a runtime dependency.
- Two API keys are now optional in `.env`; the validator requires the
  one matching `models.provider`.
- Prompt caching (Anthropic `cache_control`, Gemini context caching) is
  provider-specific. Phase 1 ships without it; both adapters expose
  hooks where caching can land in a follow-up without changing the
  Protocol.
- Tests run against mocked clients only — no API key is required for
  CI to pass.
- ADR 0001 stays in the record as historical context; its decision is
  partially superseded but the reasoning about explicit-control
  over-agent-loop still applies.
