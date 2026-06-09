# 2026-06-01 — First end-to-end run, and the right kind of refusal

The corpus was ingested last night (129 chunks: 123 from the Lawn Care
Nut 2025 Warm-Season E-Guide PDF, six from the seed URL list). Time to
ask the first acceptance question and see whether anything actually
worked.

```
.venv/bin/lawn-agents --ask "Is it too late to treat with GrubX?"
```

It took three tries to even get to an answer. Each failure was a
different layer of the stack, and each one taught me something.

## Failure 1 — Pydantic's schema isn't Gemini's schema

The first run got all the way to the synthesizer and then died:

```
400 INVALID_ARGUMENT. Unknown name "additional_properties" at
'generation_config.response_schema': Cannot find field.
```

I'd been passing `response_model=Recommendation` straight into
`google-genai`'s `GenerateContentConfig`. That works in their docs;
it does *not* work for any non-trivial Pydantic model. Pydantic v2
emits `additionalProperties: false` on every `BaseModel` schema, and
`google-genai` faithfully translates that to its protobuf
`additional_properties` field — which the Gemini API rejects, because
Gemini's OpenAPI subset doesn't accept it. Same story for unresolved
`$ref` entries inside `$defs`.

The fix in
[PR #27](https://github.com/mattlawck/lawn-agents/pull/27) was
philosophical as much as mechanical: stop asking Gemini to enforce
the schema at generation time, embed it in the system prompt
instead, and let Pydantic's `model_validate_json` enforce structure
post-hoc. The orchestrator's existing re-prompt-once-then-refuse loop
handles validation failures — that's where the real safety lives
anyway (ADR 0003). Schema-as-prompt-hint isn't worse than
schema-as-API-constraint when Pydantic is the second-tier validator.

While I was looking, I also fixed
[PR #26](https://github.com/mattlawck/lawn-agents/pull/26) — a race in
`LanceDBStore._ensure_table` where `list_tables()` returned empty for
an index `open_table` could still load, so the fall-through called
`create_table` against existing data and raised "already exists" on
every second run. Try `open_table` first, create on miss. Three
lines.

Both PRs landed with regression tests targeted at the *fix*, not the
failure mode. (I proposed adding a live Gemini smoke test and an
end-to-end ingest→retrieve roundtrip as preventive coverage; we
declined both, on the rule that I shouldn't pile on tests that only
re-prove what `structlog` already surfaces.)

## Failure 2 — "limit: 0"

Run two. The schema fix worked — the request reached Gemini, got past
validation, and then hit:

```
429 RESOURCE_EXHAUSTED.
Quota exceeded for metric: generate_content_free_tier_requests,
limit: 0, model: gemini-2.5-pro
```

That's not "you used up your daily free Pro requests." That's "Pro
has no free tier on this account." Gemini 2.5 Pro is paid-only on the
API; the $24.66 of AI Studio credit I'd been mentally relying on
sits idle until billing is enabled on the GCP project tied to the
API key.

Five minutes in the console fixed it. No code change.

## Failure 3 — Pro 503 spikes

Run three. Billing live. New error:

```
503 UNAVAILABLE. This model is currently experiencing high demand.
Spikes in demand are usually temporary. Please try again later.
```

Twice in a row. Pro has been famously load-spiky and the retry-info
field even tells you `retryDelay: 3s` — but the orchestrator doesn't
look at retry-info. Any non-`ValidationError` exception in the
synthesizer call falls straight through to a `Refused` panel. So a
transient infrastructure failure got the same UX treatment as a
content-policy refusal. Not great, and a structural follow-up.

For *this* session: flip the synthesizer to Flash via one config line
and re-run. Pro pricing isn't the point yet; an answer is.

```yaml
# config.yaml
synthesizer: "gemini-2.5-flash"   # was: gemini-2.5-pro
```

## What Flash gave me

```
╭─ Refused ────────────────────────────────────────────────────────╮
│ The provided knowledge sources do not contain information         │
│ regarding GrubX or the appropriate timing for grub control        │
│ treatments. For specific guidance on grub control products and    │
│ application windows, please consult resources like the Clemson    │
│ Home & Garden Information Center (HGIC), the Sod Solutions /     │
│ Super-Sod Zeon guide, or your local extension agent.             │
╰───────────────────────────────────────────────────────────────────╯
```

This is the right answer.

The pipeline did its job. The router classified ad-hoc. Weather,
soil, and drought all fetched. The local RAG retrieved passages.
Flash read them, recognized that the corpus doesn't speak to GrubX
specifically, set `refused=true`, and pointed me at where to look. The
Pydantic schema validated the refusal — you can tell because the
panel renders cleanly instead of as a 4KB traceback dump.

This is also the **first time the never-guess guardrail (ADR 0003) has
fired against a real LLM**, not a `FakeChatModel`. The integration
tests in PR #24 prove the wiring; today proves the discipline. Flash
could have invented a date. Phase 1 succeeds because it didn't.

## What the refusal didn't do, and that's interesting

GrubX is Scotts' brand. The active ingredient varies by SKU, but the
recent formulations are chlorantraniliprole — and extension
publications absolutely do speak to chlorantraniliprole timing for
white-grub control in warm-season turf. The Clemson HGIC factsheet on
white grub management is one URL away.

Flash didn't bridge the brand to the chemistry. It treated "GrubX" as
a literal term it had to find in the sources. That's a smaller
question than the refusal makes it look:

- Did the local RAG retrieve anything grub-related at all, and at
  what score?
- Did `is_weak()` return True or False — i.e., did the research
  subagent even fire on this query?
- Should the synthesizer prompt explicitly encourage brand →
  active-ingredient mapping as a step before refusing?

Threads for the next session. I want to understand whether the refusal
was correct caution or premature surrender before deciding what to
change.

## What the blog should say

Three things, when this becomes a post:

1. **Real APIs are hostile in three different layers.** Schema
   validation, quota, and capacity all reject you differently and the
   error JSON looks different each time. The orchestrator that
   treated all three as "Refused" is not wrong — it's underspecified.
   Worth a passage on the difference between *the model refused* and
   *the model couldn't be reached*.
2. **A structural guardrail beats a strict generation mode.** I lost
   `response_schema` enforcement and didn't lose any safety, because
   Pydantic is the real enforcer. If the generation-time schema had
   been the only line of defense, dropping it would have been scary.
3. **The first real refusal is a milestone, not a bug.** Flash
   refused on its first live question. That should be celebrated and
   then interrogated — *was* it the right refusal? — because a system
   that refuses too eagerly is just as broken as one that hallucinates
   too freely.

Next: figure out why "GrubX" stopped Flash cold when chlorantraniliprole
wouldn't have.
