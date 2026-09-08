# ADR 0010 — Claim-support validation (cited vs. supported)

- **Status**: Accepted — 2026-09-08
- **Deciders**: Matt

## Context

ADR 0003 describes the never-guess guardrail as three concentric
defenses: prompt, schema, tests. The schema layer was supposed to be
the real enforcer — the thing a model cannot talk its way past.

In practice the schema layer was one validator:

```python
@model_validator(mode="after")
def _requires_citation_for_chemicals(self) -> Self:
    if isinstance(self.category, ChemicalCategory) and not self.citations:
        raise ValueError(...)
```

It counts citations. It has never checked what they point at.

The synthesizer prompt *does* ask for more. Rule 1: any chemical
specific "must quote a passage from the `<sources>` block." Rule 5:
"cite verbatim or near-verbatim." `Citation.snippet` is documented as
"The quoted passage backing the claim." All of that was prompt-level —
the one layer a model can silently ignore.

On 2026-09-08 two probes against the live 311-chunk corpus produced
answers that were confidently wrong and passed every existing check.

**Failure 1 — fabricated active ingredient.** Asked about doveweed, the
synthesizer recommended Celsius and described its actives as
"thiencarbazone-methyl, **foramsulfuron**, dicamba."

`foramsulfuron` appears in **zero of 311 chunks**. It is in no source
in the corpus. The item's citation was the real Celsius label, which
states plainly:

> "Two of the three active ingredients in Celsius WG Herbicide
> (thiencarbazone-methyl and iodosulfuron-methyl-sodium) inhibit
> acetolactate synthase (ALS)."

A real citation, correctly attributed, wrapped around an invented fact.

**Failure 2 — unsupported product↔weed pairing.** Asked about
goosegrass, the synthesizer recommended Celsius chemistry, citing
Clemson's general warm-season weed factsheet. The Celsius label in the
corpus never mentions goosegrass — 0 of its 30 chunks. No source pairs
that product with that weed.

Neither failure escalated. `is_weak` scored retrieval as strong, so no
lexical check, no LLM gate, no research subagent. The system was
confident, cited, and wrong.

The common root: **presence of a citation is not support for a claim**,
and nothing in the codebase distinguished the two.

## Decision

Add a deterministic post-synthesis check — `lawn_agents.grounding` —
that validates a schema-valid `Recommendation` against the passages
that were actually placed in `<sources>`. Failure re-prompts once with
specifics, then refuses, reusing the ADR 0003 retry path.

Three checks, all on chemical-category items only:

| Check | Rule | Catches |
|---|---|---|
| **Provenance** | `citation.source_id` must belong to a passage in `<sources>` | Invented sources; citing a source after retrieval failed |
| **Snippet** | `citation.snippet` tokens must overlap the cited passage above a threshold | Invented quotes; enforces prompt rule 5 for the first time |
| **Chemical term** | Any brand or active ingredient from the bridge vocabulary that the action names must appear in a cited passage | Fabricated chemistry (`foramsulfuron`) |

It is deliberately **deterministic** — no LLM, no tokens, no latency.
That means it runs on every synthesis, in the planner as well as the
orchestrator, and it can be unit-tested in CI without API calls. Given
that the guardrail exists to catch model error, implementing it *with a
model* would have been circular.

Snippet matching is **token-set containment, not substring equality**,
at a configurable threshold (default 0.6). Rule 5 explicitly permits "a
quote or a tight paraphrase," and PDF extraction introduces whitespace
and hyphenation noise that breaks literal matching for legitimate
quotes. The threshold is the knob for how much paraphrase to tolerate.

## Consequences

**Good:**

- The gap between what ADR 0003 promised and what it enforced is
  closed for the fabrication case. A named chemical that appears in no
  cited passage is now a hard failure.
- Retrieval failure now degrades correctly. Previously, if the index
  threw, `<sources>` rendered empty and a model that produced a cited
  recommendation anyway sailed through. It now refuses.
- Prompt rule 5 is enforced rather than merely requested.
- It runs in the planner too, where it matters more: a year-long plan
  names more products than any single answer.

**Bad / accepted:**

- **It does not catch Failure 2.** Verified live: the goosegrass answer
  still passes, because the Clemson factsheet it cites genuinely
  mentions both goosegrass and Celsius chemistry — in different
  contexts. Co-presence in a chunk is not evidence of pairing, and a
  post-hoc text check cannot distinguish them without unacceptable
  false positives. Failure 2 is fundamentally a *retrieval* problem:
  the chunks that pair goosegrass with fluazifop existed and never
  reached the synthesizer. Fixing it belongs in retrieval quality, not
  here. This ADR deliberately solves the half it can solve well.
- **Vocabulary matching is brittle on abbreviated names.** The
  goosegrass answer said "thiencarbazone + iodosulfuron"; the bridge
  vocabulary holds "thiencarbazone-methyl" and
  "iodosulfuron-methyl-sodium", so neither matched and neither was
  checked. Substring matching on chemical names is a blunt instrument.
- **The check is strict in a way that can feel unfair.** If an action
  names a product's full three-ingredient list while the cited chunk
  mentions only two, the third is flagged. That is intended — if the
  passage you cite doesn't say it, don't claim it — but it will
  generate re-prompts on answers a human would consider fine.
- Fixture debt surfaced immediately: several existing tests cited
  snippets that appeared nowhere in their own passages. Those fixtures
  could never have caught the real bug. They have been made faithful,
  which is a quiet improvement in the test suite's honesty.

## Alternatives considered

**An LLM judge scoring whether each claim is supported.** Rejected as
circular — using a model to catch a model's fabrication introduces a
second thing that can hallucinate, on the safety-critical path, at cost
and latency on every call. A deterministic check that catches less but
never invents is the better trade for a guardrail.

**Requiring exact substring quotes.** Rejected: it would fail
legitimate paraphrase that rule 5 explicitly allows, and PDF-extracted
text has whitespace and hyphenation artifacts that break literal
matching even for faithful quotes.

**Making it a Pydantic validator on `Recommendation`.** Not possible as
designed — the validator has no access to the retrieved passages. This
is very likely why the check was never written: the natural home for it
looked like the schema, and the schema can't see the evidence. It has
to live where `passages` is in scope, which is the orchestrator.

**Blocking rather than re-prompting.** Rejected: the existing ADR 0003
pattern of one corrective retry then refuse already works, and a model
told exactly which term was unsupported can often fix it by quoting the
source properly instead of refusing outright.
