# ADR 0009 — Source trust tiers

- **Status**: Accepted — 2026-09-02
- **Deciders**: Matt

## Context

ADR 0003 established the never-guess guardrail: a chemical-category
`CalendarItem` must carry at least one `Citation`, enforced by a
Pydantic validator and re-prompted once on failure. It has worked. The
2026-06-01 GrubX refusal and the 2026-09-02 acceptance run both show
the model grounding its chemical claims rather than inventing them.

But the guardrail checks that a citation *exists*. It says nothing
about what the citation points at.

The 2026-09-02 acceptance run made that gap concrete. Asked to plan a
lawn peaking on a July 13th birthday, the synthesizer produced a
nine-item schedule naming three specific products:

- Stress Blend 7-0-20 — cited to *2025 Warm Season E-Guide*, p.105
- 17-7-6 Freedom — cited to the same guide, p.68
- 28-0-0 N-Charge — cited to the same guide, p.51

Every one of those citations is real and verifiable. The guardrail
passed, correctly. But all three are Yard Mastery SKUs, and the
E-Guide is a Lawn Care Nut product — a vendor guide recommending the
vendor's own line. Meanwhile the same plan drew its soil-temperature
thresholds from Clemson HGIC, a vendor-neutral extension source.

To the synthesizer, those two kinds of evidence were indistinguishable.
`_format_sources` rendered every passage identically:

```
[1] source_id='...' title='...' page=105 score=0.712
```

There is no signal in that line that separates "Clemson says apply
nitrogen at this soil temperature" from "the guide selling this
fertilizer says to buy this fertilizer."

This is not a hallucination problem and it is not fixable with a
stricter validator — the citations are genuine. It is a *weighting*
problem, and weighting is the synthesizer's job. What was missing was
the information it needed to do that job.

Matt's own working notes already draw this distinction. The
chemical-adding workflow separates two axes of evidence: **timing /
planning** (narrow, curated authority — Clemson HGIC, NCSU TurfFiles,
UGA/UF Extension) versus **product / application** (broader sources OK
— CDMS labels, manufacturer guides, specialist retailer pages for SKU
math). The rule of thumb there is that *trust travels with the
document, not the domain hosting it*: a Bayer label mirrored on a
retailer's site is still a label. That taxonomy existed in prose and
in the seed-URL comments; it had never reached the code.

## Decision

Classify every retrieved passage into a `SourceTier` and surface the
tier in the `<sources>` block, then teach both prompts to weigh tiers.

Four tiers:

| Tier | Meaning | Best for |
|---|---|---|
| `extension` | Land-grant extension, university, government | Timing, rates, thresholds, agronomic judgment |
| `label` | EPA-registered label or manufacturer product guide | Rate, weed list, REI, turf tolerance |
| `vendor` | Sod producer, retailer, product marketing | Cultivar specifics, package/coverage math |
| `unknown` | Unmatched provenance | Treated as `vendor` |

**Classification is configured, not hardcoded.** `knowledge.source_tiers`
in `config.yaml` holds three pattern lists. Each pattern is a
case-insensitive substring matched against the passage's URL,
`source_id`, and `source_title` combined.

Two consequences of that choice, both deliberate:

1. **Substring, not host matching.** A local corpus PDF has no URL.
   Matching against the title too means `"Warm Season E-Guide"` tiers
   the same way `"hgic.clemson.edu"` does — one mechanism for both
   kinds of provenance, rather than a separate corpus-file mapping.

2. **Precedence is `label` → `extension` → `vendor`.** Most specific
   first, so trust travels with the document. A label PDF mirrored on
   `trianglecc.com` matches `-label` before anything else can claim
   it, which is exactly the behavior the cross-match rule assumes.

The prompts (rule 7 in both `synthesizer.md` and `planner.md`) instruct
the model to prefer extension sources for timing and agronomy, prefer
labels for application specifics, and **disclose in the action text**
when a vendor or unknown source is the only support for naming a
specific branded product. Where an extension source and a vendor source
disagree, follow extension and note the disagreement.

## Consequences

**Good:**

- The vendor-bias problem becomes visible to the model rather than
  invisible to everyone. It can now say "per the Yard Mastery guide"
  instead of presenting vendor copy in the same voice as Clemson.
- The taxonomy already in Matt's head is now executable and testable.
- Adding a source tier is a config edit, not a code change — the same
  extension point the domain allowlist already uses.
- `_format_sources` had drifted into byte-identical private copies in
  `orchestrator.py` and `planner.py`. Tiering gave them a reason to
  diverge, so they are now one `knowledge.format_sources`.

**Bad / accepted:**

- **Substring matching is blunt.** `.edu` as an extension pattern will
  tier a student blog on a university domain as extension. The
  alternative — parsing hosts and maintaining a suffix list — is more
  code for a corpus the user curates by hand anyway. Revisit if the
  research subagent starts pulling in wider domains.
- **This is prompt-level enforcement, not structural.** Unlike ADR
  0003's citation requirement, nothing in the schema *forces* the model
  to respect tiers. That is intentional for now: the right response to
  a vendor-only source is disclosure, not refusal, and "refuse unless
  extension-backed" would break legitimate SKU-math answers. If
  disclosure proves unreliable in practice, the escalation path is a
  validator that requires a disclosure phrase when every citation on a
  chemical item is `vendor`/`unknown`.
- **Tiers are computed at prompt-format time, not stored on the
  chunk.** Re-tiering is therefore free (edit config, next query picks
  it up) but costs a few string scans per query. At `rerank_top_k=5`
  this is not measurable.

## Alternatives considered

**Rank or filter by tier during retrieval.** Boost extension passages
so they crowd out vendor ones. Rejected: it would have suppressed the
SKU-math passages that make a recommendation actionable, and it hides
the trade-off inside a scoring function instead of showing the model
both and letting it weigh them. Retrieval should find what's relevant;
weighting authority is a synthesis decision.

**Store the tier on the chunk at ingest time.** Rejected: it freezes a
judgment that the user should be able to revise, and would require a
re-ingest to change a classification. Config-time classification means
Matt can reclassify a source and see the effect on the next question.

**A numeric trust score rather than named tiers.** Rejected as false
precision. There is no meaningful sense in which Clemson is 0.9 and
Yard Mastery is 0.6, and the model reasons better about a named
category with a stated purpose than about a number with no units.
