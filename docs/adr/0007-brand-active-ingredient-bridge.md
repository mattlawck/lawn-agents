# ADR 0007 — Brand → active-ingredient bridge

- **Status**: Accepted — 2026-06-08
- **Deciders**: Matt

## Context

Extension publications (Clemson HGIC, NCSU TurfFiles, UGA Extension, UF
IFAS) — the authoritative sources behind ADR 0003's never-guess
guardrail — discuss turf chemicals by **active ingredient**:
*chlorantraniliprole*, *imidacloprid*, *halosulfuron-methyl*. Real users
ask questions by **retail brand**: *GrubX*, *Merit*, *Sedgehammer*.

On the first end-to-end run (journal entry
`docs/journal/2026-06-01-first-real-run.md`), Gemini 2.5 Flash refused
the question *"Is it too late to treat with GrubX?"* — pointing the user
at Clemson HGIC for more information. The refusal was structurally
correct: the corpus had no passage mentioning "GrubX" literally, so
ADR 0003 fired. But the refusal was also *wrong in spirit*: Clemson HGIC
absolutely speaks to chlorantraniliprole timing for white-grub control,
and GrubX has been chlorantraniliprole-based for years. The synthesizer
treated the brand as an opaque term it had to find verbatim, and never
considered the chemistry.

This is a small, well-defined gap with three plausible fixes.

## Decision

A **curated brand → active-ingredient bridge**, loaded from
`data/chemicals.yaml`, injected into the synthesizer prompt as an
optional `<brand_bridge>` block when (and only when) the user's question
mentions a known brand.

Mechanics:

1. **`data/chemicals.yaml`** is the single source of truth. Each entry
   maps a brand to one or more active ingredients, a category
   (`insecticide`, `herbicide`, `fungicide`, `fertilizer`,
   `micronutrient`), and an optional short note. Schema enforced by
   `ChemicalsConfig` / `ChemicalBrand` (`src/lawn_agents/models.py`),
   loaded by `Settings.load`.
2. **`orchestrator.detect_brands_in_question`** does a case-insensitive,
   word-boundary regex match against the user's question. Returns a
   `dict[str, ChemicalBrand]` of matches, possibly empty.
3. **`orchestrator._brand_bridge_text`** renders matches into an
   `<brand_bridge>` XML block, which the synthesizer system prompt
   (`prompts/synthesizer.md`) documents as: passages discussing an
   active ingredient apply to the corresponding brand; citations point
   at the passage, not at the bridge itself.
4. **Planner symmetry.** `planner._planner_user_prompt` runs the same
   detector against the freeform `target` string and includes the same
   `<brand_bridge>` block. Anywhere a question reaches the synthesizer,
   the bridge runs. (See "Planner integration" below.)
5. **No active ingredient is invented.** When no brand matches, the
   block is omitted entirely and the prompt is byte-identical to the
   pre-bridge baseline. The bridge is opportunistic, not pervasive.

The bridge does **not** weaken ADR 0003. It does not become a source.
The synthesizer still must ground every chemical recommendation in a
cited passage from `<sources>`. The bridge only tells the model
*"passages about chlorantraniliprole are admissible evidence for GrubX
questions"* — the passage itself, and its citation, still has to exist.

## Alternatives considered

1. **LLM-side brand expansion.** Prompt the synthesizer to "consider
   common active ingredients for any brand mentioned." Rejected: this is
   exactly the hallucination surface ADR 0003 was built to close. Asking
   the model to recall brand chemistry from training data reintroduces
   the failure mode the guardrail forbids.
2. **Fuzzy / synonym-aware retrieval at the RAG layer.** Expand the
   query embedding with brand synonyms before retrieval. Rejected for
   now: opaque (the synthesizer can't see why a passage was retrieved),
   harder to test, and the failure mode we saw was synthesis-side, not
   retrieval-side. Worth revisiting if retrieval becomes the bottleneck.
3. **Pre-ingest brand glossary as a corpus document.** Author a
   "brand glossary" markdown file and ingest it like any other source.
   Rejected: the synthesizer would then cite the glossary as the source
   of the timing recommendation, which is wrong — the timing comes from
   the Clemson factsheet, not from us. Keeping the bridge structurally
   separate from sources preserves citation integrity.

## Planner integration

The planner takes a freeform `target` string (e.g.,
`"July 2026 pre-emergent window"`, `"plan around Acelepryn cycle"`).
Today's targets are date/scope strings without brands, but the planner's
synthesizer call has the same structural shape as the ad-hoc one, and
the same failure mode would surface the moment a brand appears in a
target.

Wiring cost was assessed and is negligible:

- Token cost when a brand matches: ~100–140 added input tokens
  (preamble + one to two brand lines). At Gemini 2.5 Flash rates
  ($0.30/1M input), ~$0.000042 per matched-brand invocation; at
  2.5 Pro ($1.25/1M), ~$0.000175. When no brand matches, the block is
  empty and cost is zero.
- Code surface: one parameter threaded through `_synthesize_plan` →
  `_planner_user_prompt`, plus one unit test asserting the bridge
  appears for a brand-named target.

Symmetry won. Both entry points run the bridge.

## Consequences

- **Maintenance burden.** `data/chemicals.yaml` is hand-curated and will
  drift from manufacturer reality (Scotts reformulates SKUs; brands
  change ingredient over time). Mitigation: keep the seed list small,
  scoped to warm-season Southeast US turf, and annotate brands with
  short notes (e.g., *"recent formulations are chlorantraniliprole"*)
  rather than asserting timeless truth. If a brand's chemistry changes,
  the bridge can lag behind reality and produce a wrong answer; this is
  a real risk, accepted in exchange for the brand-blind refusal
  problem.
- **Scope.** The seed list covers products a Mt Pleasant SC homeowner
  on Zeon Zoysia is plausibly going to encounter. It is not a
  comprehensive pesticide database. PRs to add brands are welcome;
  out-of-scope expansion (cool-season-only products, agricultural
  formulations) will be declined.
- **No new failure mode against ADR 0003.** The bridge cannot cause the
  synthesizer to recommend an uncited product. The schema validator
  (`Recommendation`) still requires `citations: list[Citation]` with
  `min_length=1` for chemical-category items. The bridge only changes
  which retrieved passages the synthesizer is willing to *associate*
  with a brand-named question.
- **Tests.** `tests/unit/test_chemicals.py` covers schema validation,
  the seeded YAML's contents, `Settings.load` integration, and frozen
  brand entries. `tests/unit/test_orchestrator.py` /
  `tests/unit/test_planner.py` gain bridge-injection assertions.

## Cross-references

- ADR 0003 — never-guess guardrail. The bridge runs *inside* that
  guardrail, not around it.
- ADR 0005 — self-extending RAG. The bridge complements the research
  subagent: the subagent extends the corpus for unfamiliar topics; the
  bridge makes the existing corpus reachable from brand-named
  questions.
- `docs/journal/2026-06-01-first-real-run.md` — the GrubX refusal that
  motivated this ADR.
