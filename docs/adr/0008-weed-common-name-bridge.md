# ADR 0008 — Weed common-name → scientific-name bridge

- **Status**: Accepted — 2026-06-09
- **Deciders**: Matt

## Context

ADR 0007 closed the *brand*-vocabulary gap: users ask about "GrubX,"
extension publications discuss "chlorantraniliprole," and the brand
bridge tells the synthesizer those are the same thing. The Phase 1
demo confirmed it works — the GrubX question went from a brand-blind
refusal to a grounded recommendation citing the Clemson white-grub
factsheet.

The *next* canonical question — *"I have Japanese clover in the yard,
what should I use?"* — re-surfaces the same shape of problem on the
other axis:

- The user asks by **weed common name**: *Japanese clover*.
- Manufacturer labels use **scientific name** *or* an older common-name
  form. The Bayer Celsius WG label lists this exact weed as
  **"Annual lespedeza" (*Lespedeza striata*)** — same plant, but
  *Kummerowia striata* is the modern Linnaean placement and
  *"Japanese clover"* is the everyday name a homeowner would use.
- Clemson HGIC's weed-management factsheet covers chemistries
  (*thiencarbazone-methyl + iodosulfuron-methyl + dicamba*, sold as
  Celsius WG) that control this weed — but its weed-by-weed prose uses
  the same "annual lespedeza" terminology as the label.

Result: BGE-small embedding similarity between *"Japanese clover"* and
*"annual lespedeza"* / *"Lespedeza striata"* is too low for retrieval to
bridge. Probed directly against the live corpus
(2026-06-09; 281 chunks; Bayer Celsius WG label = 19 chunks):

- `"Japanese clover"` → top hit at score 0.696 is the **sedge**
  factsheet (irrelevant); the Bayer Celsius label is **not in top-5**.
- `"annual lespedeza control"` → top hit at score 0.688 is the
  zoysiagrass calendar; label still not in top-5.
- `"thiencarbazone-methyl"` → Bayer label at **0.743 (top hit)**.

So the corpus has the answer. Retrieval can find it under the
technical name. It cannot bridge from the question's vocabulary.

This is identical in shape to the brand-vocabulary gap. The bridge
solution generalizes — to the weed axis, with weed common names as
the input.

## Decision

A **curated weed common-name → alias bridge**, loaded from
`data/weeds.yaml`, injected into the synthesizer prompt as an optional
`<weed_bridge>` block when (and only when) the user's question or
planner target mentions a known weed common name.

Mechanics, parallel to ADR 0007:

1. **`data/weeds.yaml`** is the single source of truth. Each entry maps
   a common name to a list of `aliases` (scientific names + label-form
   common names + historical synonyms), a `category` (`broadleaf`,
   `grassy`, `sedge`), and an optional short note. Schema enforced by
   `WeedsConfig` / `WeedAlias` (`src/lawn_agents/models.py`), loaded by
   `Settings.load`.
2. **`orchestrator.detect_weeds_in_question`** does a case-insensitive,
   word-boundary regex match against the question. Returns a
   `dict[str, WeedAlias]` of matches.
3. **`orchestrator._weed_bridge_text`** renders matches into a
   `<weed_bridge>` XML block. The synthesizer system prompt
   (`prompts/synthesizer.md`) documents the contract: passages
   discussing any of the aliases apply to the original question's
   common name. Citations still point at the original passage; the
   bridge is not a source.
4. **Planner symmetry** matches ADR 0007. `planner._planner_user_prompt`
   runs the same detector against the freeform `target` string.
5. **No fabrication.** When no weed matches, the block is omitted and
   the prompt is byte-identical to the pre-bridge baseline. The bridge
   is opportunistic, not pervasive.

The bridge does not weaken ADR 0003. The synthesizer must still ground
every chemical recommendation in a cited passage. The bridge only tells
the model *"passages about annual lespedeza apply to Japanese clover
questions"* — the passage, and its citation, must still exist.

## Alternatives considered

1. **Generalize ADR 0007 into a single "alias bridge."** Combine
   brand-bridge and weed-bridge into one mechanism. Rejected for now:
   the brand-bridge data shape (active ingredients + category) and the
   weed-bridge shape (aliases + category) are similar but not
   identical, and the synthesizer guidance differs ("brand X is sold
   as chemistry Y" vs "weed X is also called Y"). A premature
   abstraction would obscure both. Revisit if a third axis appears.
2. **Augment retrieval with synonym expansion at query time.** Inject
   alias terms into the embedding-time query before retrieval.
   Rejected for the same reason ADR 0007 rejected it: opaque
   (the synthesizer can't see *why* a passage was retrieved), harder
   to test, and the failure mode is synthesis-side awareness, not
   retrieval recall when the right vocabulary is supplied.
3. **Use a larger embedding model.** BGE-small struggles with
   common→technical name mapping; BGE-large or e5-large would do
   better. Deferred: heavier change with broader effects (memory,
   latency, embedding-dim migration). The bridge is the cheaper and
   more transparent fix.
4. **Pre-ingest a "weed glossary" document.** Like ADR 0007 considered
   and rejected for brands: the synthesizer would cite the glossary
   as the source of the chemistry recommendation, which is wrong —
   the chemistry comes from the label or extension factsheet.

## Scope of the seed list

Seed `data/weeds.yaml` covers warm-season SE turf weeds a Mt Pleasant
SC homeowner on Zeon Zoysia is likely to encounter, plus the specific
weeds the brand-bridge products (Celsius WG, Sedgehammer, Acelepryn,
etc.) actually control. **Not** a comprehensive weed taxonomy; PRs to
add weeds welcome, out-of-scope expansions (cool-season-only,
agricultural) declined.

Initial coverage:

- Broadleaf: Japanese clover (= annual lespedeza), dollarweed (=
  pennywort), doveweed, Virginia buttonweed, Florida betony, Florida
  pusley, spurge, lawn burweed, chickweed (common + mouse-ear), henbit
- Grassy: large/smooth crabgrass, dallisgrass, goosegrass, sandbur
- Sedge: yellow nutsedge, purple nutsedge, green kyllinga

## Consequences

- **Maintenance burden.** Weed scientific names occasionally migrate
  between genera (the Japanese clover / *Lespedeza striata* →
  *Kummerowia striata* shift is the classic example). Mitigation: keep
  both old and new names in `aliases` so the bridge survives either
  label revision and either user vocabulary.
- **No new failure mode against ADR 0003.** The bridge cannot cause
  the synthesizer to recommend an uncited product. The `Recommendation`
  schema still requires `citations: list[Citation]` with `min_length=1`
  for chemical-category items.
- **Tests.** `tests/unit/test_weeds.py` covers schema, seeded YAML, and
  `Settings.load` integration. `tests/unit/test_orchestrator.py` and
  `tests/unit/test_planner.py` gain bridge-injection assertions.

## Cross-references

- ADR 0003 — never-guess guardrail. The bridge runs *inside* the
  guardrail.
- ADR 0007 — brand bridge. This ADR is its sequel; same pattern,
  different axis.
- The Japanese clover retrieval probe is in the working notes of
  PR #31, which shipped ADR 0007. That probe motivated this ADR.
