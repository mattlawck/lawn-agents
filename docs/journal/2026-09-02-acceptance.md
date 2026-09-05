# 2026-09-02 — The acceptance run, three months late

Picked this up after a long gap. Last commit was June 12; the plan at
the time was to re-run the two acceptance scenarios against the bridges
that PRs #31, #32, and #39 had just landed, and then the session ended.
Three months of not touching it later, here's what I found.

## First: the environment had rotted

Nothing ran. `make check` died at the type step with

```
make: .venv/bin/mypy: No such file or directory
```

which is a lie — `mypy` is right there in `.venv/bin`. What was missing
was the *interpreter its shebang points at*. The venv was built against
Homebrew's `python@3.14`, and somewhere in the intervening three months
that formula left my machine. Every console script in the venv is a
text file whose first line names a dead path. `ruff` kept working the
whole time, because it's a standalone binary with no shebang to break.

`uv` was gone too. So the recovery was: `brew install uv`, then
`uv sync --all-extras --dev` to rebuild from the committed lockfile.
Total time about ninety seconds, and `make check` came back green —
295 tests, 89.53% coverage.

The lesson worth writing down: **a lockfile is only half of
reproducibility.** `uv.lock` pinned every Python package perfectly and
none of it mattered, because the thing the lock doesn't pin is the
interpreter. Reinstalling `uv` rather than the Homebrew Python is the
actual fix — uv manages its own toolchains, so the next `brew cleanup`
can't take the project down with it.

## Scenario 2 — GrubX, revisited

This is the one that refused back on June 1, and the refusal was the
whole reason ADR 0007's brand bridge exists. Same question, three
months of bridge work later:

```
╭─ Recommendation ──────────────────────────────────────────────────╮
│ It is too late for curative grub treatment with GrubX this season.│
╰───────────────────────────────────────────────────────────────────╯

This week:
  • insecticide — Do not apply insecticides containing
    chlorantraniliprole (e.g., GrubX) for curative grub control. The
    recommended application window for curative treatments is July
    through August, and applying now (September) may not be effective
    for this season's grub population. [1]
  • monitoring — Continue to monitor for signs of grub activity...

Sources:
  [1] hgic.clemson.edu/factsheet/white-grub-management-in-turfgrass
```

The bridge works. "GrubX" went in, `chlorantraniliprole` came out the
other side, and the Clemson white-grub factsheet — the exact document
the June entry said was "one URL away" — is doing the grounding. The
answer is temporally aware (it knows September is late) and it cites.

That's the scenario passing. But the *logs* tell a more interesting
story than the output does.

## The gate rejected the passage that answered the question

Here is the retrieval trace, in order:

```json
{"score": 0.644, "source": ".../white-grub-management-in-turfgrass/",
 "event": "knowledge.is_weak.medium_lexical_miss"}
{"relevant": false, "event": "knowledge.is_weak.gate_verdict"}
{"event": "orchestrator.retrieval_weak.invoking_research"}
{"error": "No results found.", "event": "research.search_failed"}
```

Read that again. The Clemson white-grub factsheet came back at 0.644 —
the medium band. The lexical-overlap check missed. It escalated to the
LLM relevance gate from PR #39, and **the gate said the white-grub
factsheet was not relevant to a question about grub treatment.** So the
orchestrator declared retrieval weak, fired the research subagent, the
search found nothing, and then the synthesizer went ahead and wrote a
correct, well-cited answer *out of the passage the gate had just
rejected*.

The output was right. The reasoning that produced it was wrong at two
separate layers, and I'd never have known from the rendered panel.

The cause is an asymmetry I introduced without noticing. ADR 0008's
weed bridge feeds **both** paths — `orchestrator.py:142-144` builds
`alias_terms` from the matched weeds and passes them into `is_weak` as
`extra_terms`, and `expand_query_with_weed_aliases` widens the
retrieval query. ADR 0007's brand bridge feeds **only the synthesizer
prompt**, down at line 376. It never reaches `is_weak`.

So the lexical check compared `{grubx, treat, late}` against a passage
that says `chlorantraniliprole` and `white grub` and found, correctly,
zero overlap. The one piece of vocabulary that would have bridged them
was sitting in `data/chemicals.yaml`, already parsed, sixty lines
further down the same function.

Two bridges, built five days apart, wired to different depths. The
second one taught me the lesson the first one needed.

## Scenario 1 — planning to a target date

```
lawn-agents --ask "My daughter's birthday is July 13th. What
  fertilizer and weed plan should I follow to have everything peak
  then?"
```

It rolled the date forward on its own — July 13, 2026 has passed, so
it planned to **July 13, 2027** and said so in the headline. Then it
worked backwards: fall pre-emergent at 50-55°F soil temp, spring
pre-emergent at 65°F, first macro-fert on the same 65°F gate, a 4-6
week nitrogen cadence through spring, chelated iron for color without
push growth, and a foliar 28-0-0 in early July 2027 as the final
pre-event boost. Nine items. Every chemical one cited.

That's the capability the scenario was written to test, and it works.

Two things I don't love.

**Everything rendered under "This month."** All nine items landed in
`monthly_actions`, and `notify.py:91` hardcodes that section's label to
`"This month"` — so a plan spanning Fall 2026 through July 2027 is
printed under a header claiming it's for September. The model noticed
the mismatch and worked around it by stuffing timing into the prose
("typically Spring 2027", "Late Spring / Early Summer 2027"), which is
why the output reads fine despite the header being wrong.

The annoying part: `CalendarItem` already has `earliest` and `latest`
date fields (`models.py:187-188`), and `notify._format_window` has
always rendered them when set. The schema anticipated this exact need
and the renderer was ready for it. The only missing piece was the
synthesizer prompt, which never asked — so the model put the timing
where it could, in the prose. `planner.md` had the rule; the `--ask`
path's `synthesizer.md` didn't.

A whole capability sat wired at both ends and disconnected in the
middle, and the output looked fine enough that I'd never have caught
it by reading the panel.

**The corpus has a house brand.** Three of the products named — Stress
Blend 7-0-20, 17-7-6 Freedom, 28-0-0 N-Charge — are Yard Mastery SKUs,
and they're cited to pages 51, 63, and 68 of the Warm-Season E-Guide,
which is a Lawn Care Nut product. The guardrail is satisfied: those are
real citations to a real document. But a vendor guide recommending the
vendor's own line is a different kind of evidence than Clemson saying
"apply nitrogen at this soil temperature," and right now the system
can't tell those apart. Every source in `<sources>` carries equal
weight.

That's not a bug I can fix with a validator. It's a question about
whether provenance should carry a trust tier.

## Where this leaves Phase 1

Both acceptance scenarios pass. Neither fabricated anything; neither
refused when it shouldn't have. By the standard I set in May — "failures
here mean the design isn't done even if all unit tests pass" — Phase 1
is signed off.

The three findings are all follow-ups, not blockers — and all three
are fixed in the same sitting:

1. Brand bridge doesn't reach `is_weak` — wrong gate verdict, wasted
   research call. `_bridge_lexical_terms` now merges both bridges into
   the `extra_terms` the lexical check consumes.
2. `earliest` / `latest` unpopulated — synthesizer.md gains the rule
   planner.md already had, and the `monthly_actions` heading now reads
   the item dates instead of always claiming "This month."
3. No trust tiering on sources — ADR 0009. Four tiers classified from
   configured patterns, surfaced as `tier=` in `<sources>`, with both
   prompts taught to prefer extension for agronomy and disclose when a
   vendor guide is the only support for a branded product.

## What the blog should say

1. **Lockfiles don't pin interpreters.** Perfect dependency resolution,
   zero reproducibility, because the one thing outside the lock was the
   thing that broke. Short, concrete, and everybody has hit it.
2. **The output being right is not evidence the pipeline is right.**
   The GrubX answer was good. Two layers underneath it were wrong, and
   only structlog showed me. This is the strongest argument I have for
   logging decisions rather than just results — and it's a much better
   story than any of my passing tests.
3. **Bridges have a depth, and you have to choose it deliberately.**
   Same pattern implemented twice, wired to different layers, and the
   shallower one silently degraded a decision three modules away.
