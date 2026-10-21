# Phase 2 notes — trees and shrubs

- **Status**: Working notes, not a design. Captured 2026-10-21 so Phase 2
  planning starts from real questions instead of a blank page.
- **Scope today**: `subjects/tree.py` and `subjects/shrub.py` are
  placeholders (see `tree.py` for the rollout checklist).

## Plant inventory

| Plant | Subject | Notes |
|---|---|---|
| Palms (species to confirm) | tree | Dead-frond trimming is the recurring task |
| Live oaks (young + mature) | tree | Herbicide root-uptake risk under pine straw |
| Pineapple guava (*Feijoa sellowiana*) | shrub | Fruits Sep–Nov; shaping prunes |
| Sweet viburnum (*V. odoratissimum*) | shrub | Broadleaf; dicamba-sensitive |
| Hydrangeas | shrub | Original Phase 2 scope |
| Ornamental muhly, breeze grass | shrub | Grass ornamentals — sethoxydim *targets* |

`shrub.py` currently says "hydrangeas and other ornamental shrubs"; the
inventory above is the real list.

## Seed questions (acceptance scenarios)

Phase 1 was validated against questions Matt actually asked. These are
the first ones for Phase 2:

1. *"Any reason not to trim dead palm fronds and lightly shape the
   pineapple guava right now?"* (asked 2026-10-21, late October)
2. *"I'm spraying Grass Getter / SedgeHammer near the bed this weekend —
   is anything in it at risk?"* — the cross-subject case below.

Expected shape of answer for (1), to check retrieval against once a
corpus exists — **unverified, from general knowledge, not a citation**:

- Palms: removing fully brown fronds is fine any time. Leave green and
  yellowing fronds — palms pull potassium from older fronds, and
  over-pruning ("hurricane cut") is a common cause of decline. Sanitize
  tools between palms (Fusarium wilt spreads on pruning tools).
- Pineapple guava: a light shaping trim is fine. Avoid heavy pruning in
  late fall (pushes tender growth before frost) and expect to cut off
  some fruit or next year's flowering wood. Major pruning after harvest
  or in late winter.

## Cross-subject interactions (the real reason Phase 2 is hard)

Phase 1 already had to encode shrub and tree safety inside *lawn*
chemistry — see the Grass Getter, Spectracide, and Glyphosate notes in
`data/chemicals.yaml`:

- Dicamba is root-absorbed and soil-mobile: oak-injury risk under pine
  straw, broadleaf risk to viburnum and guava.
- Sethoxydim labels list pineapple guava and several palms as tolerant;
  sweet viburnum is not listed.
- SedgeHammer label: color plants and herbaceous ornamentals may be
  injured when transplanted into treated areas — relevant to the spring
  bed being prepped on the zoysia strip.

Today these live as prose in brand notes. Phase 2 probably wants them as
structured data the synthesizer and grounding can both see: a plant ×
active-ingredient sensitivity table, so a lawn recommendation can warn
about the shrub three feet away.

## Sources to seed

Following `reference_sources` conventions (extension first):

- Clemson HGIC — palm, pineapple guava, and pruning factsheets
- UF/IFAS — palm nutrition and pruning (the authoritative palm source
  for the Southeast)
- Product labels via CDMS for anything recommended near beds

## Design questions to settle

- Router: dispatch on subject, or let one question span lawn + shrub
  (question 2 above is inherently both)?
- Calendar: add pruning items to `data/calendar.yaml` with a `subject`
  field, or a separate program file per subject?
- Pruning isn't chemical, so ADR 0003's citation requirement doesn't
  bind — but bad pruning timing is still costly. Decide whether
  non-chemical advice needs a softer grounding tier.
