"""Vocabulary bridges — translating the user's words into the corpus's.

The user and the corpus do not speak the same language, and the gap runs
both ways:

- People ask about **brands**. Extension publications discuss **active
  ingredients**. "GrubX" never appears in a Clemson factsheet;
  "chlorantraniliprole" never appears in a question. (ADR 0007)
- People use **everyday common names**. Labels use **scientific names**
  or older trade forms. "Japanese clover" is "annual lespedeza" on the
  Celsius label and *Lespedeza striata* in the taxonomy. (ADR 0008)

Unbridged, that mismatch produces a *false refusal* — the corpus covers
the question perfectly and the system says it cannot help, which is the
most damaging failure this project has, because it looks like caution.

These bridges feed three separate consumers, and every one of them had
to be wired individually. Each omission was its own bug:

1. The **synthesizer prompt**, so the model knows the mapping.
2. **`is_weak`'s lexical check**, which otherwise compares the user's
   vocabulary against a passage that never uses it and escalates to a
   relevance gate that is equally blind. (fixed 2026-09-02)
3. **Retrieval**, which otherwise never searches for the chemistry at
   all. (fixed 2026-09-14)

They live here rather than in the orchestrator because the planner needs
them too, and was reaching across module boundaries for private names to
get them.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lawn_agents.models import ChemicalBrand, ChemicalsConfig, WeedAlias, WeedsConfig


def detect_brands_in_question(
    question: str, chemicals: ChemicalsConfig
) -> dict[str, ChemicalBrand]:
    """Return chemical brands from `chemicals.brands` mentioned in `question`.

    Case-insensitive, word-boundary match. Brand names containing spaces
    are matched as exact phrases. Used by the orchestrator + planner to
    inject a brand → active-ingredient bridge into the synthesizer
    prompt; see ADR 0007.
    """
    matched: dict[str, ChemicalBrand] = {}
    q_lower = question.lower()
    for name, brand in chemicals.brands.items():
        pattern = r"\b" + re.escape(name.lower()) + r"\b"
        if re.search(pattern, q_lower):
            matched[name] = brand
    return matched


def brand_bridge_text(matched: dict[str, ChemicalBrand]) -> str:
    """Render the `<brand_bridge>` block for the synthesizer prompt.

    Empty string when nothing matched, so the caller can concatenate
    unconditionally without emitting an empty XML block.
    """
    if not matched:
        return ""
    lines = [
        "<brand_bridge>",
        (
            "The question mentions one or more product brands. Each brand's "
            "active ingredient(s) are listed below. Passages in <sources> "
            "that discuss an active ingredient apply to the corresponding "
            "brand. Cite the passage, not the bridge."
        ),
    ]
    for name, brand in sorted(matched.items()):
        ais = ", ".join(brand.active_ingredients)
        line = f"- {name} ({brand.category.value}): active ingredient(s): {ais}."
        if brand.notes:
            line += f" {brand.notes}"
        lines.append(line)
    lines.append("</brand_bridge>")
    return "\n".join(lines)


def detect_weeds_in_question(question: str, weeds: WeedsConfig) -> dict[str, WeedAlias]:
    """Return weed common names from `weeds.weeds` mentioned in `question`.

    Case-insensitive, word-boundary match. Names containing spaces are
    matched as exact phrases. Used by the orchestrator + planner to
    inject a weed common-name → alias bridge into the synthesizer
    prompt; see ADR 0008.
    """
    matched: dict[str, WeedAlias] = {}
    q_lower = question.lower()
    for name, weed in weeds.weeds.items():
        pattern = r"\b" + re.escape(name.lower()) + r"\b"
        if re.search(pattern, q_lower):
            matched[name] = weed
    return matched


def weed_bridge_text(matched: dict[str, WeedAlias]) -> str:
    """Render the `<weed_bridge>` block for the synthesizer prompt."""
    if not matched:
        return ""
    lines = [
        "<weed_bridge>",
        (
            "The question mentions one or more weed common names. Each weed's "
            "scientific names and label-form aliases are listed below. "
            "Passages in <sources> that discuss any alias (e.g., scientific "
            "name or older common name) apply to the user's question. Cite "
            "the passage, not the bridge."
        ),
    ]
    for name, weed in sorted(matched.items()):
        aliases = ", ".join(weed.aliases)
        line = f"- {name} ({weed.category.value}): also called {aliases}."
        if weed.notes:
            line += f" {weed.notes}"
        lines.append(line)
    lines.append("</weed_bridge>")
    return "\n".join(lines)


def expand_query_with_weed_aliases(question: str, matched: dict[str, WeedAlias]) -> str:
    """Append weed aliases to the retrieval query so the label surfaces.

    The bridge tells the *synthesizer* about common→technical name
    aliases, but retrieval still embeds the raw question. BGE-small
    similarity between "Japanese clover" and "Annual lespedeza" is
    weak, so the Bayer Celsius WG label (which uses the older form) is
    never retrieved on the homeowner phrasing. Appending the aliases
    to the retrieval query brings the label into top-k. The
    synthesizer still sees the original question via the `<question>`
    block — only the retrieval path is widened.
    """
    if not matched:
        return question
    extra_terms: list[str] = []
    for weed in matched.values():
        extra_terms.extend(weed.aliases)
    return f"{question} {' '.join(extra_terms)}"


def bridge_lexical_terms(
    weed_matches: dict[str, WeedAlias],
    brand_matches: dict[str, ChemicalBrand],
) -> list[str]:
    """Vocabulary the bridges add to the question, for `is_weak`'s lexical check.

    Both bridges exist because the user's vocabulary and the corpus's
    vocabulary differ: the user says "GrubX" or "Japanese clover", the
    extension factsheet says "chlorantraniliprole" or "annual
    lespedeza". The synthesizer gets told about both mappings via
    `<brand_bridge>` / `<weed_bridge>`, but `knowledge.is_weak` runs
    *before* synthesis and only sees the raw question.

    Without this, the lexical-overlap check compares the user's words
    against a passage that never uses them, misses, and escalates to
    the LLM relevance gate — which is equally blind and returns "not
    relevant." Observed live on 2026-09-02: "Is it too late to treat
    with GrubX?" retrieved the Clemson white-grub factsheet at 0.644
    (medium band), was marked weak, fired a pointless research call,
    and then the synthesizer answered correctly from that very passage.

    Feeding both bridges' terms in as `extra_terms` closes the gap. The
    weed side was already wired (ADR 0008); the brand side (ADR 0007)
    was not.
    """
    terms: list[str] = []
    for weed in weed_matches.values():
        terms.extend(weed.aliases)
    for brand in brand_matches.values():
        terms.extend(brand.active_ingredients)
    return terms
