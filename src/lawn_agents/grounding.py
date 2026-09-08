"""Claim-support validation: does the citation actually back the claim (ADR 0010)?

ADR 0003 established the never-guess guardrail and described three layers:
prompt, schema, tests. In practice the schema layer only ever checked that
a chemical-category `CalendarItem` carries *at least one* `Citation`. It
never checked what that citation points at.

The 2026-09-07 probe found two failures that sailed through it:

1. A doveweed recommendation naming Celsius's active ingredients as
   "thiencarbazone-methyl, foramsulfuron, dicamba". The word
   ``foramsulfuron`` appears in **zero** of the 311 corpus chunks — it was
   invented, inside an item whose citation was the real Celsius label
   (which states the actives are thiencarbazone-methyl and
   iodosulfuron-methyl-sodium).

2. A goosegrass recommendation proposing Celsius chemistry, cited to a
   general Clemson weed factsheet. Celsius's own label never mentions
   goosegrass.

Both had a real, correctly-attributed `Citation`. Presence was never the
problem; *support* was.

This module closes that gap with three deterministic checks — no LLM, no
tokens, so it can run on every synthesis and in CI:

- **Provenance** — the cited `source_id` must belong to a passage that was
  actually in `<sources>`. Catches invented sources.
- **Snippet grounding** — `Citation.snippet` must substantially overlap the
  passage it claims to quote. Enforces synthesizer rule 5 ("cite verbatim
  or near-verbatim"), which until now was prompt-only.
- **Chemical-term grounding** — any brand or active ingredient from the
  bridge vocabulary that the action names must appear in at least one cited
  passage. Catches the fabricated-ingredient case directly.

Scope limit, stated honestly: these checks verify that named chemistry is
*present in the cited source*. They do not verify that the source pairs
that chemistry with the specific pest being asked about. Catching the
Celsius-for-goosegrass class of error properly is a retrieval-quality
problem (the right chunks never reached the synthesizer), not something a
post-hoc text check can settle without heavy false positives. See ADR 0010.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, NamedTuple

from lawn_agents.logging import get_logger
from lawn_agents.models import ChemicalCategory

if TYPE_CHECKING:
    from collections.abc import Iterable

    from lawn_agents.config import GroundingConfig
    from lawn_agents.models import (
        CalendarItem,
        ChemicalsConfig,
        Citation,
        Passage,
        Recommendation,
    )

log = get_logger(__name__)


class GroundingError(NamedTuple):
    """One failed claim-support check, phrased for a re-prompt."""

    kind: str
    """Machine-readable check name: provenance | snippet | chemical_term."""

    detail: str
    """Human/model-readable description of what failed and why."""


def verify(
    recommendation: Recommendation,
    passages: list[Passage],
    chemicals: ChemicalsConfig,
    config: GroundingConfig,
) -> list[GroundingError]:
    """Check every chemical-category item's citations against the sources.

    Args:
        recommendation: The synthesizer's draft (already schema-valid).
        passages: The passages that were actually placed in `<sources>`.
        chemicals: Brand bridge, used as the chemical vocabulary.
        config: Thresholds and enable flag.

    Returns:
        A list of failures, empty when the recommendation is grounded. A
        refusal is always considered grounded — there is no claim to
        support.
    """
    if not config.enabled or recommendation.refused:
        return []

    by_source = _passages_by_source(passages)
    vocabulary = _chemical_vocabulary(chemicals)
    errors: list[GroundingError] = []

    for item in (*recommendation.weekly_actions, *recommendation.monthly_actions):
        if not isinstance(item.category, ChemicalCategory):
            continue
        errors.extend(_verify_item(item, by_source, vocabulary, config))

    if errors:
        log.info(
            "grounding.failed",
            count=len(errors),
            kinds=sorted({e.kind for e in errors}),
        )
    return errors


def format_failures(errors: Iterable[GroundingError]) -> str:
    """Render failures as re-prompt guidance for the synthesizer."""
    lines = [
        "Your previous response failed claim-support validation (ADR 0010).",
        "Every chemical recommendation must be supported by the passage it",
        "cites — not merely accompanied by a citation. Problems found:",
        "",
    ]
    lines.extend(f"- [{e.kind}] {e.detail}" for e in errors)
    lines.extend(
        [
            "",
            "Fix these by quoting the actual <sources> text, naming only",
            "products and active ingredients that appear in the passage you",
            "cite. If no provided passage supports a recommendation, set",
            "refused=true with a concise refusal_reason rather than",
            "substituting a product the sources do not discuss.",
        ]
    )
    return "\n".join(lines)


# --- internals ------------------------------------------------------------


def _verify_item(
    item: CalendarItem,
    by_source: dict[str, str],
    vocabulary: frozenset[str],
    config: GroundingConfig,
) -> list[GroundingError]:
    errors: list[GroundingError] = []
    cited_text_parts: list[str] = []

    for citation in item.citations:
        passage_text = by_source.get(citation.source_id)
        if passage_text is None:
            errors.append(
                GroundingError(
                    kind="provenance",
                    detail=(
                        f"item {item.action[:60]!r} cites source_id="
                        f"{citation.source_id!r}, which was not in <sources>."
                    ),
                )
            )
            continue
        cited_text_parts.append(passage_text)
        errors.extend(_verify_snippet(item, citation, passage_text, config))

    if cited_text_parts:
        errors.extend(_verify_chemical_terms(item, " ".join(cited_text_parts), vocabulary))
    return errors


def _verify_snippet(
    item: CalendarItem,
    citation: Citation,
    passage_text: str,
    config: GroundingConfig,
) -> list[GroundingError]:
    overlap = _overlap_ratio(citation.snippet, passage_text)
    if overlap >= config.snippet_overlap_threshold:
        return []
    return [
        GroundingError(
            kind="snippet",
            detail=(
                f"item {item.action[:60]!r} quotes {citation.snippet[:80]!r} "
                f"from {citation.source_id!r}, but only {overlap:.0%} of that "
                f"text appears in the passage (need "
                f"{config.snippet_overlap_threshold:.0%}). Quote the source."
            ),
        )
    ]


def _verify_chemical_terms(
    item: CalendarItem,
    cited_text: str,
    vocabulary: frozenset[str],
) -> list[GroundingError]:
    """Every chemical name the action uses must appear in a cited passage."""
    action_lower = item.action.lower()
    cited_lower = cited_text.lower()
    missing = sorted(
        term for term in vocabulary if term in action_lower and term not in cited_lower
    )
    return [
        GroundingError(
            kind="chemical_term",
            detail=(
                f"item {item.action[:60]!r} names {term!r}, which does not "
                f"appear in any passage it cites."
            ),
        )
        for term in missing
    ]


def _passages_by_source(passages: list[Passage]) -> dict[str, str]:
    """Merge passage content per source_id — a source may span several chunks."""
    merged: dict[str, str] = {}
    for p in passages:
        merged[p.source_id] = f"{merged.get(p.source_id, '')} {p.content}".strip()
    return merged


def _chemical_vocabulary(chemicals: ChemicalsConfig) -> frozenset[str]:
    """Brand names + active ingredients, lowercased.

    Deliberately excludes the `notes` prose: notes exist to *explain*
    chemistry to the synthesizer and mention neighbouring products, so
    treating them as claim vocabulary would fire on words the model was
    never asserting.
    """
    terms: set[str] = set()
    for name, brand in chemicals.brands.items():
        terms.add(name.lower())
        terms.update(ai.lower() for ai in brand.active_ingredients)
    # Very short tokens ("3336") would match inside unrelated numbers.
    return frozenset(t for t in terms if len(t) >= 5)


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _overlap_ratio(snippet: str, passage_text: str) -> float:
    """Fraction of the snippet's tokens that appear in the passage.

    Token-set containment rather than exact substring matching, because
    synthesizer rule 5 permits "a quote or a tight paraphrase" and PDF
    extraction introduces whitespace and hyphenation noise that would
    break literal matching for legitimate quotes.
    """
    snippet_tokens = set(_TOKEN_RE.findall(snippet.lower()))
    if not snippet_tokens:
        return 0.0
    passage_tokens = set(_TOKEN_RE.findall(passage_text.lower()))
    return len(snippet_tokens & passage_tokens) / len(snippet_tokens)
