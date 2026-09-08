"""Claim-support validation (ADR 0010).

Each test here corresponds to a failure observed against the live corpus on
2026-09-07, where a recommendation carried a real citation that did not
support the claim it was attached to.
"""

from __future__ import annotations

import pytest

from lawn_agents import grounding
from lawn_agents.config import GroundingConfig
from lawn_agents.models import (
    CalendarItem,
    ChemicalBrand,
    ChemicalCategory,
    ChemicalsConfig,
    Citation,
    GeneralCategory,
    Passage,
    Recommendation,
)

CELSIUS_LABEL_TEXT = (
    "MODE OF ACTION. Two of the three active ingredients in Celsius WG "
    "Herbicide (thiencarbazone-methyl and iodosulfuron-methyl-sodium) "
    "inhibit acetolactate synthase (ALS). Apply at 0.113 oz per 1000 sq ft "
    "for annual lespedeza control."
)


@pytest.fixture
def chemicals() -> ChemicalsConfig:
    return ChemicalsConfig(
        brands={
            "Celsius": ChemicalBrand(
                active_ingredients=[
                    "thiencarbazone-methyl",
                    "iodosulfuron-methyl-sodium",
                    "dicamba",
                ],
                category=ChemicalCategory.HERBICIDE,
            ),
            "Recognition": ChemicalBrand(
                active_ingredients=["trifloxysulfuron-sodium"],
                category=ChemicalCategory.HERBICIDE,
            ),
        }
    )


@pytest.fixture
def config() -> GroundingConfig:
    return GroundingConfig()


def _passage(*, source_id: str = "celsius-label", content: str = CELSIUS_LABEL_TEXT) -> Passage:
    return Passage(
        content=content,
        score=0.8,
        source_id=source_id,
        source_title="Celsius WG Label",
        url="https://example.test/celsius-label.pdf",
    )


def _rec(item: CalendarItem) -> Recommendation:
    return Recommendation(headline="x", conditions_summary="", weekly_actions=[item])


def _citation(*, source_id: str = "celsius-label", snippet: str) -> Citation:
    return Citation(
        source_id=source_id,
        source_title="Celsius WG Label",
        url="https://example.test/celsius-label.pdf",
        snippet=snippet,
    )


class TestGroundedOutputPasses:
    def test_faithful_quote_and_named_chemistry(
        self, chemicals: ChemicalsConfig, config: GroundingConfig
    ) -> None:
        item = CalendarItem(
            category=ChemicalCategory.HERBICIDE,
            action="Apply Celsius WG containing thiencarbazone-methyl at 0.113 oz per 1000 sq ft.",
            citations=[
                _citation(snippet="Apply at 0.113 oz per 1000 sq ft for annual lespedeza control.")
            ],
        )
        assert grounding.verify(_rec(item), [_passage()], chemicals, config) == []

    def test_non_chemical_items_are_not_checked(
        self, chemicals: ChemicalsConfig, config: GroundingConfig
    ) -> None:
        item = CalendarItem(category=GeneralCategory.MOWING, action="Mow at 1.5 inches")
        assert grounding.verify(_rec(item), [_passage()], chemicals, config) == []

    def test_refusal_is_always_grounded(
        self, chemicals: ChemicalsConfig, config: GroundingConfig
    ) -> None:
        rec = Recommendation(
            headline="x", conditions_summary="", refused=True, refusal_reason="no sources"
        )
        assert grounding.verify(rec, [], chemicals, config) == []

    def test_disabled_config_skips_all_checks(self, chemicals: ChemicalsConfig) -> None:
        item = CalendarItem(
            category=ChemicalCategory.HERBICIDE,
            action="Apply foramsulfuron immediately.",
            citations=[_citation(source_id="nonexistent", snippet="totally invented")],
        )
        off = GroundingConfig(enabled=False)
        assert grounding.verify(_rec(item), [_passage()], chemicals, off) == []


class TestFabricatedIngredient:
    """The doveweed failure: an active ingredient in no source anywhere.

    Observed 2026-09-07 — the model reported Celsius's actives as
    "thiencarbazone-methyl, foramsulfuron, dicamba". `foramsulfuron`
    appeared in 0 of 311 corpus chunks.
    """

    def test_ingredient_absent_from_cited_passage_is_flagged(
        self, chemicals: ChemicalsConfig, config: GroundingConfig
    ) -> None:
        chemicals = ChemicalsConfig(
            brands={
                **chemicals.brands,
                "Revolver": ChemicalBrand(
                    active_ingredients=["foramsulfuron"],
                    category=ChemicalCategory.HERBICIDE,
                ),
            }
        )
        item = CalendarItem(
            category=ChemicalCategory.HERBICIDE,
            action=(
                "Apply Celsius WG, active ingredients thiencarbazone-methyl, "
                "foramsulfuron and dicamba."
            ),
            citations=[
                _citation(snippet="Apply at 0.113 oz per 1000 sq ft for annual lespedeza control.")
            ],
        )
        errors = grounding.verify(_rec(item), [_passage()], chemicals, config)
        flagged = {e.detail.split("names ")[1].split(",")[0] for e in errors}
        assert all(e.kind == "chemical_term" for e in errors)
        assert "'foramsulfuron'" in flagged
        # `dicamba` is flagged too, and correctly so: the action asserts it
        # as an active ingredient while the cited chunk names only the two
        # ALS actives. The check is deliberately strict — if the passage
        # you cite doesn't say it, don't claim it.
        assert "'dicamba'" in flagged

    def test_product_not_discussed_by_the_cited_source_is_flagged(
        self, chemicals: ChemicalsConfig, config: GroundingConfig
    ) -> None:
        """Naming Recognition while citing a passage that never mentions it."""
        item = CalendarItem(
            category=ChemicalCategory.HERBICIDE,
            action="Tank mix Recognition for goosegrass control.",
            citations=[
                _citation(snippet="Apply at 0.113 oz per 1000 sq ft for annual lespedeza control.")
            ],
        )
        errors = grounding.verify(_rec(item), [_passage()], chemicals, config)
        assert any(e.kind == "chemical_term" and "recognition" in e.detail for e in errors)


class TestProvenance:
    def test_citation_to_a_source_not_in_sources_is_flagged(
        self, chemicals: ChemicalsConfig, config: GroundingConfig
    ) -> None:
        item = CalendarItem(
            category=ChemicalCategory.HERBICIDE,
            action="Apply something.",
            citations=[_citation(source_id="invented-source", snippet="anything")],
        )
        errors = grounding.verify(_rec(item), [_passage()], chemicals, config)
        assert [e.kind for e in errors] == ["provenance"]

    def test_empty_sources_flags_every_chemical_item(
        self, chemicals: ChemicalsConfig, config: GroundingConfig
    ) -> None:
        item = CalendarItem(
            category=ChemicalCategory.HERBICIDE,
            action="Apply something.",
            citations=[_citation(snippet="anything")],
        )
        errors = grounding.verify(_rec(item), [], chemicals, config)
        assert [e.kind for e in errors] == ["provenance"]


class TestSnippetGrounding:
    def test_invented_quote_is_flagged(
        self, chemicals: ChemicalsConfig, config: GroundingConfig
    ) -> None:
        item = CalendarItem(
            category=ChemicalCategory.HERBICIDE,
            action="Apply per label.",
            citations=[
                _citation(
                    snippet=(
                        "Broadcast four gallons of kerosene across the entire "
                        "putting surface before dawn."
                    )
                )
            ],
        )
        errors = grounding.verify(_rec(item), [_passage()], chemicals, config)
        assert any(e.kind == "snippet" for e in errors)

    def test_tight_paraphrase_is_accepted(
        self, chemicals: ChemicalsConfig, config: GroundingConfig
    ) -> None:
        """Rule 5 permits paraphrase, so this is token overlap, not equality."""
        item = CalendarItem(
            category=ChemicalCategory.HERBICIDE,
            action="Apply per label.",
            citations=[_citation(snippet="Apply 0.113 oz per 1000 sq ft for lespedeza.")],
        )
        errors = grounding.verify(_rec(item), [_passage()], chemicals, config)
        assert [e for e in errors if e.kind == "snippet"] == []

    def test_threshold_is_configurable(self, chemicals: ChemicalsConfig) -> None:
        """At 1.0 every snippet token must be present, so paraphrase fails."""
        item = CalendarItem(
            category=ChemicalCategory.HERBICIDE,
            action="Apply per label.",
            # "spring" appears nowhere in the passage.
            citations=[_citation(snippet="Apply 0.113 oz per 1000 sq ft in spring.")],
        )
        strict = GroundingConfig(snippet_overlap_threshold=1.0)
        assert any(
            e.kind == "snippet"
            for e in grounding.verify(_rec(item), [_passage()], chemicals, strict)
        )


class TestMultiChunkSources:
    def test_content_is_merged_across_chunks_of_one_source(
        self, chemicals: ChemicalsConfig, config: GroundingConfig
    ) -> None:
        """A source spans several chunks; a quote may come from any of them."""
        chunks = [
            _passage(content="MODE OF ACTION. Two of the three active ingredients"),
            _passage(content="in Celsius WG Herbicide inhibit acetolactate synthase (ALS)."),
        ]
        item = CalendarItem(
            category=ChemicalCategory.HERBICIDE,
            action="Apply Celsius WG.",
            citations=[_citation(snippet="Celsius WG Herbicide inhibit acetolactate synthase")],
        )
        assert grounding.verify(_rec(item), chunks, chemicals, config) == []


class TestFormatFailures:
    def test_renders_actionable_reprompt_text(self) -> None:
        errors = [
            grounding.GroundingError(kind="chemical_term", detail="names 'foramsulfuron'"),
            grounding.GroundingError(kind="provenance", detail="cites missing source"),
        ]
        text = grounding.format_failures(errors)
        assert "claim-support validation" in text
        assert "foramsulfuron" in text
        assert "refused=true" in text
