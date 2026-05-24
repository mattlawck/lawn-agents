"""Tests for the never-guess guardrail and citation plumbing."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from lawn_agents.models import (
    CalendarItem,
    ChemicalCategory,
    Citation,
    GeneralCategory,
    Passage,
)


def _citation() -> Citation:
    return Citation(
        source_id="hgic-1207",
        source_title="Clemson HGIC 1207 — Zoysiagrass Maintenance Calendar",
        url="https://hgic.clemson.edu/factsheet/zoysiagrass-maintenance-calendar/",
        page=3,
        snippet="Apply fertilizer at 1 lb N per 1000 sq ft after green-up.",
    )


class TestNeverGuessGuardrail:
    """ADR 0003 — chemical-category items must have a citation."""

    @pytest.mark.parametrize("category", list(ChemicalCategory))
    def test_chemical_category_without_citation_is_rejected(
        self, category: ChemicalCategory
    ) -> None:
        with pytest.raises(ValidationError, match="never-guess guardrail"):
            CalendarItem(category=category, action="apply something")

    @pytest.mark.parametrize("category", list(ChemicalCategory))
    def test_chemical_category_with_citation_is_accepted(self, category: ChemicalCategory) -> None:
        item = CalendarItem(
            category=category,
            action="apply per cited source",
            citations=[_citation()],
        )
        assert item.citations[0].source_id == "hgic-1207"

    @pytest.mark.parametrize("category", list(GeneralCategory))
    def test_general_category_does_not_require_citation(self, category: GeneralCategory) -> None:
        item = CalendarItem(category=category, action="mow at 1.5 inches")
        assert item.citations == []


class TestPassageToCitation:
    """`Passage.to_citation` rolls provenance into a `Citation`."""

    def test_round_trip_basic_fields(self) -> None:
        passage = Passage(
            content="Apply pre-emergent when 4-inch soil temp reaches 55 degrees.",
            score=0.82,
            source_id="hgic-1207",
            source_title="Clemson HGIC 1207",
            url="https://hgic.clemson.edu/factsheet/zoysiagrass-maintenance-calendar/",
            page=4,
        )
        citation = passage.to_citation()
        assert citation.source_id == "hgic-1207"
        assert citation.page == 4
        assert citation.snippet.startswith("Apply pre-emergent")
        assert citation.auto_researched is False

    def test_auto_ingested_unreviewed_passage_flags_citation(self) -> None:
        passage = Passage(
            content="Some new claim found by the research subagent.",
            score=0.6,
            source_id="auto-2026-05-22-1",
            source_title="UGA Extension — turf nitrogen tables",
            auto_ingested=True,
            requires_review=True,
        )
        assert passage.to_citation().auto_researched is True


class TestCitationImmutability:
    """Citations are frozen so they can be safely shared across calendar items."""

    def test_frozen(self) -> None:
        citation = _citation()
        with pytest.raises(ValidationError):
            citation.snippet = "tampered"  # type: ignore[misc]
