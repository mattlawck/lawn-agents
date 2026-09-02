"""Unit tests for the console notify sink."""

from __future__ import annotations

import io
from datetime import date
from pathlib import Path

import pytest

from lawn_agents.config import Settings
from lawn_agents.models import (
    CalendarItem,
    ChemicalCategory,
    Citation,
    GeneralCategory,
    Recommendation,
)
from lawn_agents.notify import ConsoleSink, Sink, _monthly_heading, build_sinks


def _citation(*, title: str = "Clemson HGIC 1207", page: int | None = 3) -> Citation:
    return Citation(
        source_id="hgic-1207",
        source_title=title,
        url="https://hgic.clemson.edu/zoysia",
        page=page,
        snippet="Apply pre-emergent when 4-inch soil temp hits 55F.",
    )


def _render(rec: Recommendation) -> str:
    buf = io.StringIO()
    sink = ConsoleSink(file=buf)
    sink.emit(rec)
    return buf.getvalue()


class TestConsoleSink:
    def test_renders_headline_and_conditions(self) -> None:
        rec = Recommendation(
            headline="Apply pre-emergent this weekend.",
            conditions_summary="4-inch soil temp 56F; no frost in 7d.",
        )
        out = _render(rec)
        assert "Apply pre-emergent this weekend." in out
        assert "4-inch soil temp" in out

    def test_renders_actions_and_numbered_citations(self) -> None:
        cite = _citation()
        rec = Recommendation(
            headline="x",
            conditions_summary="y",
            weekly_actions=[
                CalendarItem(
                    category=ChemicalCategory.HERBICIDE,
                    action="Apply pre-emergent",
                    citations=[cite],
                ),
                CalendarItem(
                    category=GeneralCategory.MOWING,
                    action="Mow at 1.5 inches",
                ),
            ],
        )
        out = _render(rec)
        assert "This week" in out
        assert "pre-emergent" in out.lower()
        assert "Mow at 1.5 inches" in out
        assert "Sources:" in out
        # Inline marker on the herbicide line (chemical category requires
        # a citation; the renderer assigns [1] for the first citation).
        body = out.split("Sources:", 1)[0]
        assert "[1]" in body
        # The Sources block lists the citation by title.
        assert "Clemson HGIC 1207" in out
        # Category renders outside rich markup (em-dash separator).
        assert "herbicide" in out.lower()
        assert "mowing" in out.lower()

    def test_deduplicates_citations_across_items(self) -> None:
        cite = _citation()
        rec = Recommendation(
            headline="x",
            conditions_summary="",
            weekly_actions=[
                CalendarItem(
                    category=ChemicalCategory.HERBICIDE,
                    action="A",
                    citations=[cite],
                ),
                CalendarItem(
                    category=ChemicalCategory.FERTILIZER,
                    action="B",
                    citations=[cite],
                ),
            ],
        )
        out = _render(rec)
        # Same citation across both items should produce exactly one
        # entry in the numbered Sources block.
        sources_block = out.split("Sources:", 1)[1]
        assert sources_block.count("[1]") == 1
        assert "[2]" not in sources_block
        # And the inline markers in the body refer to that same [1].
        body = out.split("Sources:", 1)[0]
        assert body.count("[1]") == 2

    def test_renders_refusal(self) -> None:
        rec = Recommendation(
            headline="Unable to produce a recommendation.",
            conditions_summary="",
            refused=True,
            refusal_reason="No source covers GrubX timing in this corpus.",
        )
        out = _render(rec)
        assert "refused" in out.lower()
        assert "GrubX" in out

    def test_renders_notes(self) -> None:
        rec = Recommendation(
            headline="x",
            conditions_summary="",
            notes=["Soil temp crossed 65F on May 21."],
        )
        out = _render(rec)
        assert "Notes" in out
        assert "Soil temp crossed 65F" in out

    def test_empty_actions_does_not_render_section(self) -> None:
        rec = Recommendation(headline="x", conditions_summary="y")
        out = _render(rec)
        assert "This week" not in out
        assert "This month" not in out

    def test_emit_never_raises_on_minimal_rec(self) -> None:
        rec = Recommendation(headline="x", conditions_summary="")
        sink = ConsoleSink(file=io.StringIO())
        # Should not raise.
        sink.emit(rec)


class TestMonthlyHeading:
    """`monthly_actions` is 'everything not this week', so the heading follows the dates.

    The 2026-09-02 acceptance run asked for a plan peaking on a July
    2027 birthday and printed all nine items under "This month:".
    """

    def test_undated_actions_keep_this_month(self) -> None:
        actions = [CalendarItem(category=GeneralCategory.MOWING, action="Mow at 1.5in")]
        assert _monthly_heading(actions, today=date(2026, 9, 2)) == "This month"

    def test_near_dated_actions_keep_this_month(self) -> None:
        actions = [
            CalendarItem(
                category=GeneralCategory.IRRIGATION,
                action="Water deeply",
                earliest=date(2026, 9, 10),
                latest=date(2026, 9, 20),
            )
        ]
        assert _monthly_heading(actions, today=date(2026, 9, 2)) == "This month"

    def test_far_dated_action_switches_to_planned_schedule(self) -> None:
        actions = [
            CalendarItem(
                category=GeneralCategory.IRRIGATION,
                action="Water deeply",
                earliest=date(2026, 9, 10),
            ),
            CalendarItem(
                category=GeneralCategory.GENERAL,
                action="Spot-spray broadleaf weeds",
                earliest=date(2027, 7, 1),
                latest=date(2027, 7, 10),
            ),
        ]
        assert _monthly_heading(actions, today=date(2026, 9, 2)) == "Planned schedule"

    def test_latest_alone_triggers_the_switch(self) -> None:
        actions = [
            CalendarItem(
                category=GeneralCategory.AERATION,
                action="Core aerate",
                latest=date(2027, 5, 1),
            )
        ]
        assert _monthly_heading(actions, today=date(2026, 9, 2)) == "Planned schedule"

    def test_rendered_output_uses_the_chosen_heading(self) -> None:
        rec = Recommendation(
            headline="Plan to July 2027",
            conditions_summary="",
            monthly_actions=[
                CalendarItem(
                    category=GeneralCategory.MOWING,
                    action="Mow at 1.5in",
                    earliest=date(2027, 6, 1),
                )
            ],
        )
        out = _render(rec)
        assert "Planned schedule" in out
        assert "This month" not in out


class TestBuildSinks:
    @pytest.fixture
    def settings(self, config_yaml_path: Path) -> Settings:
        return Settings.load(config_yaml_path)

    def test_console_is_default(self, settings: Settings) -> None:
        sinks = build_sinks(settings.app)
        assert len(sinks) == 1
        assert isinstance(sinks[0], ConsoleSink)
        # And it satisfies the Protocol.
        sink: Sink = sinks[0]
        assert callable(sink.emit)
