"""Phase-1 acceptance / wire-up tests.

These run the full pipeline end-to-end (router → conditions fetch →
RAG retrieve → research → synthesis → notify render) with mocked
LLMs and pinned data-source snapshots. They check that the pieces
glue together correctly and that the synthesizer's user prompt carries
the context it needs — they do NOT test LLM judgment, which requires
a real model and lives outside the CI run.

Each scenario maps to a real question the user wants the app to
handle (see [[project-acceptance-scenarios]] in repo memory).
"""

from __future__ import annotations

import io
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from lawn_agents import orchestrator, planner
from lawn_agents.config import Settings
from lawn_agents.models import (
    CalendarItem,
    ChemicalCategory,
    Citation,
    DroughtSnapshot,
    GeneralCategory,
    Passage,
    Recommendation,
    SoilSnapshot,
    WeatherSnapshot,
)
from lawn_agents.notify import ConsoleSink
from tests.unit.test_orchestrator import FakeChatModel

# --- pinned snapshots for the drought-2026 scenario -----------------------


def _coastal_sc_late_may_weather() -> WeatherSnapshot:
    """Coastal SC, late May 2026: hot, dry, no frost — drought continues."""
    return WeatherSnapshot(
        fetched_at=datetime(2026, 5, 28, 12, 0, tzinfo=UTC),
        station_id="KCHS",  # Charleston AFB
        current_temp_f=88.0,
        last_24h_precip_in=0.0,
        last_7d_precip_in=0.05,  # essentially nothing
        forecast_high_7d_f=[90.0, 91.0, 89.0, 92.0, 93.0, 91.0, 90.0],
        forecast_low_7d_f=[72.0, 73.0, 71.0, 72.0, 74.0, 73.0, 72.0],
        forecast_precip_chance_7d=[10.0, 20.0, 10.0, 10.0, 30.0, 20.0, 10.0],
        forecast_max_wind_mph_7d=[10.0, 12.0, 8.0, 10.0, 15.0, 12.0, 10.0],
        frost_risk_next_7d=False,
    )


def _coastal_sc_late_may_soil() -> SoilSnapshot:
    """Nearest SCAN station (Youmans Farm ~75 mi); soil dry from drought."""
    return SoilSnapshot(
        fetched_at=datetime(2026, 5, 28, 12, 0, tzinfo=UTC),
        station_id="2038:SC:SCAN",
        current_2in_f=82.0,
        current_4in_f=78.0,
        trailing_7d_4in_f=[76.0, 77.0, 77.0, 78.0, 78.0, 78.0, 78.0],
        current_2in_moisture_pct=8.0,  # very dry
        current_4in_moisture_pct=11.0,
        trailing_7d_4in_moisture_pct=[14.0, 13.5, 12.5, 12.0, 11.5, 11.0, 11.0],
    )


def _charleston_d3_drought() -> DroughtSnapshot:
    """Charleston County (FIPS 45019) at D3 — extreme drought (real, 2026-05)."""
    return DroughtSnapshot(
        fetched_at=datetime(2026, 5, 28, 12, 0, tzinfo=UTC),
        county_fips="45019",
        valid_date=date(2026, 5, 26),
        d_level=3,
    )


# --- pinned RAG passages ---------------------------------------------------


def _zeon_calendar_passage() -> Passage:
    return Passage(
        content=(
            "Zeon Zoysia: Apply 1 lb N per 1000 sqft per month during the "
            "active growing season (May through September). Irrigate the "
            "fertilizer in with 1/4 inch of water within 24 hours."
        ),
        score=0.84,
        source_id="supersod-zeon",
        source_title="Super-Sod Zeon Zoysia Maintenance",
        url="https://info.supersod.com/lawn-care/zeon-zoysia-lawn-maintenance",
    )


def _zeon_drought_advice_passage() -> Passage:
    return Passage(
        content=(
            "Defer high-nitrogen applications during drought; consider half "
            "rates if irrigation is limited. Mow at 2.0 inches (raise from "
            "1.5) to reduce stress."
        ),
        score=0.77,
        source_id="hgic-1207",
        source_title="Clemson HGIC 1207 — Zoysia Maintenance",
        url="https://hgic.clemson.edu/factsheet/zoysiagrass-maintenance-calendar/",
        page=3,
    )


# --- fixtures --------------------------------------------------------------


@pytest.fixture
def settings(config_yaml_path: Path) -> Settings:
    return Settings.load(config_yaml_path)


def _pipeline_injectables(
    *,
    router: FakeChatModel,
    synthesizer: FakeChatModel,
    passages: list[Passage] | None = None,
    weather: WeatherSnapshot | None = None,
    soil: SoilSnapshot | None = None,
    drought: DroughtSnapshot | None = None,
) -> dict[str, Any]:
    return {
        "router": router,
        "synthesizer": synthesizer,
        "weather_fn": lambda _c: weather or _coastal_sc_late_may_weather(),
        "soil_fn": lambda _c: soil or _coastal_sc_late_may_soil(),
        "drought_fn": lambda _c: drought or _charleston_d3_drought(),
        "retrieve_fn": lambda _q, _c: list(passages or []),
        "research_fn": lambda _q, _c: [],  # no-op to keep DDGS out of CI
    }


def _planner_injectables(
    *,
    synthesizer: FakeChatModel,
    passages: list[Passage] | None = None,
) -> dict[str, Any]:
    return {
        "synthesizer": synthesizer,
        "weather_fn": lambda _c: _coastal_sc_late_may_weather(),
        "soil_fn": lambda _c: _coastal_sc_late_may_soil(),
        "drought_fn": lambda _c: _charleston_d3_drought(),
        "retrieve_fn": lambda _q, _c: list(passages or [_zeon_calendar_passage()]),
    }


# --- guardrail end-to-end -------------------------------------------------


class TestNeverGuessGuardrailEndToEnd:
    """ADR 0003 enforced through the full orchestrator pipeline."""

    def test_double_validation_failure_returns_refusal(self, settings: Settings) -> None:
        """If the synthesizer fails schema validation twice, we refuse cleanly."""
        import pydantic_core

        err = pydantic_core.ValidationError.from_exception_data(
            "Recommendation",
            [{"type": "missing", "loc": ("headline",), "input": {}}],
        )
        synth = FakeChatModel(structured_responses=[err, err])

        result = orchestrator.answer(
            "When should I apply pre-emergent?",
            settings,
            **_pipeline_injectables(
                router=FakeChatModel(text_response="ad-hoc"),
                synthesizer=synth,
                passages=[_zeon_calendar_passage()],
            ),
        )
        assert result.refused is True
        # The re-prompt happened (two synth calls), then refusal.
        assert len(synth.structured_calls) == 2

    def test_empty_retrieval_sends_no_sources_to_synthesizer(self, settings: Settings) -> None:
        """An empty corpus must show the synthesizer it has nothing to cite."""
        synth = FakeChatModel(
            structured_responses=[
                Recommendation(
                    headline="No authoritative source found.",
                    conditions_summary="",
                    refused=True,
                    refusal_reason=(
                        "No corpus entry covers this specific product. Consult "
                        "Clemson HGIC or your local extension agent."
                    ),
                )
            ]
        )
        result = orchestrator.answer(
            "Is it too late to treat with GrubX?",
            settings,
            **_pipeline_injectables(
                router=FakeChatModel(text_response="ad-hoc"),
                synthesizer=synth,
                passages=[],  # empty corpus
            ),
        )
        _, prompt = synth.structured_calls[0]
        assert "no relevant passages retrieved" in prompt
        # Synthesizer chose to refuse — guardrail honored at the
        # prompt level even though Pydantic would have accepted a
        # general-category recommendation here.
        assert result.refused is True
        assert "GrubX" in (result.refusal_reason or "") or "extension" in (
            result.refusal_reason or ""
        )


# --- drought-2026 scenario ------------------------------------------------


class TestDrought2026Scenario:
    """Real-world data: Charleston County D3 + dry soil + hot forecast."""

    def test_plan_year_prompt_includes_drought_and_soil_dryness(self, settings: Settings) -> None:
        synth = FakeChatModel(
            structured_responses=[
                Recommendation(
                    headline="Drought-adjusted 2026 plan for Zeon Zoysia.",
                    conditions_summary="Charleston County D3; 4-inch soil 78F, 11% moisture.",
                    monthly_actions=[
                        CalendarItem(
                            category=ChemicalCategory.FERTILIZER,
                            action="Apply 0.5 lb N per 1000 sqft (half rate, drought)",
                            citations=[
                                Citation(
                                    source_id="hgic-1207",
                                    source_title="Clemson HGIC 1207",
                                    snippet=(
                                        "Defer high-N during drought; consider "
                                        "half rates if irrigation is limited."
                                    ),
                                    page=3,
                                )
                            ],
                        ),
                    ],
                )
            ]
        )
        planner.plan_year(
            2026,
            settings,
            **_planner_injectables(
                synthesizer=synth,
                passages=[_zeon_drought_advice_passage(), _zeon_calendar_passage()],
            ),
        )
        _, prompt = synth.structured_calls[0]
        # The synthesizer can see we're in extreme drought.
        assert '"d_level": 3' in prompt
        # The dry-soil signal made it through.
        assert "11" in prompt  # current 4-inch moisture %
        # The drought-specific source passage reached the prompt.
        assert "Defer high-N" in prompt or "Defer high-nitrogen" in prompt

    def test_plan_year_returns_recommendation_with_drought_aware_actions(
        self, settings: Settings
    ) -> None:
        synth = FakeChatModel(
            structured_responses=[
                Recommendation(
                    headline="Drought-adjusted plan.",
                    conditions_summary="D3 — extreme drought.",
                    monthly_actions=[
                        CalendarItem(
                            category=ChemicalCategory.FERTILIZER,
                            action="Half-rate N",
                            conditional="only if irrigation available",
                            citations=[
                                Citation(
                                    source_id="hgic-1207",
                                    source_title="Clemson HGIC 1207",
                                    snippet="consider half rates",
                                )
                            ],
                        ),
                    ],
                )
            ]
        )
        rec = planner.plan_year(
            2026,
            settings,
            **_planner_injectables(synthesizer=synth),
        )
        assert rec.refused is False
        first = rec.monthly_actions[0]
        assert isinstance(first.category, ChemicalCategory)
        assert first.citations  # citation present (guardrail satisfied)
        assert first.conditional is not None  # drought gating expressed


# --- target-date peak planning (user scenario #1) -------------------------


class TestPeakByJuly13Scenario:
    """User: 'My daughter's birthday is July 13. Plan for peak then.'"""

    def test_plan_month_july_prompt_includes_target(self, settings: Settings) -> None:
        synth = FakeChatModel(
            structured_responses=[
                Recommendation(
                    headline="Peak-by-July-13 schedule.",
                    conditions_summary="Coastal SC, D3 drought, planning toward July 13.",
                    monthly_actions=[
                        CalendarItem(
                            category=ChemicalCategory.FERTILIZER,
                            action="Apply 1 lb N early July",
                            earliest=date(2026, 7, 1),
                            latest=date(2026, 7, 5),
                            citations=[
                                Citation(
                                    source_id="supersod-zeon",
                                    source_title="Super-Sod Zeon Maintenance",
                                    snippet="1 lb N per 1000 sqft monthly",
                                )
                            ],
                        ),
                        CalendarItem(
                            category=GeneralCategory.MOWING,
                            action="Mow at 1.5 inches the week of July 7",
                            earliest=date(2026, 7, 7),
                            latest=date(2026, 7, 12),
                        ),
                    ],
                )
            ]
        )
        planner.plan_month(
            2026,
            7,
            settings,
            **_planner_injectables(synthesizer=synth),
        )
        _, prompt = synth.structured_calls[0]
        assert "2026-07" in prompt
        assert "July" in prompt
        assert 'scope="month"' in prompt
        # Drought context still flows into the planner prompt.
        assert '"d_level": 3' in prompt

    def test_recommendation_has_dated_actions(self, settings: Settings) -> None:
        synth = FakeChatModel(
            structured_responses=[
                Recommendation(
                    headline="July plan",
                    conditions_summary="",
                    monthly_actions=[
                        CalendarItem(
                            category=GeneralCategory.MOWING,
                            action="Mow week-of-July-7",
                            earliest=date(2026, 7, 7),
                            latest=date(2026, 7, 12),
                        )
                    ],
                )
            ]
        )
        rec = planner.plan_month(
            2026,
            7,
            settings,
            **_planner_injectables(synthesizer=synth),
        )
        first = rec.monthly_actions[0]
        assert first.earliest == date(2026, 7, 7)
        assert first.latest == date(2026, 7, 12)


# --- GrubX late-treatment Q&A (user scenario #2) -------------------------


class TestGrubXTimingScenario:
    """User: 'Is it too late to treat with GrubX?' — branded product Q&A.

    When the corpus has no GrubX-specific passage, the synthesizer must
    refuse rather than fabricate a product timing recommendation. This
    test wires the refusal path end-to-end and verifies the rendered
    output is clear to the user.
    """

    def test_empty_corpus_refusal_renders_to_console(self, settings: Settings) -> None:
        refusal_reason = (
            "No source in the corpus covers GrubX (imidacloprid) timing for "
            "Zeon Zoysia. Consult Clemson HGIC, your local extension agent, "
            "or the product label."
        )
        synth = FakeChatModel(
            structured_responses=[
                Recommendation(
                    headline="Unable to recommend without a cited source.",
                    conditions_summary="",
                    refused=True,
                    refusal_reason=refusal_reason,
                )
            ]
        )
        result = orchestrator.answer(
            "Is it too late to treat with GrubX?",
            settings,
            **_pipeline_injectables(
                router=FakeChatModel(text_response="ad-hoc"),
                synthesizer=synth,
                passages=[],
            ),
        )
        assert result.refused is True

        # Now render through the console sink and verify a user can see why.
        buf = io.StringIO()
        ConsoleSink(file=buf).emit(result)
        output = buf.getvalue()
        assert "Refused" in output
        assert "GrubX" in output
        assert "Clemson" in output or "extension" in output


# --- end-to-end notify wire-up -------------------------------------------


class TestEndToEndNotifyOutput:
    """Full pipeline + render: orchestrator.answer → ConsoleSink.emit."""

    def test_recommendation_renders_with_citation_footnotes(self, settings: Settings) -> None:
        synth = FakeChatModel(
            structured_responses=[
                Recommendation(
                    headline="Mow this week; defer fertilizer until rain.",
                    conditions_summary="D3 drought; 4-inch soil temp 78F.",
                    weekly_actions=[
                        CalendarItem(
                            category=GeneralCategory.MOWING,
                            action="Mow at 1.5 inches",
                        ),
                        CalendarItem(
                            category=ChemicalCategory.FERTILIZER,
                            action="Wait for irrigation or rain",
                            citations=[
                                Citation(
                                    source_id="hgic-1207",
                                    source_title="Clemson HGIC 1207",
                                    snippet="defer N during drought",
                                    page=3,
                                )
                            ],
                        ),
                    ],
                    notes=["Current drought is D3 per USDM 2026-05-26."],
                )
            ]
        )
        result = orchestrator.answer(
            "What should I do this week?",
            settings,
            **_pipeline_injectables(
                router=FakeChatModel(text_response="ad-hoc"),
                synthesizer=synth,
                passages=[_zeon_drought_advice_passage()],
            ),
        )
        buf = io.StringIO()
        ConsoleSink(file=buf).emit(result)
        output = buf.getvalue()

        # Headline panel.
        assert "Mow this week" in output
        # Conditions line.
        assert "D3 drought" in output
        # Both action sections render.
        assert "Mow at 1.5 inches" in output
        assert "Wait for irrigation or rain" in output
        # Citation marker on the fertilizer (chemical) item; Sources block.
        assert "[1]" in output
        assert "Sources:" in output
        assert "Clemson HGIC 1207" in output
        # Notes block.
        assert "USDM 2026-05-26" in output
