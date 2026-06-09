"""Unit tests for the annual / monthly planner (FakeChatModel; no live LLM)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from lawn_agents import planner
from lawn_agents.config import Settings
from lawn_agents.models import (
    CalendarItem,
    ChemicalBrand,
    ChemicalCategory,
    ChemicalsConfig,
    Citation,
    DroughtSnapshot,
    GeneralCategory,
    Passage,
    Recommendation,
    SoilSnapshot,
    WeatherSnapshot,
)
from tests.unit.test_orchestrator import FakeChatModel

# --- canned objects --------------------------------------------------------


def _good_plan() -> Recommendation:
    return Recommendation(
        headline="Forward plan for July 2026.",
        conditions_summary="D3 drought; soil temp 78F at 4 inch.",
        monthly_actions=[
            CalendarItem(
                category=ChemicalCategory.FERTILIZER,
                action="Apply 1 lb N per 1000 sqft",
                citations=[
                    Citation(
                        source_id="supersod-zeon",
                        source_title="Super-Sod Zeon Maintenance",
                        url="https://info.supersod.com/lawn-care/zeon-zoysia-lawn-maintenance",
                        snippet="Apply 1 lb N per 1000 sqft monthly during growing season.",
                    )
                ],
            ),
            CalendarItem(
                category=GeneralCategory.MOWING,
                action="Mow at 1.5 inches weekly",
            ),
        ],
    )


def _weather_snapshot() -> WeatherSnapshot:
    return WeatherSnapshot(fetched_at=datetime(2026, 5, 26, 12, 0, tzinfo=UTC))


def _soil_snapshot() -> SoilSnapshot:
    return SoilSnapshot(
        fetched_at=datetime(2026, 5, 26, 12, 0, tzinfo=UTC),
        station_id="9999:SC:SCAN",
        current_4in_f=78.0,
    )


def _drought_snapshot() -> DroughtSnapshot:
    return DroughtSnapshot(
        fetched_at=datetime(2026, 5, 26, 12, 0, tzinfo=UTC),
        county_fips="45019",
        d_level=3,
    )


def _passages() -> list[Passage]:
    return [
        Passage(
            content="Apply 1 lb N per 1000 sqft monthly during growing season.",
            score=0.85,
            source_id="supersod-zeon",
            source_title="Super-Sod Zeon Maintenance",
            url="https://info.supersod.com/lawn-care/zeon-zoysia-lawn-maintenance",
        )
    ]


@pytest.fixture
def settings(config_yaml_path: Path) -> Settings:
    return Settings.load(config_yaml_path)


def _injectables(
    *,
    synthesizer: FakeChatModel,
    weather_snap: WeatherSnapshot | None = None,
    soil_snap: SoilSnapshot | None = None,
    drought_snap: DroughtSnapshot | None = None,
    passages: list[Passage] | None = None,
) -> dict[str, Any]:
    return {
        "synthesizer": synthesizer,
        "weather_fn": lambda _c: weather_snap or _weather_snapshot(),
        "soil_fn": lambda _c: soil_snap or _soil_snapshot(),
        "drought_fn": lambda _c: drought_snap or _drought_snapshot(),
        "retrieve_fn": lambda _q, _c: passages or _passages(),
    }


# --- plan_month ------------------------------------------------------------


class TestPlanMonth:
    def test_returns_synthesizer_recommendation(self, settings: Settings) -> None:
        rec = _good_plan()
        synth = FakeChatModel(structured_responses=[rec])
        result = planner.plan_month(2026, 7, settings, **_injectables(synthesizer=synth))
        assert result.refused is False
        assert result.headline == rec.headline
        assert len(synth.structured_calls) == 1

    def test_prompt_includes_target_month_and_year(self, settings: Settings) -> None:
        rec = _good_plan()
        synth = FakeChatModel(structured_responses=[rec])
        planner.plan_month(2026, 7, settings, **_injectables(synthesizer=synth))
        _, user_prompt = synth.structured_calls[0]
        assert "2026-07" in user_prompt
        assert "July" in user_prompt
        assert 'scope="month"' in user_prompt

    def test_prompt_carries_drought_and_sources(self, settings: Settings) -> None:
        rec = _good_plan()
        synth = FakeChatModel(structured_responses=[rec])
        planner.plan_month(2026, 7, settings, **_injectables(synthesizer=synth))
        _, user_prompt = synth.structured_calls[0]
        assert '"d_level": 3' in user_prompt
        # Source content is rendered into the prompt.
        assert "1 lb N per 1000 sqft" in user_prompt
        assert "Super-Sod Zeon Maintenance" in user_prompt

    @pytest.mark.parametrize("bad_month", [0, 13, -1, 99])
    def test_invalid_month_raises(self, settings: Settings, bad_month: int) -> None:
        with pytest.raises(ValueError, match=r"month must be 1\.\.12"):
            planner.plan_month(
                2026,
                bad_month,
                settings,
                **_injectables(synthesizer=FakeChatModel(structured_responses=[])),
            )


# --- plan_year -------------------------------------------------------------


class TestPlanYear:
    def test_returns_synthesizer_recommendation(self, settings: Settings) -> None:
        rec = _good_plan()
        synth = FakeChatModel(structured_responses=[rec])
        result = planner.plan_year(2026, settings, **_injectables(synthesizer=synth))
        assert result.refused is False
        assert result.headline == rec.headline

    def test_prompt_has_year_scope(self, settings: Settings) -> None:
        rec = _good_plan()
        synth = FakeChatModel(structured_responses=[rec])
        planner.plan_year(2026, settings, **_injectables(synthesizer=synth))
        _, user_prompt = synth.structured_calls[0]
        assert 'scope="year"' in user_prompt
        assert ">2026<" in user_prompt


# --- refusal paths --------------------------------------------------------


class TestRefusalPaths:
    def test_validation_failure_triggers_reprompt(self, settings: Settings) -> None:
        import pydantic_core

        rec = _good_plan()
        synth = FakeChatModel(
            structured_responses=[
                pydantic_core.ValidationError.from_exception_data(
                    "Recommendation",
                    [{"type": "missing", "loc": ("headline",), "input": {}}],
                ),
                rec,
            ]
        )
        result = planner.plan_month(2026, 7, settings, **_injectables(synthesizer=synth))
        assert result.refused is False
        assert len(synth.structured_calls) == 2
        assert "failed schema validation" in synth.structured_calls[1][1]

    def test_double_validation_failure_returns_refusal(self, settings: Settings) -> None:
        import pydantic_core

        err = pydantic_core.ValidationError.from_exception_data(
            "Recommendation",
            [{"type": "missing", "loc": ("headline",), "input": {}}],
        )
        synth = FakeChatModel(structured_responses=[err, err])
        result = planner.plan_month(2026, 7, settings, **_injectables(synthesizer=synth))
        assert result.refused is True

    def test_synthesizer_exception_returns_refusal(self, settings: Settings) -> None:
        synth = FakeChatModel(structured_responses=[RuntimeError("upstream API 500")])
        result = planner.plan_month(2026, 7, settings, **_injectables(synthesizer=synth))
        assert result.refused is True
        assert "planner synthesis call failed" in (result.refusal_reason or "").lower()


# --- brand bridge ---------------------------------------------------------


class TestBrandBridgeInjection:
    """A brand name in the planner target injects <brand_bridge> (ADR 0007)."""

    def test_brand_in_target_renders_bridge_block(self, settings: Settings) -> None:
        settings.chemicals = ChemicalsConfig(
            brands={
                "Acelepryn": ChemicalBrand(
                    active_ingredients=["chlorantraniliprole"],
                    category=ChemicalCategory.INSECTICIDE,
                    notes="Syngenta professional product.",
                )
            }
        )
        rec = _good_plan()
        synth = FakeChatModel(structured_responses=[rec])
        planner._synthesize_plan_with_guardrail(
            scope="month",
            target="July 2026 — plan around Acelepryn cycle",
            conditions=planner._fetch_conditions(  # type: ignore[attr-defined]
                settings.app,
                lambda _c: _weather_snapshot(),
                lambda _c: _soil_snapshot(),
                lambda _c: _drought_snapshot(),
            ),
            passages=_passages(),
            chemicals=settings.chemicals,
            synthesizer=synth,
        )
        _, user_prompt = synth.structured_calls[0]
        assert "<brand_bridge>" in user_prompt
        assert "Acelepryn" in user_prompt
        assert "chlorantraniliprole" in user_prompt

    def test_no_brand_in_target_omits_bridge_block(self, settings: Settings) -> None:
        # Default settings.chemicals from config.example.yaml — but even
        # with a populated chemicals config, a target without a brand
        # name produces no bridge.
        settings.chemicals = ChemicalsConfig(
            brands={
                "GrubX": ChemicalBrand(
                    active_ingredients=["chlorantraniliprole"],
                    category=ChemicalCategory.INSECTICIDE,
                )
            }
        )
        rec = _good_plan()
        synth = FakeChatModel(structured_responses=[rec])
        planner.plan_month(2026, 7, settings, **_injectables(synthesizer=synth))
        _, user_prompt = synth.structured_calls[0]
        assert "<brand_bridge>" not in user_prompt


# --- retrieval query ------------------------------------------------------


class TestRetrievalQuery:
    def test_month_query_mentions_target_month_name(self, settings: Settings) -> None:
        q = planner._planner_retrieval_query(settings.app, scope="month", target_label="July 2026")
        assert "July 2026" in q
        assert settings.app.subject.cultivar in q

    def test_year_query_uses_throughout(self, settings: Settings) -> None:
        q = planner._planner_retrieval_query(settings.app, scope="year", target_label="2026")
        assert "throughout 2026" in q
