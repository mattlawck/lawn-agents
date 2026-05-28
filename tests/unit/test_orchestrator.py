"""Unit tests for the orchestrator (FakeChatModel; no live LLM calls)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from lawn_agents import orchestrator
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

# --- fakes ----------------------------------------------------------------


class FakeChatModel:
    """Configurable stand-in for the ChatModel Protocol.

    `text_response` drives `complete_text` (router).
    `structured_responses` is a queue of objects-or-exceptions returned
    by successive `complete_structured` calls (synthesizer).
    """

    def __init__(
        self,
        *,
        text_response: str = "ad-hoc",
        structured_responses: list[BaseModel | Exception] | None = None,
    ) -> None:
        self.text_response = text_response
        self._structured = list(structured_responses or [])
        self.text_calls: list[tuple[str, str]] = []
        self.structured_calls: list[tuple[str, str]] = []

    def complete_text(self, *, system: str, user: str) -> str:
        self.text_calls.append((system, user))
        return self.text_response

    def complete_structured[T: BaseModel](
        self, *, system: str, user: str, response_model: type[T]
    ) -> T:
        self.structured_calls.append((system, user))
        if not self._structured:
            msg = "FakeChatModel ran out of structured responses"
            raise RuntimeError(msg)
        nxt = self._structured.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        if not isinstance(nxt, response_model):
            msg = (
                f"FakeChatModel canned response type {type(nxt).__name__} "
                f"does not match expected {response_model.__name__}"
            )
            raise TypeError(msg)
        return nxt


def _good_recommendation() -> Recommendation:
    return Recommendation(
        headline="Apply pre-emergent this weekend.",
        conditions_summary="4-inch soil temp 56F; no frost in 7d.",
        weekly_actions=[
            CalendarItem(
                category=ChemicalCategory.HERBICIDE,
                action="Apply pre-emergent",
                citations=[
                    Citation(
                        source_id="hgic-1207",
                        source_title="Clemson HGIC 1207",
                        url="https://hgic.clemson.edu/zoysia",
                        page=3,
                        snippet="Apply pre-emergent when 4-inch soil temp hits 55F.",
                    )
                ],
            ),
            CalendarItem(
                category=GeneralCategory.MOWING,
                action="Mow at 1.5 inches",
            ),
        ],
    )


def _weather_snapshot() -> WeatherSnapshot:
    return WeatherSnapshot(
        fetched_at=datetime(2026, 5, 23, 12, 0, tzinfo=UTC),
        station_id="KCUB",
        current_temp_f=74.3,
        forecast_high_7d_f=[78.0, 82.0],
        forecast_low_7d_f=[62.0, 58.0],
    )


def _soil_snapshot() -> SoilSnapshot:
    return SoilSnapshot(
        fetched_at=datetime(2026, 5, 23, 12, 0, tzinfo=UTC),
        station_id="9999:SC:SCAN",
        current_2in_f=78.0,
        current_4in_f=75.0,
    )


def _drought_snapshot(d_level: int = 3) -> DroughtSnapshot:
    return DroughtSnapshot(
        fetched_at=datetime(2026, 5, 23, 12, 0, tzinfo=UTC),
        county_fips="45019",
        d_level=d_level,
    )


def _passage(*, content: str = "Apply pre-emergent at 55F.") -> Passage:
    return Passage(
        content=content,
        score=0.82,
        source_id="hgic-1207",
        source_title="Clemson HGIC 1207",
        url="https://hgic.clemson.edu/zoysia",
        page=3,
    )


# --- fixtures --------------------------------------------------------------


@pytest.fixture
def settings(config_yaml_path: Path) -> Settings:
    return Settings.load(config_yaml_path)


def _injectables(
    *,
    router: FakeChatModel | None = None,
    synthesizer: FakeChatModel | None = None,
    weather_snap: WeatherSnapshot | None | object = ...,
    soil_snap: SoilSnapshot | None | object = ...,
    drought_snap: DroughtSnapshot | None | object = ...,
    passages: list[Passage] | None = None,
    research_passages: list[Passage] | None = None,
) -> dict[str, Any]:
    weather_value = _weather_snapshot() if weather_snap is ... else weather_snap
    soil_value = _soil_snapshot() if soil_snap is ... else soil_snap
    drought_value = _drought_snapshot() if drought_snap is ... else drought_snap
    return {
        "router": router,
        "synthesizer": synthesizer,
        "weather_fn": lambda _c: weather_value,
        "soil_fn": lambda _c: soil_value,
        "drought_fn": lambda _c: drought_value,
        "retrieve_fn": lambda _q, _c: list(passages or [_passage()]),
        # Default to a no-op research so tests don't accidentally hit
        # the real DDGS search. Tests that want to exercise research
        # pass `research_passages` explicitly.
        "research_fn": lambda _q, _c: list(research_passages or []),
    }


# --- tests ----------------------------------------------------------------


class TestRouteIntent:
    @pytest.mark.parametrize(
        "raw",
        [
            "ad-hoc",
            "  scheduled-check  ",
            '"plan-month"',
            "PLAN-YEAR",
            '{"intent": "out-of-scope"}',
        ],
    )
    def test_valid_intents_are_returned(self, settings: Settings, raw: str) -> None:
        router = FakeChatModel(text_response=raw)
        intent = orchestrator.route_intent("anything", settings, router=router)
        assert intent in orchestrator._VALID_INTENTS

    def test_unparseable_defaults_to_ad_hoc(self, settings: Settings) -> None:
        router = FakeChatModel(text_response="🐶 woof woof 🐶")
        assert orchestrator.route_intent("?", settings, router=router) == "ad-hoc"


class TestAnswerHappyPath:
    def test_returns_synthesizer_recommendation(self, settings: Settings) -> None:
        rec = _good_recommendation()
        router = FakeChatModel(text_response="ad-hoc")
        synth = FakeChatModel(structured_responses=[rec])
        result = orchestrator.answer(
            "Is it time for pre-emergent?",
            settings,
            **_injectables(router=router, synthesizer=synth),
        )
        assert result.refused is False
        assert result.headline == rec.headline
        assert any(
            isinstance(item.category, ChemicalCategory) and item.citations
            for item in result.weekly_actions
        )
        # Router was consulted exactly once; synthesizer exactly once.
        assert len(router.text_calls) == 1
        assert len(synth.structured_calls) == 1

    def test_synthesizer_sees_conditions_and_sources(self, settings: Settings) -> None:
        rec = _good_recommendation()
        router = FakeChatModel(text_response="ad-hoc")
        synth = FakeChatModel(structured_responses=[rec])
        orchestrator.answer(
            "Should I fertilize?",
            settings,
            **_injectables(
                router=router,
                synthesizer=synth,
                passages=[_passage(content="ApplyFertilizerAtThreeQuartersPound.")],
            ),
        )
        _system, user_prompt = synth.structured_calls[0]
        assert "ApplyFertilizerAtThreeQuartersPound" in user_prompt
        assert "current_2in_f" in user_prompt or "current_4in_f" in user_prompt
        assert "Should I fertilize?" in user_prompt


class TestAnswerRefusalPaths:
    def test_out_of_scope_intent_refuses_without_synthesizer_call(self, settings: Settings) -> None:
        router = FakeChatModel(text_response="out-of-scope")
        synth = FakeChatModel(structured_responses=[])  # would error if called
        result = orchestrator.answer(
            "How should I prune my hydrangeas?",
            settings,
            **_injectables(router=router, synthesizer=synth),
        )
        assert result.refused is True
        assert "scope" in (result.refusal_reason or "").lower()
        assert len(synth.structured_calls) == 0

    def test_validation_failure_triggers_one_reprompt(self, settings: Settings) -> None:
        import pydantic_core

        rec = _good_recommendation()
        # First call fails; second succeeds.
        synth = FakeChatModel(
            structured_responses=[
                pydantic_core.ValidationError.from_exception_data(
                    "Recommendation",
                    [
                        {
                            "type": "missing",
                            "loc": ("headline",),
                            "input": {},
                        }
                    ],
                ),
                rec,
            ]
        )
        result = orchestrator.answer(
            "Pre-emergent timing?",
            settings,
            **_injectables(
                router=FakeChatModel(text_response="ad-hoc"),
                synthesizer=synth,
            ),
        )
        assert result.refused is False
        assert len(synth.structured_calls) == 2
        # Re-prompt user content includes the original prompt plus a correction note.
        assert "failed schema validation" in synth.structured_calls[1][1]

    def test_double_validation_failure_returns_refusal(self, settings: Settings) -> None:
        import pydantic_core

        err = pydantic_core.ValidationError.from_exception_data(
            "Recommendation",
            [{"type": "missing", "loc": ("headline",), "input": {}}],
        )
        synth = FakeChatModel(structured_responses=[err, err])
        result = orchestrator.answer(
            "?",
            settings,
            **_injectables(
                router=FakeChatModel(text_response="ad-hoc"),
                synthesizer=synth,
            ),
        )
        assert result.refused is True
        assert (
            "validation" in (result.refusal_reason or "").lower()
            or "schema" in (result.refusal_reason or "").lower()
        )
        assert len(synth.structured_calls) == 2

    def test_synthesizer_exception_returns_refusal(self, settings: Settings) -> None:
        synth = FakeChatModel(structured_responses=[RuntimeError("upstream API 500")])
        result = orchestrator.answer(
            "?",
            settings,
            **_injectables(
                router=FakeChatModel(text_response="ad-hoc"),
                synthesizer=synth,
            ),
        )
        assert result.refused is True
        assert "synthesizer call failed" in (result.refusal_reason or "").lower()


class TestAnswerDegradesOnFetchFailures:
    def test_none_snapshots_still_produce_answer(self, settings: Settings) -> None:
        rec = _good_recommendation()
        result = orchestrator.answer(
            "?",
            settings,
            **_injectables(
                router=FakeChatModel(text_response="ad-hoc"),
                synthesizer=FakeChatModel(structured_responses=[rec]),
                weather_snap=None,
                soil_snap=None,
            ),
        )
        assert result.refused is False

    def test_retrieve_exception_yields_empty_sources(self, settings: Settings) -> None:
        rec = _good_recommendation()
        synth = FakeChatModel(structured_responses=[rec])

        def boom(_q: str, _c: Any) -> list[Passage]:
            raise RuntimeError("index corrupt")

        result = orchestrator.answer(
            "?",
            settings,
            router=FakeChatModel(text_response="ad-hoc"),
            synthesizer=synth,
            weather_fn=lambda _c: _weather_snapshot(),
            soil_fn=lambda _c: _soil_snapshot(),
            drought_fn=lambda _c: _drought_snapshot(),
            retrieve_fn=boom,
            # Empty research result keeps DDGS out of the unit test.
            research_fn=lambda _q, _c: [],
        )
        assert result.refused is False
        _, prompt = synth.structured_calls[0]
        assert "no relevant passages retrieved" in prompt

    def test_drought_snapshot_reaches_prompt(self, settings: Settings) -> None:
        rec = _good_recommendation()
        synth = FakeChatModel(structured_responses=[rec])
        orchestrator.answer(
            "Should I fertilize?",
            settings,
            **_injectables(
                router=FakeChatModel(text_response="ad-hoc"),
                synthesizer=synth,
                drought_snap=_drought_snapshot(d_level=3),
            ),
        )
        _, prompt = synth.structured_calls[0]
        # d_level 3 (extreme drought) should appear in the conditions JSON.
        assert '"d_level": 3' in prompt
        assert "45019" in prompt  # county_fips


class TestScheduledCheck:
    def test_invokes_same_pipeline(self, settings: Settings) -> None:
        rec = _good_recommendation()
        synth = FakeChatModel(structured_responses=[rec])
        result = orchestrator.scheduled_check(
            settings,
            **_injectables(
                router=FakeChatModel(text_response="scheduled-check"),
                synthesizer=synth,
            ),
        )
        assert result.refused is False
        assert "weekly" in synth.structured_calls[0][1].lower()


class TestResearchInvocation:
    """ADR 0005 — research subagent fires only on weak retrieval."""

    def _weak_passage(self) -> Passage:
        # Below the example config's weak_score_threshold (0.55).
        return Passage(
            content="vague-ish content",
            score=0.30,
            source_id="weak",
            source_title="Weak",
        )

    def _researched_passage(self) -> Passage:
        return Passage(
            content="Fresh chunk from web research.",
            score=0.78,
            source_id="auto-2026-05",
            source_title="Auto-researched source",
            url="https://hgic.clemson.edu/x",
            auto_ingested=True,
            requires_review=True,
        )

    def test_research_fires_on_weak_retrieval(self, settings: Settings) -> None:
        rec = _good_recommendation()
        synth = FakeChatModel(structured_responses=[rec])
        research_calls: list[str] = []

        def fake_research(q: str, _c: Any) -> list[Passage]:
            research_calls.append(q)
            return [self._researched_passage()]

        orchestrator.answer(
            "what about grubex timing?",
            settings,
            router=FakeChatModel(text_response="ad-hoc"),
            synthesizer=synth,
            weather_fn=lambda _c: _weather_snapshot(),
            soil_fn=lambda _c: _soil_snapshot(),
            drought_fn=lambda _c: _drought_snapshot(),
            retrieve_fn=lambda _q, _c: [self._weak_passage()],
            research_fn=fake_research,
        )
        assert research_calls == ["what about grubex timing?"]
        _, prompt = synth.structured_calls[0]
        # Researched passage replaced the weak local result.
        assert "Fresh chunk from web research." in prompt
        assert "[unreviewed]" in prompt

    def test_research_skipped_when_retrieval_is_strong(self, settings: Settings) -> None:
        rec = _good_recommendation()
        synth = FakeChatModel(structured_responses=[rec])
        research_calls: list[str] = []

        orchestrator.answer(
            "any question",
            settings,
            **_injectables(
                router=FakeChatModel(text_response="ad-hoc"),
                synthesizer=synth,
                research_passages=[self._researched_passage()],
            ),
        )
        # Default _injectables uses score=0.82 (strong); research_fn is
        # injected but never invoked.
        assert research_calls == []

    def test_research_skipped_when_disabled(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings.app.research, "enabled", False)
        rec = _good_recommendation()
        synth = FakeChatModel(structured_responses=[rec])
        research_calls: list[str] = []

        def fake_research(q: str, _c: Any) -> list[Passage]:
            research_calls.append(q)
            return [self._researched_passage()]

        orchestrator.answer(
            "any",
            settings,
            router=FakeChatModel(text_response="ad-hoc"),
            synthesizer=synth,
            weather_fn=lambda _c: _weather_snapshot(),
            soil_fn=lambda _c: _soil_snapshot(),
            drought_fn=lambda _c: _drought_snapshot(),
            retrieve_fn=lambda _q, _c: [self._weak_passage()],
            research_fn=fake_research,
        )
        assert research_calls == []

    def test_research_exception_is_swallowed(self, settings: Settings) -> None:
        rec = _good_recommendation()
        synth = FakeChatModel(structured_responses=[rec])

        def boom(_q: str, _c: Any) -> list[Passage]:
            raise RuntimeError("ddgs is rate-limited")

        result = orchestrator.answer(
            "?",
            settings,
            router=FakeChatModel(text_response="ad-hoc"),
            synthesizer=synth,
            weather_fn=lambda _c: _weather_snapshot(),
            soil_fn=lambda _c: _soil_snapshot(),
            drought_fn=lambda _c: _drought_snapshot(),
            retrieve_fn=lambda _q, _c: [self._weak_passage()],
            research_fn=boom,
        )
        # Research failure doesn't abort synthesis — we just keep the
        # weak passages and let the model decide.
        assert result.refused is False
