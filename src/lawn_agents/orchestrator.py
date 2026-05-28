"""Orchestrator — routes intent, fans out to agents, synthesizes the answer.

The orchestrator is the only module that sees the full pipeline:

1. Route the user's intent via a `ChatModel` (router role).
2. Fetch conditions (weather, soil temp) — drought lands with the
   annual planner in a follow-up. All fetchers fail-closed so a flaky
   endpoint can't kill the whole run.
3. Retrieve passages from the local RAG.
4. Synthesize the final answer via a `ChatModel` (synthesizer role)
   with `response_model=Recommendation`.
5. Validate the result through the Pydantic guardrail (ADR 0003); on
   validation failure, re-prompt once with the error inline, then
   surface a refusal.

Provider selection (Gemini vs. Anthropic) is decoupled behind the
`ChatModel` Protocol in `lawn_agents.llm`. See ADR 0006.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import ValidationError

from lawn_agents.agents import drought, knowledge, research, soiltemp, weather
from lawn_agents.llm import build_chat_model, parse_router_intent
from lawn_agents.logging import get_logger
from lawn_agents.models import Conditions, Recommendation

if TYPE_CHECKING:
    from collections.abc import Callable

    from lawn_agents.config import AppConfig, Settings
    from lawn_agents.llm import ChatModel
    from lawn_agents.models import DroughtSnapshot, Passage, SoilSnapshot, WeatherSnapshot

log = get_logger(__name__)

Intent = Literal["scheduled-check", "ad-hoc", "plan-month", "plan-year", "out-of-scope"]
_VALID_INTENTS: frozenset[str] = frozenset(
    ("scheduled-check", "ad-hoc", "plan-month", "plan-year", "out-of-scope")
)

PROMPTS_DIR = Path(__file__).parent / "prompts"


def route_intent(
    question: str,
    settings: Settings,
    *,
    router: ChatModel | None = None,
) -> Intent:
    """Classify the user's intent via the router model.

    Args:
        question: The user's natural-language input (or a synthetic
            trigger string for scheduled runs).
        settings: Validated settings (used to build the router model).
        router: Optional override for the router `ChatModel` (tests).

    Returns:
        One of the supported intents. Unparseable router output
        defaults to ``"ad-hoc"`` so we degrade to the most permissive
        downstream behavior.
    """
    chat = router or build_chat_model("router", settings)
    system = _load_prompt("router.md")
    raw = chat.complete_text(system=system, user=question)
    token = parse_router_intent(raw)
    if token in _VALID_INTENTS:
        return _coerce_intent(token)
    log.warning("orchestrator.route_intent_unparseable", raw=raw)
    return "ad-hoc"


def answer(
    question: str,
    settings: Settings,
    *,
    router: ChatModel | None = None,
    synthesizer: ChatModel | None = None,
    weather_fn: Callable[[AppConfig], WeatherSnapshot | None] | None = None,
    soil_fn: Callable[[AppConfig], SoilSnapshot | None] | None = None,
    drought_fn: Callable[[AppConfig], DroughtSnapshot | None] | None = None,
    retrieve_fn: Callable[[str, AppConfig], list[Passage]] | None = None,
    research_fn: Callable[[str, AppConfig], list[Passage]] | None = None,
) -> Recommendation:
    """Run the full ad-hoc-question pipeline.

    Args:
        question: The user's natural-language question.
        settings: Validated settings.
        router: Override for the router `ChatModel` (tests).
        synthesizer: Override for the synthesizer `ChatModel` (tests).
        weather_fn: Override for the weather fetcher (tests).
        soil_fn: Override for the soil-temp fetcher (tests).
        drought_fn: Override for the drought fetcher (tests).
        retrieve_fn: Override for the RAG retrieval call (tests).
        research_fn: Override for the research subagent (tests).
            Defaults to `research.search_and_ingest`. Only called when
            retrieval is weak AND `config.research.enabled` is true.

    Returns:
        A validated `Recommendation`. May be a refusal (`refused=True`)
        if the never-guess guardrail rejected the synthesizer's draft
        twice in a row.
    """
    router_chat = router or build_chat_model("router", settings)
    synthesizer_chat = synthesizer or build_chat_model("synthesizer", settings)
    wfn: Callable[[AppConfig], WeatherSnapshot | None] = weather_fn or weather.snapshot
    sfn: Callable[[AppConfig], SoilSnapshot | None] = soil_fn or soiltemp.snapshot
    dfn: Callable[[AppConfig], DroughtSnapshot | None] = drought_fn or drought.snapshot
    rfn: Callable[[str, AppConfig], list[Passage]] = retrieve_fn or knowledge.retrieve
    research_call: Callable[[str, AppConfig], list[Passage]] = research_fn or _default_research

    intent = route_intent(question, settings, router=router_chat)
    log.info("orchestrator.intent", intent=intent)

    if intent == "out-of-scope":
        return _refusal(
            "Out of scope: this build only advises on the lawn. "
            "Trees, palms, and shrubs are planned for a later phase."
        )

    conditions = _fetch_conditions(settings.app, wfn, sfn, dfn)
    passages = _safe_retrieve(question, settings.app, rfn)

    if settings.app.research.enabled and knowledge.is_weak(passages, settings.app):
        log.info("orchestrator.retrieval_weak.invoking_research")
        researched = _safe_research(question, settings.app, research_call)
        if researched:
            passages = researched

    return _synthesize_with_guardrail(
        question=question,
        intent=intent,
        conditions=conditions,
        passages=passages,
        synthesizer=synthesizer_chat,
    )


def scheduled_check(
    settings: Settings,
    *,
    router: ChatModel | None = None,
    synthesizer: ChatModel | None = None,
    weather_fn: Callable[[AppConfig], WeatherSnapshot | None] | None = None,
    soil_fn: Callable[[AppConfig], SoilSnapshot | None] | None = None,
    drought_fn: Callable[[AppConfig], DroughtSnapshot | None] | None = None,
    retrieve_fn: Callable[[str, AppConfig], list[Passage]] | None = None,
    research_fn: Callable[[str, AppConfig], list[Passage]] | None = None,
) -> Recommendation:
    """Run the weekly scheduled-check workflow.

    For Phase 1, this is `answer(...)` over a canned trigger question.
    The router's `scheduled-check` intent will specialize the synthesis
    prompt in a follow-up once we have more data on which actions
    matter most weekly vs. ad-hoc.
    """
    return answer(
        "Weekly scheduled check: what should I do for my lawn this week "
        "given current conditions and the time of year?",
        settings,
        router=router,
        synthesizer=synthesizer,
        weather_fn=weather_fn,
        soil_fn=soil_fn,
        drought_fn=drought_fn,
        retrieve_fn=retrieve_fn,
        research_fn=research_fn,
    )


# --- internals -------------------------------------------------------------


def _coerce_intent(token: str) -> Intent:
    """Narrow a string we've already validated against `_VALID_INTENTS`."""
    return token  # type: ignore[return-value]


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def _fetch_conditions(
    config: AppConfig,
    weather_fn: Callable[[AppConfig], WeatherSnapshot | None],
    soil_fn: Callable[[AppConfig], SoilSnapshot | None],
    drought_fn: Callable[[AppConfig], DroughtSnapshot | None],
) -> Conditions:
    weather_snap = _safe_call(lambda: weather_fn(config), "weather.snapshot")
    soil_snap = _safe_call(lambda: soil_fn(config), "soiltemp.snapshot")
    drought_snap = _safe_call(lambda: drought_fn(config), "drought.snapshot")
    return Conditions(
        weather=weather_snap,
        soil=soil_snap,
        drought=drought_snap,
        as_of=datetime.now(UTC),
    )


def _safe_call[T](fn: Callable[[], T | None], label: str) -> T | None:
    try:
        return fn()
    except Exception as exc:
        log.warning("orchestrator.fetch_failed", label=label, error=str(exc))
        return None


def _safe_retrieve(
    question: str,
    config: AppConfig,
    retrieve_fn: Callable[[str, AppConfig], list[Passage]],
) -> list[Passage]:
    try:
        return retrieve_fn(question, config)
    except Exception as exc:
        log.warning("orchestrator.retrieve_failed", error=str(exc))
        return []


def _safe_research(
    question: str,
    config: AppConfig,
    research_call: Callable[[str, AppConfig], list[Passage]],
) -> list[Passage]:
    try:
        return research_call(question, config)
    except Exception as exc:
        log.warning("orchestrator.research_failed", error=str(exc))
        return []


def _default_research(question: str, config: AppConfig) -> list[Passage]:
    """Thin wrapper so `research.search_and_ingest` matches the injectable signature."""
    return research.search_and_ingest(question, config)


def _synthesize_with_guardrail(
    *,
    question: str,
    intent: Intent,
    conditions: Conditions,
    passages: list[Passage],
    synthesizer: ChatModel,
) -> Recommendation:
    system = _load_prompt("synthesizer.md")
    user_prompt = _synthesizer_user_prompt(question, intent, conditions, passages)

    try:
        return synthesizer.complete_structured(
            system=system, user=user_prompt, response_model=Recommendation
        )
    except ValidationError as exc:
        log.info("orchestrator.synthesizer_validation_failed", error=str(exc))
    except Exception as exc:
        log.warning("orchestrator.synthesizer_call_failed", error=str(exc))
        return _refusal(f"synthesizer call failed: {type(exc).__name__}: {exc}")

    # One re-prompt with the error inline so the model can self-correct.
    retry_user = (
        f"{user_prompt}\n\n---\n\n"
        "Your previous response failed schema validation. Return JSON that "
        "satisfies the Recommendation schema. Chemical-category CalendarItems "
        "(fertilizer, micronutrient, herbicide, insecticide, fungicide) require "
        "at least one Citation grounded in the provided <sources>. If you cannot "
        "ground a chemical recommendation, set refused=true and refusal_reason."
    )
    try:
        return synthesizer.complete_structured(
            system=system, user=retry_user, response_model=Recommendation
        )
    except Exception as exc:
        # `ValidationError` is a subclass of `Exception` — catching it
        # alongside the parent here is redundant, so we just catch
        # `Exception` and treat any failure as terminal for the guardrail.
        log.warning("orchestrator.synthesizer_final_failure", error=str(exc))
        return _refusal(
            "synthesizer output failed schema validation twice; refusing "
            "rather than fabricating a recommendation"
        )


def _synthesizer_user_prompt(
    question: str,
    intent: Intent,
    conditions: Conditions,
    passages: list[Passage],
) -> str:
    return (
        f"<intent>{intent}</intent>\n\n"
        f"<conditions>\n{conditions.model_dump_json(indent=2)}\n</conditions>\n\n"
        f"<question>{question}</question>\n\n"
        f"<sources>\n{_format_sources(passages)}\n</sources>"
    )


def _format_sources(passages: list[Passage]) -> str:
    if not passages:
        return "(no relevant passages retrieved)"
    parts: list[str] = []
    for i, p in enumerate(passages, start=1):
        review_flag = " [unreviewed]" if p.requires_review else ""
        page_str = f", page {p.page}" if p.page is not None else ""
        url_str = f", url {p.url}" if p.url else ""
        parts.append(
            f"[{i}] source_id={p.source_id!r} title={p.source_title!r}"
            f"{page_str}{url_str}{review_flag} score={p.score:.3f}\n"
            f"    {p.content}"
        )
    return "\n\n".join(parts)


def _refusal(reason: str) -> Recommendation:
    return Recommendation(
        headline="Unable to produce a recommendation.",
        conditions_summary="",
        refused=True,
        refusal_reason=reason,
    )
