"""Forward-looking annual / monthly planner.

Generates a schedule for the configured cultivar + location over the
requested time window. Both `plan_month` and `plan_year` return a
`Recommendation` whose `monthly_actions` carry the scheduled items,
with `earliest` / `latest` dates and conditional gates expressing
when each action belongs.

The never-guess guardrail (ADR 0003) applies — chemical actions still
require citations. Re-prompt + refuse follows the same pattern as the
orchestrator.

The planner shares `_fetch_conditions` and `_safe_retrieve` with the
orchestrator (via private imports) so condition gathering stays in
one place; it has its own planner-specific system prompt at
`prompts/planner.md`.
"""

from __future__ import annotations

import calendar
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import ValidationError

from lawn_agents.agents import drought, knowledge, soiltemp, weather
from lawn_agents.llm import build_chat_model
from lawn_agents.logging import get_logger
from lawn_agents.models import Recommendation
from lawn_agents.orchestrator import _fetch_conditions, _safe_retrieve

if TYPE_CHECKING:
    from collections.abc import Callable

    from lawn_agents.config import AppConfig, Settings
    from lawn_agents.llm import ChatModel
    from lawn_agents.models import (
        Conditions,
        DroughtSnapshot,
        Passage,
        SoilSnapshot,
        WeatherSnapshot,
    )

log = get_logger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"

PlanScope = Literal["month", "year"]


def plan_month(
    year: int,
    month: int,
    settings: Settings,
    *,
    synthesizer: ChatModel | None = None,
    weather_fn: Callable[[AppConfig], WeatherSnapshot | None] | None = None,
    soil_fn: Callable[[AppConfig], SoilSnapshot | None] | None = None,
    drought_fn: Callable[[AppConfig], DroughtSnapshot | None] | None = None,
    retrieve_fn: Callable[[str, AppConfig], list[Passage]] | None = None,
) -> Recommendation:
    """Plan every action that should land within the named month.

    Args:
        year: Target year, e.g. 2026.
        month: Target month (1-12).
        settings: Validated settings.
        synthesizer: Override for the synthesizer `ChatModel` (tests).
        weather_fn: Override for the weather fetcher.
        soil_fn: Override for the soil-temp fetcher.
        drought_fn: Override for the drought fetcher.
        retrieve_fn: Override for the RAG retrieval call.

    Returns:
        A validated `Recommendation` whose `monthly_actions` populate
        the schedule, or a refusal if the never-guess guardrail
        rejected the synthesizer's draft twice in a row.
    """
    if not 1 <= month <= 12:
        msg = f"month must be 1..12, got {month}"
        raise ValueError(msg)
    month_name = calendar.month_name[month]
    target = f"{year}-{month:02d} ({month_name} {year})"
    query = _planner_retrieval_query(
        settings.app, scope="month", target_label=f"{month_name} {year}"
    )
    return _plan(
        scope="month",
        target=target,
        retrieval_query=query,
        settings=settings,
        synthesizer=synthesizer,
        weather_fn=weather_fn,
        soil_fn=soil_fn,
        drought_fn=drought_fn,
        retrieve_fn=retrieve_fn,
    )


def plan_year(
    year: int,
    settings: Settings,
    *,
    synthesizer: ChatModel | None = None,
    weather_fn: Callable[[AppConfig], WeatherSnapshot | None] | None = None,
    soil_fn: Callable[[AppConfig], SoilSnapshot | None] | None = None,
    drought_fn: Callable[[AppConfig], DroughtSnapshot | None] | None = None,
    retrieve_fn: Callable[[str, AppConfig], list[Passage]] | None = None,
) -> Recommendation:
    """Plan a year of actions month-by-month."""
    target = str(year)
    query = _planner_retrieval_query(settings.app, scope="year", target_label=str(year))
    return _plan(
        scope="year",
        target=target,
        retrieval_query=query,
        settings=settings,
        synthesizer=synthesizer,
        weather_fn=weather_fn,
        soil_fn=soil_fn,
        drought_fn=drought_fn,
        retrieve_fn=retrieve_fn,
    )


# --- internals -------------------------------------------------------------


def _plan(
    *,
    scope: PlanScope,
    target: str,
    retrieval_query: str,
    settings: Settings,
    synthesizer: ChatModel | None,
    weather_fn: Callable[[AppConfig], WeatherSnapshot | None] | None,
    soil_fn: Callable[[AppConfig], SoilSnapshot | None] | None,
    drought_fn: Callable[[AppConfig], DroughtSnapshot | None] | None,
    retrieve_fn: Callable[[str, AppConfig], list[Passage]] | None,
) -> Recommendation:
    """Shared body for `plan_month` and `plan_year`."""
    synth_chat = synthesizer or build_chat_model("synthesizer", settings)
    wfn: Callable[[AppConfig], WeatherSnapshot | None] = weather_fn or weather.snapshot
    sfn: Callable[[AppConfig], SoilSnapshot | None] = soil_fn or soiltemp.snapshot
    dfn: Callable[[AppConfig], DroughtSnapshot | None] = drought_fn or drought.snapshot
    rfn: Callable[[str, AppConfig], list[Passage]] = retrieve_fn or knowledge.retrieve

    log.info("planner.start", scope=scope, target=target, query=retrieval_query)

    conditions = _fetch_conditions(settings.app, wfn, sfn, dfn)
    passages = _safe_retrieve(retrieval_query, settings.app, rfn)

    return _synthesize_plan_with_guardrail(
        scope=scope,
        target=target,
        conditions=conditions,
        passages=passages,
        synthesizer=synth_chat,
    )


def _planner_retrieval_query(config: AppConfig, *, scope: PlanScope, target_label: str) -> str:
    """Build a retrieval query covering the cultivar + climate + target."""
    cultivar = config.subject.cultivar
    state = config.location.state
    coastal = "coastal" if config.location.coastal else "inland"
    when = f"in {target_label}" if scope == "month" else f"throughout {target_label}"
    return (
        f"{cultivar} lawn care {when} for {coastal} {state}: "
        "fertilizer, pre-emergent, post-emergent, insecticide, fungicide, "
        "mowing, irrigation, and micronutrient timing."
    )


def _synthesize_plan_with_guardrail(
    *,
    scope: PlanScope,
    target: str,
    conditions: Conditions,
    passages: list[Passage],
    synthesizer: ChatModel,
) -> Recommendation:
    system = _load_prompt("planner.md")
    user_prompt = _planner_user_prompt(scope, target, conditions, passages)

    try:
        return synthesizer.complete_structured(
            system=system, user=user_prompt, response_model=Recommendation
        )
    except ValidationError as exc:
        log.info("planner.synthesizer_validation_failed", error=str(exc))
    except Exception as exc:
        log.warning("planner.synthesizer_call_failed", error=str(exc))
        return _refusal(f"planner synthesis call failed: {type(exc).__name__}: {exc}")

    retry_user = (
        f"{user_prompt}\n\n---\n\n"
        "Your previous response failed schema validation. Return JSON that "
        "satisfies the Recommendation schema. Chemical-category CalendarItems "
        "(fertilizer, micronutrient, herbicide, insecticide, fungicide) require "
        "at least one Citation grounded in <sources>. If you cannot ground a "
        "chemical recommendation, set refused=true and refusal_reason."
    )
    try:
        return synthesizer.complete_structured(
            system=system, user=retry_user, response_model=Recommendation
        )
    except (ValidationError, Exception) as exc:
        log.warning("planner.synthesizer_final_failure", error=str(exc))
        return _refusal(
            "planner output failed schema validation twice; refusing rather than "
            "fabricating a recommendation"
        )


def _planner_user_prompt(
    scope: PlanScope,
    target: str,
    conditions: Conditions,
    passages: list[Passage],
) -> str:
    now = datetime.now(UTC).date().isoformat()
    return (
        f"<today>{now}</today>\n\n"
        f'<plan_target scope="{scope}">{target}</plan_target>\n\n'
        f"<conditions>\n{conditions.model_dump_json(indent=2)}\n</conditions>\n\n"
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


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def _refusal(reason: str) -> Recommendation:
    return Recommendation(
        headline="Unable to produce a plan.",
        conditions_summary="",
        refused=True,
        refusal_reason=reason,
    )
