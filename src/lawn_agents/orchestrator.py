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

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import ValidationError

from lawn_agents.agents import drought, knowledge, research, soiltemp, weather
from lawn_agents.llm import build_chat_model, classify_llm_error, parse_router_intent
from lawn_agents.logging import get_logger
from lawn_agents.models import (
    ChemicalBrand,
    ChemicalsConfig,
    Conditions,
    Recommendation,
    WeedAlias,
    WeedsConfig,
)

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
    weed_matches = detect_weeds_in_question(question, settings.weeds)
    passages = _retrieve_with_weed_aliases(question, weed_matches, settings.app, rfn)

    if settings.app.research.enabled and knowledge.is_weak(passages, settings.app):
        log.info("orchestrator.retrieval_weak.invoking_research")
        researched = _safe_research(
            expand_query_with_weed_aliases(question, weed_matches),
            settings.app,
            research_call,
        )
        if researched:
            passages = researched

    return _synthesize_with_guardrail(
        question=question,
        intent=intent,
        conditions=conditions,
        passages=passages,
        chemicals=settings.chemicals,
        weeds=settings.weeds,
        weed_matches=weed_matches,
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


RRF_K = 60  # Standard RRF constant from the original Cormack et al. 2009 paper.


def _retrieve_with_weed_aliases(
    question: str,
    matched: dict[str, WeedAlias],
    config: AppConfig,
    retrieve_fn: Callable[[str, AppConfig], list[Passage]],
) -> list[Passage]:
    """Multi-query retrieval with Reciprocal Rank Fusion.

    Concatenating a long question with rare technical terms dilutes
    the embedding — the prose dominates and the label drops out of
    top-k. Probed on the 2026-06-09 corpus: "Lespedeza striata"
    alone surfaces the Bayer Celsius label at score 0.601 (top hit),
    but "I have Japanese clover ... Lespedeza striata Kummerowia
    striata" drops the label out of top-8.

    Score-based max-merge ALSO doesn't work, because BGE scores aren't
    comparable across queries: 0.694 on the prosey question vs 0.601
    on an alias-only query doesn't mean the prosey result is more
    relevant — they're scored on different queries.

    Reciprocal Rank Fusion (RRF) is the canonical solution: each
    chunk's fused score is `sum(1 / (RRF_K + rank))` across queries
    that returned it. Rank-based, so scale-invariant. Chunks that
    appear in multiple queries get a boost; chunks that appear in
    only one query at a high rank still surface. We then sort by the
    fused score and return top-rerank_top_k. The synthesizer still
    sees the original question via `<question>`; only retrieval is
    widened.

    When `matched` is empty, this is a single-query path equivalent
    to `_safe_retrieve` — zero behavior change for non-weed questions.
    """
    queries: list[str] = [question]
    for weed in matched.values():
        queries.append(" ".join(weed.aliases))

    # If no aliases, skip RRF and return the raw retrieval — preserves
    # exact behavior for non-weed questions.
    if len(queries) == 1:
        return _safe_retrieve(question, config, retrieve_fn)

    # Dedupe by (source_id, content) — chunks are content-addressed at
    # ingest time, so identical content means the same chunk.
    fused_scores: dict[tuple[str, str], float] = {}
    passage_by_key: dict[tuple[str, str], Passage] = {}
    for q in queries:
        for rank, p in enumerate(_safe_retrieve(q, config, retrieve_fn), start=1):
            key = (p.source_id, p.content)
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            # Keep the passage instance with the highest raw score, so
            # downstream consumers see a meaningful per-chunk score.
            existing = passage_by_key.get(key)
            if existing is None or p.score > existing.score:
                passage_by_key[key] = p

    ranked_keys = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)
    top_k = config.knowledge.retrieval.rerank_top_k
    return [passage_by_key[k] for k in ranked_keys[:top_k]]


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
    chemicals: ChemicalsConfig,
    weeds: WeedsConfig,
    weed_matches: dict[str, WeedAlias] | None = None,
    synthesizer: ChatModel,
) -> Recommendation:
    system = _load_prompt("synthesizer.md")
    brand_bridge = _brand_bridge_text(detect_brands_in_question(question, chemicals))
    # Caller may have already detected weeds (for retrieval-query expansion);
    # re-use that work to avoid a second regex scan.
    weed_matches = (
        weed_matches if weed_matches is not None else detect_weeds_in_question(question, weeds)
    )
    weed_bridge = _weed_bridge_text(weed_matches)
    user_prompt = _synthesizer_user_prompt(
        question, intent, conditions, passages, brand_bridge, weed_bridge
    )

    try:
        return synthesizer.complete_structured(
            system=system, user=user_prompt, response_model=Recommendation
        )
    except ValidationError as exc:
        log.info("orchestrator.synthesizer_validation_failed", error=str(exc))
    except Exception as exc:
        event_suffix, reason = classify_llm_error(exc)
        log.warning(
            f"orchestrator.synthesizer_{event_suffix}",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return _refusal(reason)

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
    except ValidationError as exc:
        log.warning("orchestrator.synthesizer_final_validation_failure", error=str(exc))
        return _refusal(
            "synthesizer output failed schema validation twice; refusing "
            "rather than fabricating a recommendation"
        )
    except Exception as exc:
        # Reached retry from a validation failure but the retry itself
        # hit an SDK-level error (auth, rate-limit, server). Classify
        # so structlog + user message reflect the *actual* terminal
        # failure mode, not "validation twice."
        event_suffix, reason = classify_llm_error(exc)
        log.warning(
            f"orchestrator.synthesizer_retry_{event_suffix}",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return _refusal(reason)


def _synthesizer_user_prompt(
    question: str,
    intent: Intent,
    conditions: Conditions,
    passages: list[Passage],
    brand_bridge: str = "",
    weed_bridge: str = "",
) -> str:
    bridges = "\n\n".join(b for b in (brand_bridge, weed_bridge) if b)
    bridge_block = f"\n\n{bridges}" if bridges else ""
    return (
        f"<intent>{intent}</intent>\n\n"
        f"<conditions>\n{conditions.model_dump_json(indent=2)}\n</conditions>\n\n"
        f"<question>{question}</question>{bridge_block}\n\n"
        f"<sources>\n{_format_sources(passages)}\n</sources>"
    )


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


def _brand_bridge_text(matched: dict[str, ChemicalBrand]) -> str:
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


def _weed_bridge_text(matched: dict[str, WeedAlias]) -> str:
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
