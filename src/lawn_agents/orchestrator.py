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

from lawn_agents import bridges, synthesis
from lawn_agents.agents import drought, knowledge, research, soiltemp, weather
from lawn_agents.llm import build_chat_model, parse_router_intent
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

    from lawn_agents.config import (
        AppConfig,
        ClimateConfig,
        GroundingConfig,
        Settings,
        SourceTiersConfig,
    )
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

    conditions = fetch_conditions(settings.app, wfn, sfn, dfn)
    weed_matches = bridges.detect_weeds_in_question(question, settings.weeds)
    brand_matches = bridges.detect_brands_in_question(question, settings.chemicals)
    passages = _retrieve_with_bridges(question, weed_matches, brand_matches, settings.app, rfn)

    # Tiered relevance check: lexical-overlap miss in the medium band
    # escalates to a cheap LLM gate via the router model. Both bridges
    # contribute vocabulary — see `bridges.bridge_lexical_terms`.
    if settings.app.research.enabled and knowledge.is_weak(
        passages,
        settings.app,
        query=question,
        extra_terms=bridges.bridge_lexical_terms(weed_matches, brand_matches),
        relevance_gate=_make_relevance_gate(router_chat),
    ):
        log.info("orchestrator.retrieval_weak.invoking_research")
        researched = _safe_research(
            bridges.expand_query_with_weed_aliases(question, weed_matches),
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
        tiers=settings.app.knowledge.source_tiers,
        climate=settings.app.climate,
        grounding_config=settings.app.grounding,
        weed_matches=weed_matches,
        brand_matches=brand_matches,
        synthesizer=synthesizer_chat,
    )


# --- internals -------------------------------------------------------------


def _coerce_intent(token: str) -> Intent:
    """Narrow a string we've already validated against `_VALID_INTENTS`."""
    return token  # type: ignore[return-value]


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def fetch_conditions(
    config: AppConfig,
    weather_fn: Callable[[AppConfig], WeatherSnapshot | None],
    soil_fn: Callable[[AppConfig], SoilSnapshot | None],
    drought_fn: Callable[[AppConfig], DroughtSnapshot | None],
) -> Conditions:
    """Gather every live input, failing closed on each independently.

    Shared with the planner. Each fetcher degrades to `None` with the
    reason logged, so one unreachable endpoint narrows the answer
    instead of killing the run — the SCAN station alone failed three
    times in a single afternoon this month.
    """
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


def safe_retrieve(
    question: str,
    config: AppConfig,
    retrieve_fn: Callable[[str, AppConfig], list[Passage]],
) -> list[Passage]:
    """Retrieve passages, degrading to an empty list on failure.

    Shared with the planner. An empty `<sources>` block is a survivable
    state — the guardrails turn it into a refusal — whereas an exception
    here would lose the conditions already gathered.
    """
    try:
        return retrieve_fn(question, config)
    except Exception as exc:
        log.warning("orchestrator.retrieve_failed", error=str(exc))
        return []


RRF_K = 60  # Standard RRF constant from the original Cormack et al. 2009 paper.
RESERVED_PER_BRIDGE_QUERY = 2
"""Top hits each bridge query is guaranteed to contribute.

Two rather than one because a label's rate table is rarely its
best-matching chunk — the brand name appears throughout the document,
so the top hit is often the front panel or use-restrictions section.
"""


def _retrieve_with_bridges(
    question: str,
    weed_matches: dict[str, WeedAlias],
    brand_matches: dict[str, ChemicalBrand],
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

    Brands are widened the same way, and for the same reason. ADR 0007's
    bridge told the *synthesizer* that "Sedge Ender" means sulfentrazone,
    and (since 2026-09-02) told the relevance check too — but never
    retrieval. So asking about a product by brand never searched for its
    chemistry, and a label whose rate table says "sulfentrazone" stayed
    unreachable behind a question that says "Sedge Ender". Observed
    2026-09-14: the rate chunk was in the corpus and outside the top-5,
    and the system refused an answer it had the evidence for.

    KNOWN LIMIT — this does not reach a specific product's rate table.
    Probed 2026-09-14 against the live corpus: the query "Sedge Ender
    sulfentrazone application rate per 1000 sq ft Zoysia matrella"
    returned zero Sedge Ender chunks in its top eight, and instead
    returned the rate tables of Fusilade II, Recognition and Celsius.
    BGE-small embeds "application rate per 1000 sq ft" as the dominant
    signal and every label matches it; a brand name is a rare token
    carrying almost no semantic weight, so it gets washed out.

    No amount of query phrasing fixes that — it is what dense retrieval
    is bad at. Exact-term matching (BM25, fused with the vector scores)
    is the right tool and is not yet implemented, which is why the
    system still refuses rate questions about products whose labels are
    sitting in the corpus. Tracked as the open half of ADR 0010.

    When nothing matches, this is a single-query path equivalent to
    `safe_retrieve` — zero behavior change for unbridged questions.
    """
    queries: list[str] = [question]
    for weed in weed_matches.values():
        queries.append(" ".join(weed.aliases))
    for name, brand in brand_matches.items():
        actives = " ".join(brand.active_ingredients)
        # Brand name alongside its chemistry: labels lead with the brand,
        # extension publications lead with the active ingredient.
        queries.append(f"{name} {actives}")
        # A rate-seeking variant ("application rate per 1000 sq ft
        # <species>") was tried and dropped: under vector search it
        # embeds closer to EVERY label's rate table than to any one of
        # them, so it retrieved other products' numbers. Exact-term
        # matching solves this properly — see ADR 0011.

    # If no aliases, skip RRF and return the raw retrieval — preserves
    # exact behavior for non-weed questions.
    if len(queries) == 1:
        return safe_retrieve(question, config, retrieve_fn)

    # Dedupe by (source_id, content) — chunks are content-addressed at
    # ingest time, so identical content means the same chunk.
    fused_scores: dict[tuple[str, str], float] = {}
    passage_by_key: dict[tuple[str, str], Passage] = {}
    per_query: list[list[tuple[str, str]]] = []
    for q in queries:
        ranked_for_query: list[tuple[str, str]] = []
        for rank, p in enumerate(safe_retrieve(q, config, retrieve_fn), start=1):
            key = (p.source_id, p.content)
            ranked_for_query.append(key)
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            # Keep the passage instance with the highest raw score, so
            # downstream consumers see a meaningful per-chunk score.
            existing = passage_by_key.get(key)
            if existing is None or p.score > existing.score:
                passage_by_key[key] = p
        per_query.append(ranked_for_query)

    # Guarantee each bridge query contributes its best hits before fusing.
    #
    # RRF rewards consensus across queries — right for general relevance,
    # wrong for a question naming two products. A label chunk scores on
    # the single query that names its product and loses to chunks that
    # score moderately on four, so on 2026-09-14 a question naming Sedge
    # Ender returned no Sedge Ender passages at all.
    #
    # This only became worth doing once full-text search (ADR 0011) made
    # each brand query's top hits reliably the *right* product. Reserving
    # slots under vector-only retrieval just reserved the wrong document.
    reserved: list[tuple[str, str]] = []
    for ranked_for_query in per_query[1:]:  # skip the raw question
        for key in ranked_for_query[:RESERVED_PER_BRIDGE_QUERY]:
            if key not in reserved:
                reserved.append(key)

    # Budget is computed AFTER reserving, not before: reserving six slots
    # into a five-slot result silently drops the last bridge query's
    # hits, which is the same bug in a new place. `rerank_top_k` is a
    # floor for the unbridged case, not a ceiling once the question
    # names several things.
    top_k = max(config.knowledge.retrieval.rerank_top_k, len(reserved) + 2)
    ranked_keys = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)
    ordered = reserved + [k for k in ranked_keys if k not in reserved]
    return [passage_by_key[k] for k in ordered[:top_k]]


_RELEVANCE_GATE_SYSTEM = (
    "You are a relevance gate. Given a USER QUESTION and a RETRIEVED PASSAGE, "
    "output exactly one word: 'yes' if the passage contains information that "
    "directly helps answer the question, or 'no' otherwise. No explanation, "
    "no punctuation, just the single word."
)


def _make_relevance_gate(
    chat_model: ChatModel,
) -> Callable[[str, Passage], bool]:
    """Build a `(query, passage) -> bool` callable for `is_weak`.

    Runs the cheap router model with a single-word yes/no prompt. Used
    only when retrieval lands in the medium-confidence band AND the
    lexical-overlap check fails, so the cost is amortized across the
    rare ambiguous case rather than every query.
    """

    def gate(query: str, passage: Passage) -> bool:
        # Cap the passage content so the gate prompt stays cheap.
        snippet = passage.content[:800]
        user = f"USER QUESTION:\n{query}\n\nRETRIEVED PASSAGE:\n{snippet}\n\nAnswer (yes/no):"
        raw = chat_model.complete_text(system=_RELEVANCE_GATE_SYSTEM, user=user)
        token = raw.strip().lower().strip(".'\"`")
        return token.startswith("y")

    return gate


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
    tiers: SourceTiersConfig,
    climate: ClimateConfig,
    grounding_config: GroundingConfig,
    weed_matches: dict[str, WeedAlias] | None = None,
    brand_matches: dict[str, ChemicalBrand] | None = None,
    synthesizer: ChatModel,
) -> Recommendation:
    # Caller may have already detected weeds/brands (for retrieval-query
    # expansion and the relevance check); re-use that work to avoid a
    # second regex scan.
    brand_matches = (
        brand_matches
        if brand_matches is not None
        else bridges.detect_brands_in_question(question, chemicals)
    )
    weed_matches = (
        weed_matches
        if weed_matches is not None
        else bridges.detect_weeds_in_question(question, weeds)
    )
    user_prompt = _synthesizer_user_prompt(
        question,
        intent,
        conditions,
        passages,
        tiers,
        climate,
        bridges.brand_bridge_text(brand_matches),
        bridges.weed_bridge_text(weed_matches),
    )
    return synthesis.synthesize_with_guardrails(
        system=_load_prompt("synthesizer.md"),
        user_prompt=user_prompt,
        passages=passages,
        chemicals=chemicals,
        grounding_config=grounding_config,
        synthesizer=synthesizer,
        log_prefix="orchestrator",
        refusal=_refusal,
        schema_failure_reason=(
            "synthesizer output failed schema validation twice; refusing "
            "rather than fabricating a recommendation"
        ),
        grounding_failure_reason=(
            "The retrieved sources do not support the chemical recommendations "
            "the model produced, twice in a row. Refusing rather than naming a "
            "product the sources do not discuss. Check Clemson HGIC or your "
            "local extension agent for this specific pest."
        ),
    )


def thresholds_block(climate: ClimateConfig) -> str:
    """Render the user's configured climate thresholds for the prompt.

    These values have lived in `config.yaml` since Phase 1 and were never
    read by any code. Both prompts asserted "threshold facts encoded in
    config (soil-temp green-up at 65F...)" while hardcoding those numbers
    in the prompt text — so editing the config changed nothing, and the
    model was told the values had a provenance they did not have.

    Injecting them makes the config load-bearing and lets the same prompt
    serve a different climate without an edit.
    """
    return "\n".join(
        (
            "<thresholds>",
            "The user's configured local thresholds. Prefer these over any",
            "general figures you may recall; they are this lawn's settings.",
            f"- Green-up (4-inch soil temp): {climate.green_up_soil_temp_f}F",
            f"- Dormancy onset (4-inch soil temp): {climate.dormancy_soil_temp_f}F",
            (
                "- Spring pre-emergent trigger (4-inch soil temp rising through): "
                f"{climate.preemergent_spring_soil_temp_f}F"
            ),
            (
                "- Fall pre-emergent trigger (4-inch soil temp falling through): "
                f"{climate.preemergent_fall_soil_temp_f}F"
            ),
            f"- Average last spring frost: {climate.last_frost.isoformat()}",
            f"- Average first fall frost: {climate.first_frost.isoformat()}",
            "</thresholds>",
        )
    )


def _synthesizer_user_prompt(
    question: str,
    intent: Intent,
    conditions: Conditions,
    passages: list[Passage],
    tiers: SourceTiersConfig,
    climate: ClimateConfig,
    brand_bridge: str = "",
    weed_bridge: str = "",
) -> str:
    bridges = "\n\n".join(b for b in (brand_bridge, weed_bridge) if b)
    bridge_block = f"\n\n{bridges}" if bridges else ""
    return (
        f"<intent>{intent}</intent>\n\n"
        f"<conditions>\n{conditions.model_dump_json(indent=2)}\n</conditions>\n\n"
        f"{thresholds_block(climate)}\n\n"
        f"<question>{question}</question>{bridge_block}\n\n"
        f"<sources>\n{knowledge.format_sources(passages, tiers)}\n</sources>"
    )


def _refusal(reason: str) -> Recommendation:
    return Recommendation(
        headline="Unable to produce a recommendation.",
        conditions_summary="",
        refused=True,
        refusal_reason=reason,
    )
