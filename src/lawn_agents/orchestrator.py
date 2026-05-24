"""Orchestrator — routes intent, fans out to agents, synthesizes the answer.

The orchestrator is the only module that sees the full pipeline. It:

1. Routes the user's intent via a `ChatModel` (router role).
2. Fetches conditions (weather, soil temp, drought) in parallel — all
   fail-closed so a flaky NWS endpoint can't kill the run.
3. Retrieves passages from the local RAG; optionally invokes the
   research subagent when retrieval is weak (ADR 0005).
4. Synthesizes the final answer via a `ChatModel` (synthesizer role).
5. Validates the result through the Pydantic guardrail (ADR 0003);
   re-prompts once if validation fails, then surfaces a refusal.

Provider selection (Gemini vs. Anthropic) is decoupled behind the
`ChatModel` Protocol in `lawn_agents.llm`. See ADR 0006.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from lawn_agents.config import Settings
    from lawn_agents.models import Recommendation

Intent = Literal["scheduled-check", "ad-hoc", "plan-month", "plan-year", "out-of-scope"]


def route_intent(question: str, settings: Settings) -> Intent:
    """Classify the user's intent using the router model.

    Args:
        question: The user's natural-language input.
        settings: Validated settings (used for the router model ID and API key).

    Returns:
        One of the supported intents.
    """
    raise NotImplementedError


def answer(question: str, settings: Settings) -> Recommendation:
    """Run the full pipeline for a single user question.

    Args:
        question: The user's natural-language input.
        settings: Validated settings.

    Returns:
        A validated `Recommendation`. May be a refusal if the
        never-guess guardrail (ADR 0003) rejects the synthesizer's draft.
    """
    raise NotImplementedError


def scheduled_check(settings: Settings) -> Recommendation:
    """Run the weekly scheduled check workflow.

    Args:
        settings: Validated settings.

    Returns:
        A validated `Recommendation` describing this week's actions.
    """
    raise NotImplementedError
