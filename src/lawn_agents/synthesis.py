"""The shared synthesize → validate → re-prompt → refuse loop.

The orchestrator and the planner both turn a prompt into a validated
`Recommendation` through exactly the same sequence:

1. Call the synthesizer for structured output.
2. On schema failure, re-prompt once with corrective guidance.
3. On success, run claim-support checks (ADR 0010).
4. On grounding failure, re-prompt once with the specific unsupported
   terms.
5. Re-validate the retry; refuse rather than emit an unsupported
   chemical recommendation.
6. Classify any SDK-level error so a transport failure never renders as
   a content refusal.

Those two implementations drifted into 70% duplication — ~210 lines
expressing one policy twice. The cost showed up when ADR 0010 landed:
grounding had to be threaded through both by hand, and a miss in either
would have left a synthesis path silently unguarded. The never-guess
policy is the most safety-critical logic in the project, so having it
exist once is worth more here than anywhere else in the codebase.

Callers keep what genuinely differs — their own system prompt, user
prompt, log prefix and refusal wording — and share the control flow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import ValidationError

from lawn_agents import grounding
from lawn_agents.llm import classify_llm_error
from lawn_agents.logging import get_logger
from lawn_agents.models import Recommendation

if TYPE_CHECKING:
    from collections.abc import Callable

    from lawn_agents.config import GroundingConfig
    from lawn_agents.llm import ChatModel
    from lawn_agents.models import ChemicalsConfig, Passage

log = get_logger(__name__)

SCHEMA_RETRY_GUIDANCE = (
    "Your previous response failed schema validation. Return JSON that "
    "satisfies the Recommendation schema. Chemical-category CalendarItems "
    "(fertilizer, micronutrient, herbicide, insecticide, fungicide) require "
    "at least one Citation grounded in the provided <sources>. If you cannot "
    "ground a chemical recommendation, set refused=true and refusal_reason."
)


def synthesize_with_guardrails(
    *,
    system: str,
    user_prompt: str,
    passages: list[Passage],
    chemicals: ChemicalsConfig,
    grounding_config: GroundingConfig,
    synthesizer: ChatModel,
    log_prefix: str,
    refusal: Callable[[str], Recommendation],
    schema_failure_reason: str,
    grounding_failure_reason: str,
) -> Recommendation:
    """Run one synthesis through both guardrails, with a single retry.

    Args:
        system: System prompt for this role.
        user_prompt: Fully-built user prompt including `<sources>`.
        passages: The passages actually placed in `<sources>`; grounding
            validates citations against these.
        chemicals: Brand bridge, used as grounding's chemical vocabulary.
        grounding_config: ADR 0010 thresholds and enable flag.
        synthesizer: The `ChatModel` to call.
        log_prefix: Event namespace, e.g. `"orchestrator"` or
            `"planner"`, so structlog keeps the two paths distinguishable.
        refusal: Builds a caller-appropriate refusal `Recommendation`.
        schema_failure_reason: Refusal text when validation fails twice.
        grounding_failure_reason: Refusal text when grounding fails twice.

    Returns:
        A validated, grounded `Recommendation`, or a refusal.
    """
    try:
        draft = synthesizer.complete_structured(
            system=system, user=user_prompt, response_model=Recommendation
        )
    except ValidationError as exc:
        log.info(f"{log_prefix}.synthesizer_validation_failed", error=str(exc))
        retry_guidance = SCHEMA_RETRY_GUIDANCE
    except Exception as exc:
        return _classified_refusal(exc, log_prefix, "synthesizer", refusal)
    else:
        failures = grounding.verify(draft, passages, chemicals, grounding_config)
        if not failures:
            return draft
        log.info(
            f"{log_prefix}.grounding_failed",
            count=len(failures),
            kinds=sorted({f.kind for f in failures}),
        )
        retry_guidance = grounding.format_failures(failures)

    retry_user = f"{user_prompt}\n\n---\n\n{retry_guidance}"
    try:
        retried = synthesizer.complete_structured(
            system=system, user=retry_user, response_model=Recommendation
        )
    except ValidationError as exc:
        log.warning(f"{log_prefix}.synthesizer_final_validation_failure", error=str(exc))
        return refusal(schema_failure_reason)
    except Exception as exc:
        # The retry itself hit an SDK error (auth, rate limit, server).
        # Classify it so the message reflects the real terminal failure
        # rather than implying the model refused on content.
        return _classified_refusal(exc, log_prefix, "synthesizer_retry", refusal)

    retry_failures = grounding.verify(retried, passages, chemicals, grounding_config)
    if retry_failures:
        log.warning(
            f"{log_prefix}.grounding_final_failure",
            count=len(retry_failures),
            kinds=sorted({f.kind for f in retry_failures}),
        )
        return refusal(grounding_failure_reason)
    return retried


def _classified_refusal(
    exc: Exception,
    log_prefix: str,
    stage: str,
    refusal: Callable[[str], Recommendation],
) -> Recommendation:
    event_suffix, reason = classify_llm_error(exc)
    log.warning(
        f"{log_prefix}.{stage}_{event_suffix}",
        error=str(exc),
        error_type=type(exc).__name__,
    )
    return refusal(reason)
