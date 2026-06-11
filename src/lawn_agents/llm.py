"""Provider-agnostic chat-model interface.

Two adapters (`GeminiChat`, `AnthropicChat`) implement a single Protocol.
The orchestrator depends on the Protocol, not on either SDK, so swapping
providers is a config-file change. See ADR 0006.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel

from lawn_agents.logging import get_logger

if TYPE_CHECKING:
    from lawn_agents.config import Settings

ChatRole = Literal["router", "synthesizer"]

log = get_logger(__name__)

# Gemini cache settings. TTL chosen to span a typical homeowner session:
# enough that running `lawn-agents --ask` a few times in quick succession
# shares cache hits, short enough that stale prompt content doesn't linger
# on Google's infra after the user moves on. Storage cost is negligible at
# our prompt sizes.
_GEMINI_CACHE_TTL = "3600s"
_GEMINI_CACHE_DISPLAY_NAME_PREFIX = "lawn-agents-"


class ChatModel(Protocol):
    """A chat model that can return either free-form text or a typed object.

    Adapters implement this Protocol structurally — no inheritance
    required. The orchestrator depends only on this Protocol, so swapping
    providers is a single line in `config.yaml`.
    """

    def complete_text(self, *, system: str, user: str) -> str:
        """Generate free-form text. Used by the router for intent classification."""

    def complete_structured[T: BaseModel](
        self, *, system: str, user: str, response_model: type[T]
    ) -> T:
        """Generate output that validates against `response_model`.

        Raises:
            pydantic.ValidationError: If the model output doesn't validate.
            RuntimeError: If the provider returned no usable content.
        """


class GeminiChat:
    """`ChatModel` backed by Google's `google-genai` SDK (AI Studio or Vertex)."""

    def __init__(self, *, api_key: str, model: str) -> None:
        """Initialize the Gemini client.

        Args:
            api_key: AI Studio API key.
            model: Model ID, e.g. `gemini-2.5-flash` or `gemini-2.5-pro`.
        """
        from google import genai
        from google.genai import types

        # Enable the SDK's built-in retry on 408/429/5xx with exponential
        # backoff and jitter. Defaults: 5 attempts, 1.0s initial delay,
        # 60.0s max delay, exp_base=2.0, jitter=1.0. We were silently
        # failing on transient 503s before this because retry_options was
        # unset; see fix/synth-retry-on-5xx.
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(retry_options=types.HttpRetryOptions()),
        )
        self._model = model
        # In-process cache of (system_instruction hash → server-side cache.name).
        # Populated lazily on first call. See `_get_or_create_system_cache`.
        self._cache_by_hash: dict[str, str] = {}

    def _get_or_create_system_cache(self, system_instruction: str) -> str | None:
        """Return cache.name for the system instruction, or None if uncacheable.

        Three reuse tiers:
        1. **In-process**: same `GeminiChat` instance, same system text → dict hit.
        2. **Cross-process**: `caches.list()` filtered by `display_name` finds a
           server-side cache from a prior CLI invocation within the 3600s TTL.
        3. **Create**: no existing cache → `caches.create()` with the hash as
           `display_name` so future invocations can find this one.

        Returns None if cache creation fails. The most common reason is the
        2,048-token minimum for Gemini 2.5 Flash/Pro: short prompts (e.g., the
        router system) silently fall through to the uncached path. The caller
        should pass `system_instruction` directly in that case.
        """
        digest = hashlib.sha256(system_instruction.encode("utf-8")).hexdigest()[:16]
        if digest in self._cache_by_hash:
            return self._cache_by_hash[digest]

        display_name = f"{_GEMINI_CACHE_DISPLAY_NAME_PREFIX}{digest}"

        try:
            for cached in self._client.caches.list():
                if (
                    getattr(cached, "display_name", None) == display_name
                    and cached.name is not None
                ):
                    self._cache_by_hash[digest] = cached.name
                    log.info(
                        "llm.gemini_cache_reused",
                        model=self._model,
                        cache_name=cached.name,
                    )
                    return cached.name
        except Exception as exc:
            log.warning("llm.gemini_cache_list_failed", error=str(exc))
            # Fall through to creation — listing failure shouldn't block.

        try:
            from google.genai import types

            cache = self._client.caches.create(
                model=f"models/{self._model}",
                config=types.CreateCachedContentConfig(
                    display_name=display_name,
                    system_instruction=system_instruction,
                    ttl=_GEMINI_CACHE_TTL,
                ),
            )
        except Exception as exc:
            # The 2,048-token minimum is the usual culprit; we don't differentiate
            # because the fallback is the same in every failure case.
            log.info("llm.gemini_cache_skip", model=self._model, error=str(exc))
            return None
        if cache.name is None:
            log.warning("llm.gemini_cache_no_name", model=self._model)
            return None
        self._cache_by_hash[digest] = cache.name
        log.info("llm.gemini_cache_created", model=self._model, cache_name=cache.name)
        return cache.name

    def complete_text(self, *, system: str, user: str) -> str:
        """See `ChatModel.complete_text`."""
        from google.genai import types

        cache_name = self._get_or_create_system_cache(system)
        config = (
            types.GenerateContentConfig(cached_content=cache_name)
            if cache_name
            else types.GenerateContentConfig(system_instruction=system)
        )
        response = self._client.models.generate_content(
            model=self._model,
            contents=user,
            config=config,
        )
        _log_gemini_usage(self._model, "complete_text", response)
        return (response.text or "").strip()

    def complete_structured[T: BaseModel](
        self, *, system: str, user: str, response_model: type[T]
    ) -> T:
        """See `ChatModel.complete_structured`."""
        from google.genai import types

        # Gemini's response_schema rejects Pydantic v2's `additionalProperties`
        # and `$ref`s. Embed the schema in the prompt and validate with
        # Pydantic; the orchestrator handles parse errors via re-prompt-then-refuse.
        schema_hint = json.dumps(response_model.model_json_schema(), indent=2)
        system_with_schema = (
            f"{system}\n\n"
            "Reply with ONLY a JSON object matching this schema "
            f"(no prose, no code fences):\n{schema_hint}"
        )
        cache_name = self._get_or_create_system_cache(system_with_schema)
        config = (
            types.GenerateContentConfig(
                cached_content=cache_name,
                response_mime_type="application/json",
            )
            if cache_name
            else types.GenerateContentConfig(
                system_instruction=system_with_schema,
                response_mime_type="application/json",
            )
        )
        response = self._client.models.generate_content(
            model=self._model,
            contents=user,
            config=config,
        )
        _log_gemini_usage(self._model, "complete_structured", response)
        text = response.text or ""
        if not text:
            msg = "Gemini returned an empty structured response."
            raise RuntimeError(msg)
        return response_model.model_validate_json(text)


class AnthropicChat:
    """`ChatModel` backed by Anthropic's official Python SDK.

    Kept alongside `GeminiChat` so the user can A/B providers or fall
    back without re-architecting. Phase 1 default is Gemini.
    """

    def __init__(self, *, api_key: str, model: str) -> None:
        """Initialize the Anthropic client.

        Args:
            api_key: `console.anthropic.com` API key.
            model: Model ID, e.g. `claude-sonnet-4-6` or `claude-opus-4-7`.
        """
        from anthropic import Anthropic

        # The SDK default timeout is 10 minutes (intended for very long
        # streaming requests). For our interactive CLI use case 60s is a
        # better ceiling — user has lost interest long before then. The
        # SDK still applies its default `max_retries=2` to transient
        # 408/409/429/5xx + connection errors with exponential backoff,
        # so 60s x ~3 attempts is the worst-case wall clock.
        self._client = Anthropic(api_key=api_key, timeout=60.0)
        self._model = model

    def complete_text(self, *, system: str, user: str) -> str:
        """See `ChatModel.complete_text`.

        Sends `system` as a list with `cache_control: ephemeral` so the
        SDK caches the system prompt server-side. Cache reads bill at
        0.1x the base input rate; cache writes at 1.25x (5-min TTL).
        Below the model's minimum cacheable-token threshold (1024 for
        Sonnet 4.6, 4096 for Opus 4.7), Anthropic silently ignores the
        flag — no error, no benefit. See `_log_anthropic_usage` for
        `cache_read_input_tokens` / `cache_creation_input_tokens`
        observability.
        """
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user}],
        )
        _log_anthropic_usage(self._model, "complete_text", response)
        for block in response.content:
            if getattr(block, "type", None) == "text":
                return str(getattr(block, "text", "")).strip()
        return ""

    def complete_structured[T: BaseModel](
        self, *, system: str, user: str, response_model: type[T]
    ) -> T:
        """See `ChatModel.complete_structured`.

        Both the system prompt and the tool definition carry
        `cache_control: ephemeral`. Per Anthropic's docs, tools are
        cached as one unit and live above system in the cache
        hierarchy; placing the breakpoint on the (single) tool caches
        the whole tools block. The system block caches independently.
        """
        tool_name = "respond"
        tool_schema = response_model.model_json_schema()
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user}],
            tools=[
                {
                    "name": tool_name,
                    "description": "Return the structured response.",
                    "input_schema": tool_schema,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tool_choice={"type": "tool", "name": tool_name},
        )
        _log_anthropic_usage(self._model, "complete_structured", response)
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                payload = getattr(block, "input", None)
                if payload is None:
                    continue
                if isinstance(payload, str):
                    return response_model.model_validate_json(payload)
                return response_model.model_validate(payload)
        msg = "Anthropic response did not include a tool_use block."
        raise RuntimeError(msg)


def build_chat_model(role: ChatRole, settings: Settings) -> ChatModel:
    """Construct a `ChatModel` for the given role from validated settings.

    Args:
        role: Which model to build — the cheap router or the synthesizer.
        settings: Validated `Settings` (knows the provider and both keys).

    Returns:
        A `GeminiChat` or `AnthropicChat` instance, typed as `ChatModel`.

    Raises:
        ValueError: If the configured provider is unknown or its API key
            is missing.
    """
    provider = settings.app.models.provider
    model_id = settings.app.models.router if role == "router" else settings.app.models.synthesizer

    if provider == "gemini":
        if settings.gemini_api_key is None:
            msg = "provider=gemini but GEMINI_API_KEY is unset. Add it to .env (see .env.example)."
            raise ValueError(msg)
        return GeminiChat(
            api_key=settings.gemini_api_key.get_secret_value(),
            model=model_id,
        )
    if provider == "anthropic":
        if settings.anthropic_api_key is None:
            msg = (
                "provider=anthropic but ANTHROPIC_API_KEY is unset. "
                "Add it to .env (see .env.example)."
            )
            raise ValueError(msg)
        return AnthropicChat(
            api_key=settings.anthropic_api_key.get_secret_value(),
            model=model_id,
        )
    # Unreachable per the Literal["gemini", "anthropic"] type. The
    # explicit raise (rather than `assert_never`) keeps CodeQL's
    # py/mixed-returns from flagging a potential None fall-through;
    # the `type: ignore` suppresses mypy's warn_unreachable at this
    # defensive guard.
    msg = f"unsupported provider: {provider!r}"  # type: ignore[unreachable]
    raise ValueError(msg)


def _log_gemini_usage(model: str, op: str, response: object) -> None:
    """Emit a structlog event with Gemini token usage if available."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return
    log.info(
        "llm.gemini_usage",
        model=model,
        op=op,
        input_tokens=getattr(usage, "prompt_token_count", None),
        output_tokens=getattr(usage, "candidates_token_count", None),
        cached_tokens=getattr(usage, "cached_content_token_count", None),
        total_tokens=getattr(usage, "total_token_count", None),
    )


def _log_anthropic_usage(model: str, op: str, response: object) -> None:
    """Emit a structlog event with Anthropic token usage + request_id."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    log.info(
        "llm.anthropic_usage",
        model=model,
        op=op,
        request_id=getattr(response, "_request_id", None),
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
        cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", None),
        cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", None),
    )


def classify_llm_error(exc: BaseException) -> tuple[str, str]:
    """Classify a provider-SDK exception for structlog + user-facing messaging.

    Returns ``(event_suffix, user_reason)``:

    - ``event_suffix`` slots into a structlog event name (e.g.,
      ``orchestrator.synthesizer_<suffix>``). One value per distinct
      failure class so dashboards can break down by mode.
    - ``user_reason`` is the short string that goes into a `Refused`
      panel — actionable per class (auth → "check API key"; rate-limit
      → "try again"; server → "API is having issues").

    Keeps SDK imports inside this module so callers (orchestrator,
    planner) stay decoupled from provider SDKs per ADR 0006. Lazy
    imports also mean the function doesn't pay the cost of loading a
    provider SDK that isn't installed.

    Anthropic docs explicitly recommend catching typed SDK classes
    rather than string-matching error messages:
    https://platform.claude.com/docs/en/api/errors#sdk-error-types
    """
    # Anthropic: import lazily so we don't force a dep load when the
    # configured provider is Gemini. `contextlib.suppress(ImportError)`
    # is the idiomatic "this branch is optional if the SDK isn't
    # installed" pattern — also satisfies CodeQL's `py/empty-except`
    # check, which can't tell that the `pass` was intentional.
    with contextlib.suppress(ImportError):
        import anthropic

        if isinstance(exc, anthropic.AuthenticationError):
            return ("auth_failed", "API authentication failed; check your ANTHROPIC_API_KEY.")
        if isinstance(exc, anthropic.PermissionDeniedError):
            return ("permission_denied", "API key lacks permission for this resource.")
        if isinstance(exc, anthropic.RateLimitError):
            return (
                "rate_limited",
                "Hit the Anthropic rate limit; try again in a moment.",
            )
        if isinstance(exc, anthropic.APIConnectionError):
            return (
                "connection_failed",
                "Could not reach Anthropic; check your network connection.",
            )
        if isinstance(exc, anthropic.InternalServerError):
            return (
                "server_error",
                "Anthropic API is having issues; please try again shortly.",
            )
        if isinstance(exc, anthropic.BadRequestError):
            return ("bad_request", f"Anthropic rejected the request: {exc}")

    # Gemini / google-genai.
    with contextlib.suppress(ImportError):
        from google.genai import errors as genai_errors

        if isinstance(exc, genai_errors.ServerError):
            return (
                "server_error",
                "Gemini API is having issues; please try again shortly.",
            )
        if isinstance(exc, genai_errors.ClientError):
            return ("client_error", f"Gemini rejected the request: {exc}")

    # Fallback — unknown exception. Caller still gets a structured event
    # and a meaningful (if generic) refusal reason.
    return ("call_failed", f"{type(exc).__name__}: {exc}")


def parse_router_intent(raw: str) -> str:
    """Normalize a router model's free-form output to a single intent token.

    Args:
        raw: The router's `complete_text` output.

    Returns:
        Lowercased, trimmed token. The orchestrator further validates
        against the supported intent literals.
    """
    # Defensive: some models occasionally wrap output in JSON or quotes.
    text = raw.strip().lower()
    if text.startswith("{"):
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "intent" in obj:
                text = str(obj["intent"]).strip().lower()
        except json.JSONDecodeError:
            # Not JSON; fall through to the plain-text normalization below.
            pass
    return text.strip("\"'`. \t\n")
