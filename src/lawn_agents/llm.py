"""Provider-agnostic chat-model interface.

Two adapters (`GeminiChat`, `AnthropicChat`) implement a single Protocol.
The orchestrator depends on the Protocol, not on either SDK, so swapping
providers is a config-file change. See ADR 0006.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel

from lawn_agents.logging import get_logger

if TYPE_CHECKING:
    from lawn_agents.config import Settings

ChatRole = Literal["router", "synthesizer"]

log = get_logger(__name__)


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

    def complete_text(self, *, system: str, user: str) -> str:
        """See `ChatModel.complete_text`."""
        from google.genai import types

        response = self._client.models.generate_content(
            model=self._model,
            contents=user,
            config=types.GenerateContentConfig(system_instruction=system),
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
        response = self._client.models.generate_content(
            model=self._model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system_with_schema,
                response_mime_type="application/json",
            ),
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
        """See `ChatModel.complete_text`."""
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
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
        """See `ChatModel.complete_structured`."""
        tool_name = "respond"
        tool_schema = response_model.model_json_schema()
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[
                {
                    "name": tool_name,
                    "description": "Return the structured response.",
                    "input_schema": tool_schema,
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
