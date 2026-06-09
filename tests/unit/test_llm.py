"""Tests for the ChatModel Protocol, adapters, and factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from lawn_agents.config import Settings
from lawn_agents.llm import (
    AnthropicChat,
    ChatModel,
    GeminiChat,
    build_chat_model,
    parse_router_intent,
)


class TestParseRouterIntent:
    """The router model's output sometimes arrives wrapped; normalize it."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("ad-hoc", "ad-hoc"),
            ("AD-HOC", "ad-hoc"),
            ("  scheduled-check  ", "scheduled-check"),
            ("'plan-month'", "plan-month"),
            ('"plan-year"', "plan-year"),
            ("`out-of-scope`", "out-of-scope"),
            ("ad-hoc.", "ad-hoc"),
            ('{"intent": "ad-hoc"}', "ad-hoc"),
            ('{"intent": "PLAN-MONTH"}', "plan-month"),
        ],
    )
    def test_normalizes(self, raw: str, expected: str) -> None:
        assert parse_router_intent(raw) == expected

    def test_malformed_json_falls_back_to_text(self) -> None:
        assert parse_router_intent('{not-json: "ad-hoc"}') == '{not-json: "ad-hoc"}'


class TestBuildChatModelFactory:
    """`build_chat_model` picks the right adapter based on settings."""

    def test_gemini_provider_returns_gemini_adapter(self, config_yaml_path: Path) -> None:
        settings = Settings.load(config_yaml_path)
        router = build_chat_model("router", settings)
        synth = build_chat_model("synthesizer", settings)
        assert isinstance(router, GeminiChat)
        assert isinstance(synth, GeminiChat)

    def test_anthropic_provider_returns_anthropic_adapter(
        self, config_yaml_path: Path, tmp_path: Path
    ) -> None:
        # Write a one-off YAML overriding only the provider/model IDs.
        text = config_yaml_path.read_text(encoding="utf-8").replace(
            'provider: "gemini"', 'provider: "anthropic"'
        )
        text = text.replace('router: "gemini-2.5-flash"', 'router: "claude-sonnet-4-6"')
        text = text.replace('synthesizer: "gemini-2.5-pro"', 'synthesizer: "claude-opus-4-7"')
        override = tmp_path / "config.yaml"
        override.write_text(text, encoding="utf-8")
        settings = Settings.load(override)
        router = build_chat_model("router", settings)
        assert isinstance(router, AnthropicChat)

    def test_missing_gemini_key_raises(
        self, config_yaml_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            Settings.load(config_yaml_path)


class TestChatModelProtocolSatisfaction:
    """Both adapters structurally satisfy the `ChatModel` Protocol."""

    def test_gemini_is_chat_model(self) -> None:
        adapter = GeminiChat(api_key="test-dummy", model="gemini-2.5-flash")
        # `isinstance` against a non-runtime Protocol won't work; use
        # the duck-typed contract instead.
        chat: ChatModel = adapter
        assert callable(chat.complete_text)
        assert callable(chat.complete_structured)

    def test_anthropic_is_chat_model(self) -> None:
        adapter = AnthropicChat(api_key="sk-ant-test-dummy", model="claude-sonnet-4-6")
        chat: ChatModel = adapter
        assert callable(chat.complete_text)
        assert callable(chat.complete_structured)


class _ToyResponse(BaseModel):
    headline: str
    score: int


class _FakeGenAIResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeGenAIModels:
    def __init__(self, text: str) -> None:
        self._text = text
        self.last_kwargs: dict[str, Any] | None = None

    def generate_content(self, **kwargs: Any) -> _FakeGenAIResponse:
        self.last_kwargs = kwargs
        return _FakeGenAIResponse(self._text)


class _FakeGenAIClient:
    def __init__(self, text: str) -> None:
        self.models = _FakeGenAIModels(text)


class TestGeminiCompleteStructured:
    """`GeminiChat.complete_structured` must not pass `response_schema` to the API.

    Pydantic v2 schemas include `additionalProperties: false` and `$ref`,
    both of which the Gemini API rejects. We embed the schema in the
    system prompt and validate post-hoc with Pydantic instead.
    """

    def _build_adapter(self, *, response_text: str) -> tuple[GeminiChat, _FakeGenAIModels]:
        adapter = GeminiChat(api_key="test-dummy", model="gemini-2.5-pro")
        fake_client = _FakeGenAIClient(response_text)
        adapter._client = fake_client  # type: ignore[assignment]
        return adapter, fake_client.models

    def test_does_not_pass_response_schema(self) -> None:
        adapter, fake = self._build_adapter(response_text='{"headline": "ok", "score": 1}')
        adapter.complete_structured(system="sys", user="usr", response_model=_ToyResponse)
        assert fake.last_kwargs is not None
        config = fake.last_kwargs["config"]
        assert getattr(config, "response_schema", None) is None
        assert config.response_mime_type == "application/json"

    def test_embeds_schema_in_system_instruction(self) -> None:
        adapter, fake = self._build_adapter(response_text='{"headline": "ok", "score": 1}')
        adapter.complete_structured(system="orig system", user="usr", response_model=_ToyResponse)
        assert fake.last_kwargs is not None
        instruction = fake.last_kwargs["config"].system_instruction
        assert "orig system" in instruction
        assert "headline" in instruction
        assert "score" in instruction

    def test_parses_response_via_pydantic(self) -> None:
        adapter, _ = self._build_adapter(response_text='{"headline": "hi", "score": 7}')
        result = adapter.complete_structured(system="sys", user="usr", response_model=_ToyResponse)
        assert result.headline == "hi"
        assert result.score == 7

    def test_empty_response_raises(self) -> None:
        adapter, _ = self._build_adapter(response_text="")
        with pytest.raises(RuntimeError, match="empty structured response"):
            adapter.complete_structured(system="sys", user="usr", response_model=_ToyResponse)


class TestGeminiClientRetryConfig:
    """Regression: the Gemini client must be constructed with retry enabled.

    The SDK's `HttpOptions(retry_options=HttpRetryOptions())` flips on the
    default retry behavior (5 attempts, 408/429/5xx, exp backoff with
    jitter, 1-60s). Without it the SDK does NOT retry transient errors
    and we surface 503s to the user as hard refusals.
    """

    def test_client_constructed_with_retry_options(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from google import genai
        from google.genai import types

        captured: dict[str, Any] = {}

        original_init = genai.Client.__init__

        def spy_init(self: Any, **kwargs: Any) -> None:
            # Don't call the real init — it tries to validate the API key.
            captured.update(kwargs)

        monkeypatch.setattr(genai.Client, "__init__", spy_init)
        try:
            GeminiChat(api_key="test-dummy", model="gemini-2.5-flash")
        finally:
            monkeypatch.setattr(genai.Client, "__init__", original_init)

        http_options = captured.get("http_options")
        assert http_options is not None
        assert isinstance(http_options, types.HttpOptions)
        assert http_options.retry_options is not None
        assert isinstance(http_options.retry_options, types.HttpRetryOptions)


class TestAnthropicClientConfig:
    """Anthropic client should use a 60s timeout, not the 10-min SDK default."""

    def test_client_constructed_with_60s_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import anthropic

        captured: dict[str, Any] = {}

        original_init = anthropic.Anthropic.__init__

        def spy_init(self: Any, **kwargs: Any) -> None:
            captured.update(kwargs)

        monkeypatch.setattr(anthropic.Anthropic, "__init__", spy_init)
        try:
            AnthropicChat(api_key="sk-ant-test-dummy", model="claude-sonnet-4-6")
        finally:
            monkeypatch.setattr(anthropic.Anthropic, "__init__", original_init)

        assert captured.get("timeout") == 60.0


class _FakeGeminiUsageMetadata:
    def __init__(self) -> None:
        self.prompt_token_count = 100
        self.candidates_token_count = 200
        self.cached_content_token_count = 0
        self.total_token_count = 300


class _FakeGeminiResponseWithUsage:
    def __init__(self, text: str) -> None:
        self.text = text
        self.usage_metadata = _FakeGeminiUsageMetadata()


class _FakeAnthropicUsage:
    def __init__(self) -> None:
        self.input_tokens = 100
        self.output_tokens = 200
        self.cache_read_input_tokens = 0
        self.cache_creation_input_tokens = 0


class _FakeAnthropicResponse:
    def __init__(self) -> None:
        self.usage = _FakeAnthropicUsage()
        self.content: list[Any] = []
        self._request_id = "req_test_123"


class TestUsageLogging:
    """Each adapter logs token usage (and Anthropic request_id) per call."""

    def test_gemini_logs_usage(self) -> None:
        import structlog

        from lawn_agents.llm import _log_gemini_usage

        with structlog.testing.capture_logs() as captured:
            _log_gemini_usage(
                "gemini-2.5-flash", "complete_structured", _FakeGeminiResponseWithUsage("ok")
            )
        events = [e for e in captured if e.get("event") == "llm.gemini_usage"]
        assert len(events) == 1
        event = events[0]
        assert event["input_tokens"] == 100
        assert event["output_tokens"] == 200
        assert event["total_tokens"] == 300
        assert event["model"] == "gemini-2.5-flash"
        assert event["op"] == "complete_structured"

    def test_anthropic_logs_usage_and_request_id(self) -> None:
        import structlog

        from lawn_agents.llm import _log_anthropic_usage

        with structlog.testing.capture_logs() as captured:
            _log_anthropic_usage(
                "claude-sonnet-4-6", "complete_structured", _FakeAnthropicResponse()
            )
        events = [e for e in captured if e.get("event") == "llm.anthropic_usage"]
        assert len(events) == 1
        event = events[0]
        assert event["input_tokens"] == 100
        assert event["output_tokens"] == 200
        assert event["request_id"] == "req_test_123"
        assert event["model"] == "claude-sonnet-4-6"

    def test_no_usage_attribute_no_log(self) -> None:
        import structlog

        from lawn_agents.llm import _log_anthropic_usage, _log_gemini_usage

        class _Bare:
            pass

        with structlog.testing.capture_logs() as captured:
            _log_gemini_usage("m", "op", _Bare())
            _log_anthropic_usage("m", "op", _Bare())
        assert not [e for e in captured if "_usage" in e.get("event", "")]
