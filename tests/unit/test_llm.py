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
