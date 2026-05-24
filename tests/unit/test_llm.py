"""Tests for the ChatModel Protocol, adapters, and factory."""

from __future__ import annotations

from pathlib import Path

import pytest

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
