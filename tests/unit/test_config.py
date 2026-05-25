"""Tests for the Settings loader (config.yaml + .env)."""

from __future__ import annotations

from pathlib import Path

import pytest

from lawn_agents.config import Settings


class TestSettingsLoader:
    """Smoke + structural assertions on the repo's `config.yaml`."""

    def test_loads_example_config(self, config_yaml_path: Path) -> None:
        settings = Settings.load(config_yaml_path)
        # Example config ships with a placeholder SC location.
        assert settings.app.location.state == "SC"
        assert settings.app.subject.cultivar == "Zeon Zoysia"
        assert settings.app.subject.kind == "lawn"
        assert settings.app.location.zip  # any non-empty placeholder

    def test_coastal_flag_default_is_false(self, config_yaml_path: Path) -> None:
        # Placeholder example is inland; users override for their location.
        settings = Settings.load(config_yaml_path)
        assert settings.app.location.coastal is False

    def test_research_allowlist_contains_supersod(self, config_yaml_path: Path) -> None:
        settings = Settings.load(config_yaml_path)
        # Use set membership so the `in` check is unambiguously element
        # equality (not substring) — keeps CodeQL's
        # py/incomplete-url-substring-sanitization rule from firing.
        allowed = set(settings.app.research.domain_allowlist)
        assert "info.supersod.com" in allowed
        assert "hgic.clemson.edu" in allowed

    def test_provider_and_models_pinned(self, config_yaml_path: Path) -> None:
        settings = Settings.load(config_yaml_path)
        assert settings.app.models.provider == "gemini"
        assert "flash" in settings.app.models.router
        assert "pro" in settings.app.models.synthesizer

    def test_secret_is_secret(self, config_yaml_path: Path) -> None:
        settings = Settings.load(config_yaml_path)
        assert settings.gemini_api_key is not None
        assert "test-dummy" in settings.gemini_api_key.get_secret_value()
        assert "test-dummy" not in repr(settings.gemini_api_key)

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            Settings.load(tmp_path / "does-not-exist.yaml")

    def test_missing_key_for_selected_provider_raises(
        self, config_yaml_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            Settings.load(config_yaml_path)
