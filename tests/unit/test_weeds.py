"""Tests for the weed common-name → alias bridge (ADR 0008)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from lawn_agents.config import Settings
from lawn_agents.models import WeedAlias, WeedsConfig


class TestWeedsConfigSchema:
    """`WeedsConfig` parses the YAML schema and rejects invalid entries."""

    def test_empty_is_valid(self) -> None:
        cfg = WeedsConfig()
        assert cfg.weeds == {}

    def test_typical_entry_round_trip(self) -> None:
        cfg = WeedsConfig.model_validate(
            {
                "weeds": {
                    "Japanese clover": {
                        "aliases": ["Annual lespedeza", "Lespedeza striata"],
                        "category": "broadleaf",
                        "notes": "Same plant, multiple names.",
                    }
                }
            }
        )
        assert "Japanese clover" in cfg.weeds
        weed = cfg.weeds["Japanese clover"]
        assert weed.aliases == ["Annual lespedeza", "Lespedeza striata"]
        assert weed.category.value == "broadleaf"

    def test_empty_aliases_rejected(self) -> None:
        with pytest.raises(ValidationError, match="aliases"):
            WeedsConfig.model_validate(
                {
                    "weeds": {
                        "BadEntry": {
                            "aliases": [],
                            "category": "broadleaf",
                        }
                    }
                }
            )

    def test_invalid_category_rejected(self) -> None:
        with pytest.raises(ValidationError, match="category"):
            WeedsConfig.model_validate(
                {
                    "weeds": {
                        "BadEntry": {
                            "aliases": ["something"],
                            "category": "not-a-real-category",
                        }
                    }
                }
            )


class TestSeededWeedsFile:
    """The committed `data/weeds.yaml` parses and seeds expected weeds."""

    @pytest.fixture
    def weeds(self, repo_root: Path) -> WeedsConfig:
        raw = yaml.safe_load((repo_root / "data" / "weeds.yaml").read_text(encoding="utf-8"))
        return WeedsConfig.model_validate(raw)

    def test_japanese_clover_aliases_include_annual_lespedeza(self, weeds: WeedsConfig) -> None:
        weed = weeds.weeds["Japanese clover"]
        assert "Annual lespedeza" in weed.aliases
        assert "Lespedeza striata" in weed.aliases
        assert "Kummerowia striata" in weed.aliases
        assert weed.category.value == "broadleaf"

    def test_nutsedge_categorized_as_sedge(self, weeds: WeedsConfig) -> None:
        assert weeds.weeds["Yellow nutsedge"].category.value == "sedge"
        assert weeds.weeds["Purple nutsedge"].category.value == "sedge"

    def test_crabgrass_categorized_as_grassy(self, weeds: WeedsConfig) -> None:
        assert weeds.weeds["Large crabgrass"].category.value == "grassy"

    def test_seeded_set_covers_all_categories(self, weeds: WeedsConfig) -> None:
        categories = {w.category.value for w in weeds.weeds.values()}
        assert categories == {"broadleaf", "grassy", "sedge"}


class TestSettingsLoadsWeeds:
    """`Settings.load` reads `data/weeds.yaml` when present."""

    def test_load_populates_weeds(
        self, tmp_path: Path, config_yaml_path: Path, repo_root: Path
    ) -> None:
        weeds_abs = repo_root / "data" / "weeds.yaml"
        text = config_yaml_path.read_text(encoding="utf-8")
        text += f'\nweeds_file: "{weeds_abs}"\n'
        override = tmp_path / "config.yaml"
        override.write_text(text, encoding="utf-8")
        settings = Settings.load(override)
        assert isinstance(settings.weeds, WeedsConfig)
        assert "Japanese clover" in settings.weeds.weeds

    def test_missing_weeds_file_yields_empty_config(
        self, tmp_path: Path, config_yaml_path: Path
    ) -> None:
        text = config_yaml_path.read_text(encoding="utf-8")
        text += f'\nweeds_file: "{tmp_path / "nope.yaml"}"\n'
        override = tmp_path / "config.yaml"
        override.write_text(text, encoding="utf-8")
        settings = Settings.load(override)
        assert settings.weeds.weeds == {}


class TestWeedAliasFrozen:
    """`WeedAlias` is frozen so cached entries can't be mutated."""

    def test_frozen_rejects_attribute_mutation(self) -> None:
        weed = WeedAlias(
            aliases=["Annual lespedeza"],
            category="broadleaf",  # type: ignore[arg-type]
        )
        with pytest.raises(ValidationError):
            weed.aliases = ["something-else"]  # type: ignore[misc]
