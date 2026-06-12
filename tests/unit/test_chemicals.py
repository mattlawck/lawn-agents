"""Tests for the brand → active-ingredient bridge (ADR 0007)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from lawn_agents.config import Settings
from lawn_agents.models import ChemicalBrand, ChemicalsConfig


class TestChemicalsConfigSchema:
    """`ChemicalsConfig` parses the YAML schema and rejects invalid entries."""

    def test_empty_is_valid(self) -> None:
        cfg = ChemicalsConfig()
        assert cfg.brands == {}

    def test_typical_entry_round_trip(self) -> None:
        cfg = ChemicalsConfig.model_validate(
            {
                "brands": {
                    "GrubX": {
                        "active_ingredients": ["chlorantraniliprole"],
                        "category": "insecticide",
                        "notes": "Scotts consumer brand.",
                    }
                }
            }
        )
        assert "GrubX" in cfg.brands
        brand = cfg.brands["GrubX"]
        assert brand.active_ingredients == ["chlorantraniliprole"]
        assert brand.category.value == "insecticide"

    def test_empty_active_ingredients_rejected(self) -> None:
        with pytest.raises(ValidationError, match="active_ingredients"):
            ChemicalsConfig.model_validate(
                {
                    "brands": {
                        "BadEntry": {
                            "active_ingredients": [],
                            "category": "insecticide",
                        }
                    }
                }
            )

    def test_invalid_category_rejected(self) -> None:
        with pytest.raises(ValidationError, match="category"):
            ChemicalsConfig.model_validate(
                {
                    "brands": {
                        "BadEntry": {
                            "active_ingredients": ["something"],
                            "category": "not-a-real-category",
                        }
                    }
                }
            )


class TestSeededChemicalsFile:
    """The committed `data/chemicals.yaml` parses and seeds expected brands."""

    @pytest.fixture
    def chemicals(self, repo_root: Path) -> ChemicalsConfig:
        raw = yaml.safe_load((repo_root / "data" / "chemicals.yaml").read_text(encoding="utf-8"))
        return ChemicalsConfig.model_validate(raw)

    def test_grubx_maps_to_chlorantraniliprole(self, chemicals: ChemicalsConfig) -> None:
        brand = chemicals.brands["GrubX"]
        assert brand.active_ingredients == ["chlorantraniliprole"]
        assert brand.category.value == "insecticide"

    def test_celsius_has_three_active_ingredients(self, chemicals: ChemicalsConfig) -> None:
        brand = chemicals.brands["Celsius"]
        assert len(brand.active_ingredients) == 3
        assert "dicamba" in brand.active_ingredients

    def test_sedgehammer_is_halosulfuron(self, chemicals: ChemicalsConfig) -> None:
        brand = chemicals.brands["Sedgehammer"]
        assert brand.active_ingredients == ["halosulfuron-methyl"]
        assert brand.category.value == "herbicide"

    def test_seeded_set_covers_all_chemical_categories(self, chemicals: ChemicalsConfig) -> None:
        categories = {b.category.value for b in chemicals.brands.values()}
        assert categories == {"insecticide", "herbicide", "fungicide", "fertilizer"}

    def test_milorganite_present_with_correct_npk(self, chemicals: ChemicalsConfig) -> None:
        """Milorganite is 6-4-0 (reformulated from 5-3-0 around 2024).

        The notes field is the load-bearing one for drought-fertilization
        questions: it captures the low-salt / low-burn-risk property that
        differentiates biosolids from synthetic fast-release urea.
        """
        brand = chemicals.brands["Milorganite"]
        assert brand.category.value == "fertilizer"
        assert "nitrogen" in brand.active_ingredients
        assert brand.notes is not None
        assert "6-4-0" in brand.notes
        assert "low salt index" in brand.notes.lower() or "low-salt" in brand.notes.lower()

    def test_sta_green_16_0_10_present(self, chemicals: ChemicalsConfig) -> None:
        brand = chemicals.brands["Sta-Green 16-0-10"]
        assert brand.category.value == "fertilizer"
        assert brand.notes is not None
        assert "16-0-10" in brand.notes

    def test_sedge_ender_is_sulfentrazone(self, chemicals: ChemicalsConfig) -> None:
        """Bonide Sedge Ender — PPO inhibitor; different chemistry from Sedgehammer.

        Notes capture the complementary nature with halosulfuron-methyl so
        the synthesizer can reason about sequential / paired applications.
        """
        brand = chemicals.brands["Sedge Ender"]
        assert brand.active_ingredients == ["sulfentrazone"]
        assert brand.category.value == "herbicide"
        assert brand.notes is not None
        assert "PPO" in brand.notes
        # Cross-reference to halosulfuron-methyl chemistry; the synthesizer
        # uses this to reason about complementary tank-mix / sequential apps.
        assert "halosulfuron-methyl" in brand.notes

    def test_sedgehammer_notes_cross_reference_sedge_ender(
        self, chemicals: ChemicalsConfig
    ) -> None:
        """Sedgehammer entry calls out the paired-use strategy with sulfentrazone."""
        brand = chemicals.brands["Sedgehammer"]
        assert brand.notes is not None
        assert "sulfentrazone" in brand.notes
        # Mention of the translocation behavior — the load-bearing detail
        # for why the two chemistries are complementary.
        assert "translocat" in brand.notes.lower()


class TestSettingsLoadsChemicals:
    """`Settings.load` reads `data/chemicals.yaml` when present."""

    def test_load_populates_chemicals(
        self, tmp_path: Path, config_yaml_path: Path, repo_root: Path
    ) -> None:
        # The autouse `_isolate_env` fixture chdir's to a tmp cwd, so the
        # default relative `data/chemicals.yaml` won't resolve. Point at
        # the committed file by absolute path via a config override.
        chemicals_abs = repo_root / "data" / "chemicals.yaml"
        text = config_yaml_path.read_text(encoding="utf-8")
        text += f'\nchemicals_file: "{chemicals_abs}"\n'
        override = tmp_path / "config.yaml"
        override.write_text(text, encoding="utf-8")
        settings = Settings.load(override)
        assert isinstance(settings.chemicals, ChemicalsConfig)
        assert "GrubX" in settings.chemicals.brands

    def test_missing_chemicals_file_yields_empty_config(
        self, tmp_path: Path, config_yaml_path: Path
    ) -> None:
        # Point chemicals_file at a path that doesn't exist; load should
        # still succeed with an empty ChemicalsConfig.
        text = config_yaml_path.read_text(encoding="utf-8")
        text += f'\nchemicals_file: "{tmp_path / "nope.yaml"}"\n'
        override = tmp_path / "config.yaml"
        override.write_text(text, encoding="utf-8")
        settings = Settings.load(override)
        assert settings.chemicals.brands == {}


class TestChemicalBrandFrozen:
    """`ChemicalBrand` is frozen so cached entries can't be mutated."""

    def test_frozen_rejects_attribute_mutation(self) -> None:
        brand = ChemicalBrand(
            active_ingredients=["imidacloprid"],
            category="insecticide",  # type: ignore[arg-type]
        )
        with pytest.raises(ValidationError):
            brand.active_ingredients = ["something-else"]  # type: ignore[misc]
