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
        assert categories == {"insecticide", "herbicide", "fungicide"}


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
