"""The standing annual program shell (`data/calendar.yaml`).

The shell carries timing and gating for recurring actions. It must never
carry products or rates — those are chemical specifics under ADR 0003,
resolved from cited passages at synthesis time and checked by ADR 0010.
A hardcoded rate here would bypass the entire guardrail stack.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from lawn_agents.config import ClimateConfig, Settings
from lawn_agents.models import (
    ChemicalCategory,
    GateDirection,
    ProgramConfig,
    Urgency,
)


@pytest.fixture
def program(repo_root: Path) -> ProgramConfig:
    raw = yaml.safe_load((repo_root / "data" / "calendar.yaml").read_text())
    return ProgramConfig.model_validate(raw)


class TestShippedProgramParses:
    def test_loads_and_has_items(self, program: ProgramConfig) -> None:
        assert len(program.items) >= 8

    def test_ids_are_unique(self, program: ProgramConfig) -> None:
        ids = [i.id for i in program.items]
        assert len(ids) == len(set(ids))

    def test_gates_reference_real_climate_fields(self, program: ProgramConfig) -> None:
        valid = set(ClimateConfig.model_fields)
        for item in program.items:
            if item.gate is not None:
                assert item.gate.threshold_ref in valid, item.id

    def test_every_chemical_item_has_a_rationale(self, program: ProgramConfig) -> None:
        """A chemical action with no stated reasoning is folklore."""
        for item in program.items:
            if isinstance(item.category, ChemicalCategory):
                assert item.rationale.strip(), item.id

    def test_critical_items_have_real_lead_time(self, program: ProgramConfig) -> None:
        """Critical items drive unprompted reminders, so they need buying time."""
        for item in program.items:
            if item.urgency is Urgency.CRITICAL:
                assert item.lead_days >= 14, item.id


class TestShellCarriesNoChemicalSpecifics:
    """ADR 0003's line: the shell schedules, the RAG prescribes."""

    def test_no_rates_in_the_shell(self, repo_root: Path) -> None:
        text = (repo_root / "data" / "calendar.yaml").read_text().lower()
        # Rate-shaped phrases that would indicate a smuggled prescription.
        for pattern in ("oz per 1000", "oz/1000", "lb per 1000", "lbs/1000", "fl oz"):
            assert pattern not in text, f"calendar.yaml contains a rate: {pattern!r}"

    def test_no_brand_names_in_item_names(self, repo_root: Path, program: ProgramConfig) -> None:
        raw = yaml.safe_load((repo_root / "data" / "chemicals.yaml").read_text())
        brands = [b.lower() for b in raw["brands"]]
        for item in program.items:
            for brand in brands:
                assert brand not in item.name.lower(), f"{item.id} names a product"


class TestLargePatchTimingCorrection:
    """The fungicide window is the one place the shell overrides the program.

    The vendor add-on schedules fungicide May-July for "brown patch".
    Brown patch is the cool-season disease; zoysia gets large patch, which
    NCSU puts at onset when soil temps decline through 70F — locally a
    median of Oct 17 across 12 years of SCAN data.
    """

    def test_fungicide_is_scheduled_for_fall(self, program: ProgramConfig) -> None:
        item = next(i for i in program.items if i.id == "fungicide-large-patch-fall")
        assert item.window.start_month >= 9
        assert item.gate is not None
        assert item.gate.direction is GateDirection.FALLING

    def test_rationale_records_why_the_vendor_window_was_rejected(
        self, program: ProgramConfig
    ) -> None:
        item = next(i for i in program.items if i.id == "fungicide-large-patch-fall")
        assert "brown-patch" in item.rationale or "brown patch" in item.rationale


class TestGateValidation:
    def test_unknown_threshold_ref_fails_at_load(
        self, config_yaml_path: Path, tmp_path: Path
    ) -> None:
        """A typo'd gate must fail loudly at startup, not resolve to nothing.

        This is the whole reason the check exists: `threshold_ref` is an
        indirection, and a mistyped one would produce a gate that never
        fires — silently, exactly like the `climate:` settings that sat
        unread for three months.
        """
        bad_program = tmp_path / "calendar.yaml"
        bad_program.write_text(
            yaml.safe_dump(
                {
                    "items": [
                        {
                            "id": "typo-gate",
                            "name": "Broken",
                            "category": "herbicide",
                            "window": {
                                "start_month": 3,
                                "start_day": 1,
                                "end_month": 4,
                                "end_day": 1,
                            },
                            "gate": {
                                "direction": "rising",
                                "threshold_ref": "greenup_soil_temp_f",  # missing underscore
                            },
                        }
                    ]
                }
            )
        )
        cfg = tmp_path / "config.yaml"
        text = config_yaml_path.read_text()
        cfg.write_text(f'{text}\nprogram_file: "{bad_program}"\n')

        with pytest.raises(ValidationError, match="unknown climate thresholds"):
            Settings.load(cfg)

    def test_duplicate_ids_are_rejected(self) -> None:
        window = {"start_month": 3, "start_day": 1, "end_month": 4, "end_day": 1}
        with pytest.raises(ValidationError, match="duplicate calendar item ids"):
            ProgramConfig.model_validate(
                {
                    "items": [
                        {"id": "dup", "name": "A", "category": "mowing", "window": window},
                        {"id": "dup", "name": "B", "category": "mowing", "window": window},
                    ]
                }
            )

    def test_sustained_days_defaults_to_one(self) -> None:
        cfg = ProgramConfig.model_validate(
            {
                "items": [
                    {
                        "id": "x",
                        "name": "X",
                        "category": "herbicide",
                        "window": {
                            "start_month": 3,
                            "start_day": 1,
                            "end_month": 4,
                            "end_day": 1,
                        },
                        "gate": {
                            "direction": "rising",
                            "threshold_ref": "green_up_soil_temp_f",
                        },
                    }
                ]
            }
        )
        assert cfg.items[0].gate is not None
        assert cfg.items[0].gate.sustained_days == 1
