"""Program evaluation: window arithmetic and soil-temperature gates.

The window and the gate are deliberately independent. The window is pure
date math and keeps working when the SCAN station is unreachable — which
it often is — so the "go buy product" nudge survives an outage even when
"conditions say go" can't be determined.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest
import yaml

from lawn_agents import schedule
from lawn_agents.config import ClimateConfig, Settings
from lawn_agents.models import ProgramConfig, ProgramItem, SoilSnapshot


@pytest.fixture
def climate(config_yaml_path: Path) -> ClimateConfig:
    return Settings.load(config_yaml_path).app.climate


def _item(**overrides: object) -> ProgramItem:
    base: dict[str, object] = {
        "id": "test-item",
        "name": "Test item",
        "category": "herbicide",
        "urgency": "critical",
        "window": {"start_month": 9, "start_day": 15, "end_month": 10, "end_day": 20},
        "lead_days": 14,
    }
    base.update(overrides)
    return ProgramConfig.model_validate({"items": [base]}).items[0]


def _soil(trailing: list[float]) -> SoilSnapshot:
    return SoilSnapshot(
        fetched_at=datetime(2026, 9, 11, tzinfo=UTC),
        station_id="2038:SC:SCAN",
        current_4in_f=trailing[-1] if trailing else None,
        trailing_7d_4in_f=trailing,
    )


class TestWindowArithmetic:
    def test_before_lead_time_is_not_yet(self, climate: ClimateConfig) -> None:
        a = schedule.evaluate(_item(), climate, today=date(2026, 8, 1))
        assert a.status is schedule.ItemStatus.NOT_YET
        assert not a.actionable

    def test_inside_lead_time_is_approaching(self, climate: ClimateConfig) -> None:
        a = schedule.evaluate(_item(), climate, today=date(2026, 9, 5))
        assert a.status is schedule.ItemStatus.APPROACHING
        assert a.actionable
        assert "10 days" in a.reason

    def test_lead_boundary_is_inclusive(self, climate: ClimateConfig) -> None:
        """Exactly `lead_days` out must already be approaching."""
        a = schedule.evaluate(_item(), climate, today=date(2026, 9, 1))
        assert a.status is schedule.ItemStatus.APPROACHING

    def test_after_window_is_passed(self, climate: ClimateConfig) -> None:
        a = schedule.evaluate(_item(), climate, today=date(2026, 11, 1))
        assert a.status is schedule.ItemStatus.PASSED
        assert not a.actionable

    def test_near_window_end_is_closing(self, climate: ClimateConfig) -> None:
        a = schedule.evaluate(_item(gate=None), climate, today=date(2026, 10, 16))
        assert a.status is schedule.ItemStatus.CLOSING
        assert "4 days left" in a.reason

    def test_open_window_without_gate_is_ready(self, climate: ClimateConfig) -> None:
        a = schedule.evaluate(_item(), climate, today=date(2026, 9, 20))
        assert a.status is schedule.ItemStatus.READY


class TestWindowEdgeCases:
    def test_end_day_31_clamps_in_short_months(self, climate: ClimateConfig) -> None:
        """`end_day: 31` is a natural way to write 'end of month'."""
        item = _item(window={"start_month": 2, "start_day": 1, "end_month": 2, "end_day": 31})
        _, end = schedule.window_dates(item, date(2026, 2, 10))
        assert end == date(2026, 2, 28)

    def test_wrapping_window_contains_january(self, climate: ClimateConfig) -> None:
        """Dec 1 - Feb 15 must resolve across New Year, not backwards."""
        item = _item(window={"start_month": 12, "start_day": 1, "end_month": 2, "end_day": 15})
        start, end = schedule.window_dates(item, date(2026, 1, 10))
        assert start == date(2025, 12, 1)
        assert end == date(2026, 2, 15)
        assert start <= date(2026, 1, 10) <= end

    def test_wrapping_window_from_december_looks_forward(self, climate: ClimateConfig) -> None:
        item = _item(window={"start_month": 12, "start_day": 1, "end_month": 2, "end_day": 15})
        start, end = schedule.window_dates(item, date(2026, 12, 10))
        assert start == date(2026, 12, 1)
        assert end == date(2027, 2, 15)


class TestGates:
    def _gated(self, direction: str, sustained: int = 3) -> ProgramItem:
        return _item(
            gate={
                "metric": "soil_temp_4in_f",
                "direction": direction,
                "threshold_ref": "preemergent_fall_soil_temp_f",  # 70F
                "sustained_days": sustained,
            }
        )

    def test_falling_gate_satisfied(self, climate: ClimateConfig) -> None:
        a = schedule.evaluate(
            self._gated("falling"),
            climate,
            today=date(2026, 9, 20),
            soil=_soil([75, 72, 69, 68, 67]),
        )
        assert a.gate_status is schedule.GateStatus.SATISFIED
        assert a.status is schedule.ItemStatus.READY

    def test_falling_gate_unmet_when_still_warm(self, climate: ClimateConfig) -> None:
        a = schedule.evaluate(
            self._gated("falling"),
            climate,
            today=date(2026, 9, 20),
            soil=_soil([83, 84, 78, 81, 82]),
        )
        assert a.gate_status is schedule.GateStatus.UNMET
        assert a.status is schedule.ItemStatus.WAITING_ON_GATE
        assert "82F" in a.reason

    def test_sustained_requirement_rejects_a_single_cold_day(self, climate: ClimateConfig) -> None:
        """One dip below threshold must not open the gate.

        The whole reason `sustained_days` exists: a bare first-crossing
        fires on a warm week in January.
        """
        a = schedule.evaluate(
            self._gated("falling"),
            climate,
            today=date(2026, 9, 20),
            soil=_soil([80, 80, 75, 72, 68]),
        )
        assert a.gate_status is schedule.GateStatus.UNMET

    def test_rising_gate_satisfied(self, climate: ClimateConfig) -> None:
        item = _item(
            window={"start_month": 1, "start_day": 1, "end_month": 12, "end_day": 31},
            gate={
                "metric": "soil_temp_4in_f",
                "direction": "rising",
                "threshold_ref": "green_up_soil_temp_f",  # 65F
                "sustained_days": 3,
            },
        )
        a = schedule.evaluate(
            item, climate, today=date(2026, 4, 1), soil=_soil([60, 63, 66, 67, 68])
        )
        assert a.gate_status is schedule.GateStatus.SATISFIED


class TestDegradesWithoutSoilData:
    """The station is frequently unreachable; the window must still work."""

    def test_approaching_needs_no_soil_data(self, climate: ClimateConfig) -> None:
        a = schedule.evaluate(_item(), climate, today=date(2026, 9, 5), soil=None)
        assert a.status is schedule.ItemStatus.APPROACHING
        assert a.actionable, "the 'go buy product' nudge must survive an outage"

    def test_open_window_without_soil_waits_rather_than_guessing(
        self, climate: ClimateConfig
    ) -> None:
        item = _item(
            gate={
                "metric": "soil_temp_4in_f",
                "direction": "falling",
                "threshold_ref": "preemergent_fall_soil_temp_f",
                "sustained_days": 3,
            }
        )
        a = schedule.evaluate(item, climate, today=date(2026, 9, 20), soil=None)
        assert a.gate_status is schedule.GateStatus.UNKNOWN
        assert a.status is schedule.ItemStatus.WAITING_ON_GATE
        assert "unavailable" in a.reason

    def test_history_shorter_than_sustained_days_is_unknown(self, climate: ClimateConfig) -> None:
        item = _item(
            gate={
                "metric": "soil_temp_4in_f",
                "direction": "falling",
                "threshold_ref": "preemergent_fall_soil_temp_f",
                "sustained_days": 5,
            }
        )
        a = schedule.evaluate(item, climate, today=date(2026, 9, 20), soil=_soil([68, 67]))
        assert a.gate_status is schedule.GateStatus.UNKNOWN


class TestEvaluateAll:
    """Runs against the real `data/calendar.yaml`, not a fixture.

    Loaded explicitly via `repo_root` because `program_file` is
    cwd-relative and conftest chdirs to a temp directory — the same
    reason `chemicals_file` and `weeds_file` don't auto-load in tests.
    """

    @pytest.fixture
    def shipped(self, repo_root: Path) -> ProgramConfig:
        raw = yaml.safe_load((repo_root / "data" / "calendar.yaml").read_text())
        return ProgramConfig.model_validate(raw)

    def test_shipped_program_sorts_actionable_first(
        self, shipped: ProgramConfig, config_yaml_path: Path
    ) -> None:
        settings = Settings.load(config_yaml_path)
        results = schedule.evaluate_all(
            shipped.items,
            settings.app.climate,
            today=date(2026, 9, 11),
            soil=_soil([83, 84, 78, 81, 80, 81, 82]),
        )
        actionable = [a for a in results if a.actionable]
        assert actionable, "mid-September should have live items"
        # Actionable items come before the rest.
        first_non_actionable = next(
            (i for i, a in enumerate(results) if not a.actionable), len(results)
        )
        assert all(a.actionable for a in results[:first_non_actionable])

    def test_fall_preemergent_is_approaching_in_early_september(
        self, shipped: ProgramConfig, config_yaml_path: Path
    ) -> None:
        settings = Settings.load(config_yaml_path)
        results = schedule.evaluate_all(
            shipped.items, settings.app.climate, today=date(2026, 9, 11)
        )
        fall = next(a for a in results if a.item.id == "preemergent-fall")
        assert fall.status is schedule.ItemStatus.APPROACHING
