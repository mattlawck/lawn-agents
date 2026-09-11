"""Reconciling the standing program against Todoist's open tasks.

The rules this enforces, in order of how much damage getting them wrong
would do:

1. Never modify a hand-written task. Overlaps are reported only.
2. Never create a second task for something already tracked.
3. Don't interrupt unprompted for routine work.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

from lawn_agents import reconcile
from lawn_agents.config import ClimateConfig, Settings
from lawn_agents.models import ProgramConfig, ProgramItem
from lawn_agents.todoist import Task, marker


@pytest.fixture
def climate(config_yaml_path: Path) -> ClimateConfig:
    return Settings.load(config_yaml_path).app.climate


@pytest.fixture
def shipped(repo_root: Path) -> ProgramConfig:
    raw = yaml.safe_load((repo_root / "data" / "calendar.yaml").read_text())
    return ProgramConfig.model_validate(raw)


def _task(content: str, *, due: date | None = None, task_id: str = "t1") -> Task:
    from lawn_agents.todoist import parse_marker

    parsed = parse_marker(content)
    return Task(
        id=task_id,
        content=content,
        description="",
        due=due,
        labels=(),
        item_id=parsed[0] if parsed else None,
        year=parsed[1] if parsed else None,
    )


def _item(item_id: str, name: str, urgency: str = "critical") -> ProgramItem:
    return ProgramConfig.model_validate(
        {
            "items": [
                {
                    "id": item_id,
                    "name": name,
                    "category": "herbicide",
                    "urgency": urgency,
                    "window": {
                        "start_month": 9,
                        "start_day": 15,
                        "end_month": 10,
                        "end_day": 20,
                    },
                    "lead_days": 14,
                }
            ]
        }
    ).items[0]


TODAY = date(2026, 9, 11)


class TestOverdueReview:
    def test_separates_owned_from_hand_written(self, climate: ClimateConfig) -> None:
        tasks = [
            _task(f"Fall pre-emergent {marker('preemergent-fall', 2026)}", due=date(2026, 9, 1)),
            _task("Spray Sedge Ender (sulfentrazone) on nutsedge", due=date(2026, 9, 5)),
        ]
        plan = reconcile.reconcile([], tasks, climate, today=TODAY)
        assert [t.item_id for t in plan.overdue_owned] == ["preemergent-fall"]
        assert [t.content for t in plan.overdue_manual] == [
            "Spray Sedge Ender (sulfentrazone) on nutsedge"
        ]

    def test_future_tasks_are_not_overdue(self, climate: ClimateConfig) -> None:
        plan = reconcile.reconcile(
            [], [_task("later", due=date(2026, 12, 1))], climate, today=TODAY
        )
        assert plan.overdue_manual == []


class TestDeduplication:
    def test_tracked_item_is_not_reproposed(self, climate: ClimateConfig) -> None:
        item = _item("preemergent-fall", "Fall pre-emergent")
        existing = _task(
            f"Fall pre-emergent {marker('preemergent-fall', 2026)}", due=date(2026, 9, 15)
        )
        plan = reconcile.reconcile([item], [existing], climate, today=TODAY)
        assert plan.proposals == []
        assert len(plan.tracked) == 1

    def test_missing_item_is_proposed(self, climate: ClimateConfig) -> None:
        item = _item("preemergent-fall", "Fall pre-emergent")
        plan = reconcile.reconcile([item], [], climate, today=TODAY)
        assert [p.item.id for p in plan.proposals] == ["preemergent-fall"]
        assert plan.proposals[0].marker == "[preemergent-fall/2026]"


class TestOverlapsAreReportedNeverTouched:
    def test_misspelled_hand_written_task_is_recognised(self, climate: ClimateConfig) -> None:
        """`Apply Pre Emergnent` covers the fall pre-emergent.

        Matched on the shared `emerg` stem — generic words like "fall"
        are stripped first, so one distinctive stem is enough.
        """
        item = _item("preemergent-fall", "Fall pre-emergent")
        plan = reconcile.reconcile([item], [_task("Apply Pre Emergnent")], climate, today=TODAY)
        assert len(plan.overlaps) == 1
        assert plan.overlaps[0].task.content == "Apply Pre Emergnent"

    def test_an_overlap_suppresses_the_proposal(self, climate: ClimateConfig) -> None:
        """Better to report a possible duplicate than create a real one."""
        item = _item("preemergent-fall", "Fall pre-emergent")
        plan = reconcile.reconcile([item], [_task("Apply Pre Emergnent")], climate, today=TODAY)
        assert plan.proposals == []

    def test_unrelated_tasks_do_not_match(self, climate: ClimateConfig) -> None:
        item = _item("preemergent-fall", "Fall pre-emergent")
        plan = reconcile.reconcile(
            [item], [_task("Oak Trees Spikes"), _task("Azalea & Roses Plan")], climate, today=TODAY
        )
        assert plan.overlaps == []
        assert len(plan.proposals) == 1

    def test_generic_seasonal_words_alone_do_not_match(self, climate: ClimateConfig) -> None:
        """ "Fall" appears everywhere; it must not pair unrelated work."""
        item = _item("preemergent-fall", "Fall pre-emergent")
        plan = reconcile.reconcile([item], [_task("Fall cleanup")], climate, today=TODAY)
        assert plan.overlaps == []


class TestUrgentOnly:
    def test_watchdog_mode_skips_routine_items(self, climate: ClimateConfig) -> None:
        """An unattended reminder has to earn the interruption."""
        items = [
            _item("critical-thing", "Critical thing", urgency="critical"),
            _item("routine-thing", "Routine thing", urgency="routine"),
        ]
        plan = reconcile.reconcile(items, [], climate, today=TODAY, urgent_only=True)
        assert [p.item.id for p in plan.proposals] == ["critical-thing"]

    def test_interactive_mode_includes_routine(self, climate: ClimateConfig) -> None:
        items = [
            _item("critical-thing", "Critical thing", urgency="critical"),
            _item("routine-thing", "Routine thing", urgency="routine"),
        ]
        plan = reconcile.reconcile(items, [], climate, today=TODAY, urgent_only=False)
        assert len(plan.proposals) == 2


class TestProposalContent:
    def test_chemical_items_defer_product_and_rate(self, climate: ClimateConfig) -> None:
        """The shell must never smuggle in a prescription."""
        item = _item("preemergent-fall", "Fall pre-emergent")
        plan = reconcile.reconcile([item], [], climate, today=TODAY)
        body = plan.proposals[0].description
        assert "not fixed here" in body or "deliberately not fixed" in body

    def test_dates_keep_their_capitalisation(self, climate: ClimateConfig) -> None:
        """`.capitalize()` would render 'Sep 15' as 'sep 15'."""
        item = _item("preemergent-fall", "Fall pre-emergent")
        plan = reconcile.reconcile([item], [], climate, today=TODAY)
        assert "Sep 15" in plan.proposals[0].description

    def test_due_never_falls_in_the_past(self, climate: ClimateConfig) -> None:
        item = _item("preemergent-fall", "Fall pre-emergent")
        plan = reconcile.reconcile([item], [], climate, today=date(2026, 10, 1))
        assert plan.proposals[0].due >= date(2026, 10, 1)


class TestAgainstShippedProgram:
    def test_quiet_when_everything_is_tracked(
        self, shipped: ProgramConfig, climate: ClimateConfig
    ) -> None:
        """Silence is the goal — a weekly nag gets muted."""
        from lawn_agents import schedule

        actionable = [
            a.item
            for a in schedule.evaluate_all(shipped.items, climate, today=TODAY)
            if a.actionable
        ]
        tasks = [
            _task(f"{i.name} {marker(i.id, 2026)}", due=date(2026, 12, 1), task_id=i.id)
            for i in actionable
        ]
        plan = reconcile.reconcile(shipped.items, tasks, climate, today=TODAY)
        assert plan.proposals == []
        assert not plan.needs_attention
