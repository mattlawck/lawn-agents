"""Remembering last run's open tasks, so their absence carries meaning.

Absence is the completion signal, but only against a memory of what was
there before. Without one, a task you completed and a task that never
existed are indistinguishable — so the moment you finish something is
the moment the system loses the thread.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

from lawn_agents import tasklog
from lawn_agents.models import ProgramConfig
from lawn_agents.todoist import Task, marker


def _task(item_id: str, task_id: str = "t1", name: str = "Thing") -> Task:
    return Task(
        id=task_id,
        content=f"{name} {marker(item_id, 2026)}",
        description="",
        due=date(2026, 9, 15),
        labels=(),
        item_id=item_id,
        year=2026,
    )


@pytest.fixture
def shipped(repo_root: Path) -> ProgramConfig:
    raw = yaml.safe_load((repo_root / "data" / "calendar.yaml").read_text())
    return ProgramConfig.model_validate(raw)


class TestRoundTrip:
    def test_save_then_load(self, tmp_path: Path) -> None:
        path = tmp_path / "snap.json"
        tasklog.save(path, [_task("preemergent-fall", "a"), _task("scout-grubs-fall", "b")])
        loaded = tasklog.load(path)
        assert loaded is not None
        assert {t.item_id for t in loaded.tasks} == {"preemergent-fall", "scout-grubs-fall"}

    def test_hand_written_tasks_are_not_recorded(self, tmp_path: Path) -> None:
        """The system does not track what it did not create."""
        manual = Task("m", "Apply Pre Emergnent", "", None, (), None, None)
        path = tmp_path / "snap.json"
        tasklog.save(path, [manual, _task("preemergent-fall", "a")])
        loaded = tasklog.load(path)
        assert loaded is not None
        assert [t.item_id for t in loaded.tasks] == ["preemergent-fall"]


class TestSelfCorrecting:
    """Failures cost one cycle of detection, never a run.

    This snapshot is deliberately not the durable state the project has
    avoided elsewhere: nothing downstream reads it as fact, so the worst
    a wrong file can do is prompt one unnecessary question.
    """

    def test_missing_file_is_a_first_run(self, tmp_path: Path) -> None:
        assert tasklog.load(tmp_path / "absent.json") is None

    def test_corrupt_file_does_not_raise(self, tmp_path: Path) -> None:
        path = tmp_path / "snap.json"
        path.write_text("{ this is not json")
        assert tasklog.load(path) is None

    def test_unknown_schema_version_is_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "snap.json"
        path.write_text('{"schema_version": 99, "taken_at": "2026-09-15T00:00:00+00:00"}')
        assert tasklog.load(path) is None

    def test_unwritable_path_does_not_raise(self, tmp_path: Path) -> None:
        """Bookkeeping must never fail a run."""
        blocker = tmp_path / "blocked"
        blocker.write_text("i am a file, not a directory")
        tasklog.save(blocker / "snap.json", [_task("preemergent-fall")])

    def test_no_previous_snapshot_reports_nothing_gone(self) -> None:
        """A first run must not claim everything was just completed."""
        assert tasklog.disappeared(None, [_task("preemergent-fall")]) == []


class TestDisappearance:
    def test_task_present_then_absent_is_detected(self, tmp_path: Path) -> None:
        path = tmp_path / "snap.json"
        tasklog.save(path, [_task("preemergent-fall", "a"), _task("scout-grubs-fall", "b")])
        previous = tasklog.load(path)
        gone = tasklog.disappeared(previous, [_task("preemergent-fall", "a")])
        assert [g.item_id for g in gone] == ["scout-grubs-fall"]

    def test_still_open_is_not_disappeared(self, tmp_path: Path) -> None:
        path = tmp_path / "snap.json"
        tasks = [_task("preemergent-fall", "a")]
        tasklog.save(path, tasks)
        assert tasklog.disappeared(tasklog.load(path), tasks) == []


class TestFollowups:
    """Only inspections leave a question behind."""

    def test_monitoring_item_asks_what_you_found(self, shipped: ProgramConfig) -> None:
        gone = [tasklog.SeenTask("b", "scout-grubs-fall", "Scout for grub damage", None)]
        followups = tasklog.needing_followup(gone, shipped.items)
        assert [f[0].item_id for f in followups] == ["scout-grubs-fall"]

    def test_action_item_asks_nothing(self, shipped: ProgramConfig) -> None:
        """Finishing an application is fully described by finishing it."""
        gone = [tasklog.SeenTask("a", "preemergent-fall", "Fall pre-emergent", None)]
        assert tasklog.needing_followup(gone, shipped.items) == []

    def test_unknown_item_id_is_skipped(self, shipped: ProgramConfig) -> None:
        """A renamed or retired shell item must not crash the diff."""
        gone = [tasklog.SeenTask("x", "no-such-item", "Ghost", None)]
        assert tasklog.needing_followup(gone, shipped.items) == []
