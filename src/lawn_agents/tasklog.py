"""Remembering which tasks were open last run, so their absence means something.

Todoist is the state layer and absence is the completion signal — but
absence only carries information if you remember what was there before.
Each run otherwise starts blind: a task you completed and a task that
never existed look identical, so the moment you finish something is the
moment the system loses the thread.

That matters most for monitoring work. "Apply pre-emergent" is binary
and completion says everything. "Scout for grub damage" is not:
completing it records that you *looked*, never what you *saw*, and what
you saw is the entire point. Eight grubs per square foot and none at all
collapse to the same checkbox, and next June's preventive decision is
made with no memory of either.

So this keeps a snapshot of the owned tasks seen on the previous run and
diffs it against the current one. Anything that vanished was completed
(or deleted — the two are indistinguishable from the open list alone,
which is itself worth surfacing).

**This is deliberately not the durable state this project has otherwise
avoided.** A mis-logged application date silently corrupts every later
recommendation; that kind of state has to wait for real completion
history. This snapshot is derived and self-correcting: nothing
downstream reads it as fact, it only decides whether to ask a question.
Lose the file and the next run rebuilds it. Get it wrong and the cost is
one spurious "did you finish this?" — which re-syncs immediately.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

from lawn_agents.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from lawn_agents.models import ProgramItem
    from lawn_agents.todoist import Task

log = get_logger(__name__)

SCHEMA_VERSION = 1


@dataclass(slots=True, frozen=True)
class SeenTask:
    """One owned task as it looked on a previous run."""

    task_id: str
    item_id: str
    content: str
    due: date | None

    def to_json(self) -> dict[str, Any]:
        """Serialise for the snapshot file."""
        return {
            "task_id": self.task_id,
            "item_id": self.item_id,
            "content": self.content,
            "due": self.due.isoformat() if self.due else None,
        }

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> SeenTask:
        """Rebuild from the snapshot file."""
        raw_due = row.get("due")
        return cls(
            task_id=str(row["task_id"]),
            item_id=str(row["item_id"]),
            content=str(row.get("content", "")),
            due=date.fromisoformat(raw_due) if raw_due else None,
        )


@dataclass(slots=True, frozen=True)
class Snapshot:
    """The owned tasks observed at a point in time."""

    taken_at: datetime
    tasks: tuple[SeenTask, ...]

    def by_id(self) -> dict[str, SeenTask]:
        """Index by Todoist task id."""
        return {t.task_id: t for t in self.tasks}


def load(path: Path) -> Snapshot | None:
    """Read the previous snapshot, or None on a first run or bad file.

    Any failure returns None rather than raising. A missing or corrupt
    snapshot must never block a run — the cost is one cycle without
    disappearance detection, and the next save repairs it.
    """
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != SCHEMA_VERSION:
            log.info("tasklog.schema_mismatch", found=raw.get("schema_version"))
            return None
        return Snapshot(
            taken_at=datetime.fromisoformat(raw["taken_at"]),
            tasks=tuple(SeenTask.from_json(r) for r in raw.get("tasks", [])),
        )
    except Exception as exc:
        log.warning("tasklog.load_failed", error=str(exc), path=str(path))
        return None


def save(path: Path, tasks: list[Task]) -> None:
    """Record the owned tasks currently open.

    Unowned tasks are excluded: the system does not track what it did not
    create, and a hand-written task disappearing is none of its business.
    """
    rows = [
        SeenTask(
            task_id=t.id,
            item_id=t.item_id or "",
            content=t.content,
            due=t.due,
        ).to_json()
        for t in tasks
        if t.owned and t.item_id
    ]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "taken_at": datetime.now(UTC).isoformat(),
        "tasks": rows,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info("tasklog.saved", count=len(rows), path=str(path))
    except Exception as exc:
        # Never fail a run over bookkeeping.
        log.warning("tasklog.save_failed", error=str(exc), path=str(path))


def disappeared(previous: Snapshot | None, current: list[Task]) -> list[SeenTask]:
    """Owned tasks present last run and absent now — completed or deleted."""
    if previous is None:
        return []
    open_ids = {t.id for t in current}
    return [t for t in previous.tasks if t.task_id not in open_ids]


def needing_followup(
    gone: list[SeenTask],
    items: list[ProgramItem],
) -> list[tuple[SeenTask, ProgramItem]]:
    """Disappeared tasks whose program item produced information, not just work.

    Only `monitoring` items qualify. Finishing an application is fully
    described by finishing it; finishing an inspection is not, and the
    result has no other place to live.
    """
    by_id = {i.id: i for i in items}
    out: list[tuple[SeenTask, ProgramItem]] = []
    for seen in gone:
        item = by_id.get(seen.item_id)
        if item is not None and item.category.value == "monitoring":
            out.append((seen, item))
    return out
