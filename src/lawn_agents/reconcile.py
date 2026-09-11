"""Reconcile the standing program against what's actually in Todoist.

This is the open-task review that opens a planning session, and the same
logic the weekly watchdog runs unattended.

It answers four questions in one pass:

1. **What's slipping?** An owned task past its due date means the work
   didn't happen. Unowned overdue tasks are surfaced too — a
   hand-written task going stale is exactly the signal that the plan and
   the season have diverged.
2. **What's already tracked?** An owned task for an item means the
   program has it in hand; don't create a second one.
3. **What overlaps?** A hand-written task may cover a program item
   without carrying its marker. Those are reported, never modified —
   rewriting something the user wrote by hand is not this system's
   business.
4. **What's missing?** An actionable item with no task is a proposal.

Nothing here writes. `Plan` is a proposal the caller renders for
approval; `apply` performs it only once a human has said yes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from lawn_agents import schedule, todoist
from lawn_agents.logging import get_logger
from lawn_agents.models import ChemicalCategory

if TYPE_CHECKING:
    from datetime import date

    from lawn_agents.config import ClimateConfig
    from lawn_agents.models import ProgramItem, SoilSnapshot
    from lawn_agents.schedule import Assessment
    from lawn_agents.todoist import Task, TodoistClient

log = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class Proposal:
    """A task the program would like to create."""

    item: ProgramItem
    assessment: Assessment
    title: str
    description: str
    due: date

    @property
    def marker(self) -> str:
        """Title suffix identifying this task's program item and year."""
        return todoist.marker(self.item.id, self.due.year)


@dataclass(slots=True, frozen=True)
class Overlap:
    """A hand-written task that appears to cover a program item."""

    task: Task
    item: ProgramItem
    why: str


@dataclass(slots=True)
class Plan:
    """The reconciliation result — a proposal, not an action."""

    overdue_owned: list[Task] = field(default_factory=list)
    overdue_manual: list[Task] = field(default_factory=list)
    tracked: list[Task] = field(default_factory=list)
    overlaps: list[Overlap] = field(default_factory=list)
    proposals: list[Proposal] = field(default_factory=list)

    @property
    def needs_attention(self) -> bool:
        """True when there is anything worth interrupting a human for."""
        return bool(self.overdue_owned or self.overdue_manual or self.proposals)


def reconcile(
    items: list[ProgramItem],
    tasks: list[Task],
    climate: ClimateConfig,
    *,
    today: date,
    soil: SoilSnapshot | None = None,
    urgent_only: bool = False,
) -> Plan:
    """Compare the program's wants against Todoist's open tasks.

    Args:
        items: The standing program.
        tasks: Open tasks from the configured project.
        climate: Thresholds the gates reference.
        today: Evaluation date.
        soil: Current soil snapshot, or None when unavailable.
        urgent_only: Propose tasks only for `critical` items. The weekly
            watchdog sets this — an unattended reminder has to earn the
            interruption, and a system that files routine chores every
            Monday gets muted.

    Returns:
        A `Plan`. Nothing is written.
    """
    plan = Plan()
    owned_by_item = {t.item_id: t for t in tasks if t.owned and t.item_id}

    for task in tasks:
        if not task.is_overdue(today):
            continue
        (plan.overdue_owned if task.owned else plan.overdue_manual).append(task)

    for assessment in schedule.evaluate_all(items, climate, today=today, soil=soil):
        item = assessment.item
        existing = owned_by_item.get(item.id)
        if existing is not None:
            plan.tracked.append(existing)
            continue
        if not assessment.actionable:
            continue

        overlap = _find_overlap(item, tasks)
        if overlap is not None:
            plan.overlaps.append(overlap)
            continue
        if urgent_only and item.urgency.value != "critical":
            continue
        plan.proposals.append(_propose(assessment, today))

    log.info(
        "reconcile.done",
        overdue_owned=len(plan.overdue_owned),
        overdue_manual=len(plan.overdue_manual),
        tracked=len(plan.tracked),
        overlaps=len(plan.overlaps),
        proposals=len(plan.proposals),
    )
    return plan


def apply(plan: Plan, client: TodoistClient, *, label: str | None) -> list[Task]:
    """Create the proposed tasks. Call only after a human approves."""
    created: list[Task] = []
    for proposal in plan.proposals:
        created.append(
            client.create_task(
                content=f"{proposal.title} {proposal.marker}",
                description=proposal.description,
                due=proposal.due,
                labels=[label] if label else [],
            )
        )
    return created


# --- internals ------------------------------------------------------------

_STOPWORDS = frozenset({"the", "a", "an", "for", "and", "of", "on", "in", "to", "with"})

# Words that carry no identifying power in a lawn-task list. "Fall
# pre-emergent" and "Bag Mow Leaves" both contain "fall"-ish seasonal
# framing; matching on it would pair almost anything with anything.
_GENERIC = frozenset(
    {
        "fall",
        "sprin",
        "summe",
        "winte",
        "late",
        "early",
        "first",
        "secon",
        "round",
        "seaso",
        "lawn",
        "apply",
        "check",
        "prevent",
        "annua",
    }
)


def _propose(assessment: Assessment, today: date) -> Proposal:
    item = assessment.item
    due = max(assessment.window_start, today)
    reason = assessment.reason.strip()
    # Not `.capitalize()` — that lowercases the rest of the string and
    # turns "Sep 15" into "sep 15".
    lines = [(reason[:1].upper() + reason[1:]).rstrip(".") + "."]
    if item.rationale:
        lines.extend(["", item.rationale.strip()])
    lines.extend(
        [
            "",
            f"Window: {assessment.window_start:%b %d} - {assessment.window_end:%b %d}.",
        ]
    )
    if item.gate is not None:
        lines.append(f"Gate: {assessment.gate_status.value}.")
    if isinstance(item.category, ChemicalCategory):
        lines.extend(
            [
                "",
                "Product and rate are deliberately not fixed here — ask "
                "lawn-agents for the cited recommendation before applying, "
                "and source what you don't have.",
            ]
        )
    return Proposal(
        item=item,
        assessment=assessment,
        title=item.name,
        description="\n".join(lines),
        due=due,
    )


def _find_overlap(item: ProgramItem, tasks: list[Task]) -> Overlap | None:
    """Spot a hand-written task that probably covers `item`.

    Intentionally fuzzy and intentionally advisory. Matching by content
    words is good enough to say "you may already have this" and nowhere
    near good enough to act on, which is why overlaps are only ever
    reported. `Apply Pre Emergnent` — misspelling and all — should be
    recognised as covering the fall pre-emergent without the system
    presuming to rewrite it.
    """
    item_words = _content_words(item.name) - _GENERIC
    if not item_words:
        return None
    for task in tasks:
        if task.owned:
            continue
        task_words = _content_words(task.content)
        # Generic words are stripped before comparing, so a single
        # shared stem is meaningful: "emerg" links "Fall pre-emergent" to
        # a hand-typed "Apply Pre Emergnent", misspelling and all.
        shared = (item_words & task_words) - _GENERIC
        if shared:
            return Overlap(
                task=task,
                item=item,
                why=f"shares {sorted(shared)} with {item.name!r}",
            )
    return None


def _content_words(text: str) -> set[str]:
    import re

    raw = re.findall(r"[a-z]+", text.lower())
    # Trim a trailing letter so light misspellings still match:
    # "emergnent" and "emergent" share the "emerg" stem.
    return {w[:5] for w in raw if len(w) > 3 and w not in _STOPWORDS}
