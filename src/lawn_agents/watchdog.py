"""The weekly unattended check — silent unless something needs you.

This is what launchd runs. Its job is *not* to produce advice into a log
nobody reads; it is to notice that a window is opening and pull the user
back into the conversation before they miss it.

Two properties define it:

**Silence is the default.** It speaks only when a `critical` item is
approaching and nothing already tracks it, or when an open task has gone
past due. A weekly digest that always says something gets muted within a
month, and a muted watchdog is worse than none — it creates the belief
that you'd have been told.

**No LLM.** Deciding whether soil temperature has crossed a threshold is
arithmetic, and the reasoning behind each item already lives in the
shell's `rationale`. So the unattended path costs nothing per run, adds
no latency, cannot hit a rate limit at 7am, and structurally cannot
hallucinate a recommendation while the user is asleep. Prose advice is
what `--ask` is for, and that is a conversation the user initiates.

The watchdog does write — it creates Todoist tasks without asking,
because "unattended" and "ask first" are incompatible. That is why it is
restricted to `critical` items: the interruption has to be worth it, and
a stray task is cheap to delete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from lawn_agents import reconcile, tasklog, todoist
from lawn_agents.agents import soiltemp
from lawn_agents.logging import get_logger

if TYPE_CHECKING:
    from datetime import date

    from lawn_agents.config import Settings
    from lawn_agents.models import SoilSnapshot
    from lawn_agents.reconcile import Plan
    from lawn_agents.todoist import Task, TodoistClient

log = get_logger(__name__)


@dataclass(slots=True)
class WatchResult:
    """What the run found and did."""

    plan: Plan | None = None
    created: list[Task] = field(default_factory=list)
    soil: SoilSnapshot | None = None
    skipped_reason: str | None = None
    completed: list[tasklog.SeenTask] = field(default_factory=list)
    """Owned tasks open last run and gone now — completed or deleted."""

    followups: list[tuple[tasklog.SeenTask, Any]] = field(default_factory=list)
    """Disappeared monitoring items whose result has nowhere else to live."""

    @property
    def spoke(self) -> bool:
        """True when the run had something worth surfacing."""
        return bool(
            self.created
            or self.followups
            or (self.plan and (self.plan.overdue_owned or self.plan.overdue_manual))
        )


def run(
    settings: Settings,
    *,
    today: date | None = None,
    client: TodoistClient | None = None,
    create: bool = True,
) -> WatchResult:
    """Evaluate the program and file tasks for anything critical and untracked.

    Args:
        settings: Validated settings.
        today: Evaluation date; defaults to the current UTC date.
        client: Injected Todoist client (tests).
        create: When False, evaluate and report without writing. Useful
            for a dry run and for tests.

    Returns:
        A `WatchResult`. `spoke` is False on a quiet week, which is the
        expected outcome most of the time.
    """
    today = today or datetime.now(UTC).date()

    if not settings.app.todoist.enabled:
        log.info("watchdog.disabled")
        return WatchResult(skipped_reason="todoist is not enabled in config")

    token = settings.todoist_api_token
    if token is None:
        log.warning("watchdog.no_token")
        return WatchResult(skipped_reason="TODOIST_API_TOKEN is not set")
    if not settings.app.todoist.project_id:
        log.warning("watchdog.no_project")
        return WatchResult(skipped_reason="todoist.project_id is not set in config.yaml")

    # Soil data is optional by design: the window half of every gate is
    # pure date arithmetic, so an unreachable station degrades the run to
    # "a window is opening" without silencing it.
    soil = soiltemp.snapshot(settings.app)
    if soil is None or soil.current_4in_f is None:
        log.info("watchdog.soil_unavailable")

    owns_client = client is None
    todo = client or todoist.TodoistClient(
        token.get_secret_value(), settings.app.todoist.project_id
    )
    try:
        # A transient network failure must not produce a stack trace in a
        # log nobody reads. The unattended path degrades to a stated
        # skip and tries again next week, the same way every data
        # fetcher in this project fails closed.
        try:
            tasks = todo.open_tasks()
        except todoist.TodoistError as exc:
            log.warning("watchdog.todoist_unavailable", error=str(exc))
            return WatchResult(soil=soil, skipped_reason=f"Todoist unreachable: {exc}")
        # Absence is the completion signal, but only against a memory of
        # what was there before.
        previous = tasklog.load(settings.app.todoist.state_file)
        gone = tasklog.disappeared(previous, tasks)
        followups = tasklog.needing_followup(gone, settings.program.items)
        plan = reconcile.reconcile(
            settings.program.items,
            tasks,
            settings.app.climate,
            today=today,
            soil=soil,
            # The whole point: routine work does not justify interrupting
            # someone who hasn't asked.
            urgent_only=True,
        )
        created: list[Task] = []
        try:
            if create and plan.proposals:
                created = reconcile.apply(plan, todo, label=settings.app.todoist.label)
        except todoist.TodoistError as exc:
            # Partial creation is fine: de-duplication is by marker, so
            # next week fills whatever didn't land.
            log.warning("watchdog.create_failed", error=str(exc), created=len(created))
        # Recorded regardless of `create`: the snapshot is an observation
        # of what was open when we looked, which is equally true on a dry
        # run. Gating it on writing would mean a dry run silently left the
        # memory stale and the next real run mis-read the difference.
        tasklog.save(settings.app.todoist.state_file, [*tasks, *created])
    finally:
        if owns_client:
            todo.close()

    result = WatchResult(plan=plan, created=created, soil=soil, completed=gone, followups=followups)
    log.info(
        "watchdog.done",
        spoke=result.spoke,
        created=len(created),
        overdue=len(plan.overdue_owned) + len(plan.overdue_manual),
        completed=len(gone),
        followups=len(followups),
    )
    return result


def render(result: WatchResult) -> str:
    """Human-readable summary, or empty string on a quiet week.

    Returning "" for a quiet run is what keeps launchd's log readable:
    weeks where nothing is due leave no output at all, so anything in the
    log is worth looking at.
    """
    if result.skipped_reason:
        return f"lawn-agents watchdog skipped: {result.skipped_reason}"
    if not result.spoke:
        return ""

    plan = result.plan
    assert plan is not None  # spoke implies a plan
    lines: list[str] = ["lawn-agents — something needs your attention", ""]

    if result.created:
        lines.append(f"Filed {len(result.created)} task(s):")
        lines.extend(f"  • {t.content}" for t in result.created)
        lines.append("")

    if result.followups:
        lines.append("You finished these — what did you find?")
        for seen, item in result.followups:
            lines.append(f"  • {seen.content}")
            if item.rationale:
                lines.append(f"      {_clip(item.rationale, 200)}")
        lines.append(
            "  (Completing an inspection records that you looked, not what "
            "you saw. Tell lawn-agents and it can act on it.)"
        )
        lines.append("")

    overdue = [*plan.overdue_owned, *plan.overdue_manual]
    if overdue:
        lines.append(f"Past due and still open ({len(overdue)}):")
        lines.extend(f"  • {t.due} — {t.content}" for t in overdue)
        lines.append("")

    # Only relevant when something was actually scheduled. On a run whose
    # sole content is "what did you find?", it reads as boilerplate.
    if result.created:
        lines.append("Ask lawn-agents for the cited recommendation before applying anything.")
    return "\n".join(lines).rstrip()


def _clip(text: str, limit: int) -> str:
    """Collapse whitespace and truncate on a word boundary."""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[:limit].rsplit(" ", 1)[0] + "…"
