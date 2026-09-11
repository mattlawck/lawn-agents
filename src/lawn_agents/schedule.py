"""Evaluate the standing program against today's date and conditions.

Turns `data/calendar.yaml` from a document into a decision: for each
recurring item, is it too early, worth preparing for, actionable right
now, blocked on a measurement, or already past?

Two independent signals, deliberately kept separate:

- **The window** is calendar arithmetic. It needs no live data, so it
  still works when the soil-temperature station is down — which it
  frequently is. This is what drives the "start sourcing product"
  nudge.
- **The gate** is a measured condition (4-inch soil temperature rising
  or falling through a threshold, sustained). It needs live data and
  degrades to `UNKNOWN` without it.

Keeping them apart is what lets the weekly watchdog stay useful during
an outage: it can still say "fall pre-emergent opens in two weeks, go
buy it" even when it can't yet say "conditions say go."
"""

from __future__ import annotations

from datetime import date, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, NamedTuple

from lawn_agents.logging import get_logger

if TYPE_CHECKING:
    from lawn_agents.config import ClimateConfig
    from lawn_agents.models import ProgramGate, ProgramItem, SoilSnapshot

log = get_logger(__name__)


class GateStatus(StrEnum):
    """Whether the measured condition authorizing an item is met."""

    SATISFIED = "satisfied"
    UNMET = "unmet"
    UNKNOWN = "unknown"
    """No gate data available — the station was unreachable, or the
    trailing history is shorter than `sustained_days`."""

    NONE = "none"
    """The item has no gate; the window alone governs."""


class ItemStatus(StrEnum):
    """What the program wants done with an item today."""

    NOT_YET = "not_yet"
    APPROACHING = "approaching"
    """Inside `lead_days` of the window opening — time to source product
    and make a plan, not to apply."""

    READY = "ready"
    """Window is open and the gate is satisfied (or there is none)."""

    WAITING_ON_GATE = "waiting_on_gate"
    """Window is open but conditions do not authorize action yet."""

    CLOSING = "closing"
    """Window is open and ends within a week."""

    PASSED = "passed"


class Assessment(NamedTuple):
    """One item's verdict for a given day."""

    item: ProgramItem
    status: ItemStatus
    gate_status: GateStatus
    window_start: date
    window_end: date
    reason: str

    @property
    def actionable(self) -> bool:
        """True when the item warrants surfacing to the user."""
        return self.status in {
            ItemStatus.APPROACHING,
            ItemStatus.READY,
            ItemStatus.WAITING_ON_GATE,
            ItemStatus.CLOSING,
        }

    @property
    def suggested_due(self) -> date:
        """When a task for this item should fall due."""
        return max(self.window_start, date.today())


CLOSING_SOON_DAYS = 7


def evaluate(
    item: ProgramItem,
    climate: ClimateConfig,
    *,
    today: date,
    soil: SoilSnapshot | None = None,
) -> Assessment:
    """Assess one program item against a date and current conditions."""
    start, end = window_dates(item, today)
    gate_status, gate_reason = _evaluate_gate(item.gate, climate, soil)

    if today < start - timedelta(days=item.lead_days):
        return Assessment(
            item,
            ItemStatus.NOT_YET,
            gate_status,
            start,
            end,
            f"window opens {start:%b %d}",
        )
    if today < start:
        days = (start - today).days
        return Assessment(
            item,
            ItemStatus.APPROACHING,
            gate_status,
            start,
            end,
            f"window opens in {days} day{'s' if days != 1 else ''} ({start:%b %d})",
        )
    if today > end:
        return Assessment(
            item,
            ItemStatus.PASSED,
            gate_status,
            start,
            end,
            f"window closed {end:%b %d}",
        )

    # Inside the window.
    if gate_status is GateStatus.UNMET:
        return Assessment(item, ItemStatus.WAITING_ON_GATE, gate_status, start, end, gate_reason)
    if gate_status is GateStatus.UNKNOWN:
        return Assessment(
            item,
            ItemStatus.WAITING_ON_GATE,
            gate_status,
            start,
            end,
            f"{gate_reason}; cannot confirm conditions",
        )
    if (end - today).days <= CLOSING_SOON_DAYS:
        return Assessment(
            item,
            ItemStatus.CLOSING,
            gate_status,
            start,
            end,
            f"window closes {end:%b %d} — {(end - today).days} days left",
        )
    return Assessment(
        item,
        ItemStatus.READY,
        gate_status,
        start,
        end,
        gate_reason if item.gate else f"window open through {end:%b %d}",
    )


def evaluate_all(
    items: list[ProgramItem],
    climate: ClimateConfig,
    *,
    today: date,
    soil: SoilSnapshot | None = None,
) -> list[Assessment]:
    """Assess every item, most urgent first."""
    order = {
        ItemStatus.READY: 0,
        ItemStatus.CLOSING: 1,
        ItemStatus.WAITING_ON_GATE: 2,
        ItemStatus.APPROACHING: 3,
        ItemStatus.PASSED: 4,
        ItemStatus.NOT_YET: 5,
    }
    assessments = [evaluate(i, climate, today=today, soil=soil) for i in items]
    return sorted(assessments, key=lambda a: (order[a.status], a.window_start))


def window_dates(item: ProgramItem, today: date) -> tuple[date, date]:
    """Resolve the item's month/day window to concrete dates near `today`.

    Windows repeat annually and may wrap the year end (e.g. Dec 1 to
    Feb 15). For a wrapping window the pair straddles New Year, anchored
    so `today` falls inside it where possible — otherwise a January date
    would resolve against a window that already closed the previous year.
    """
    w = item.window
    start = _safe_date(today.year, w.start_month, w.start_day)
    end = _safe_date(today.year, w.end_month, w.end_day)

    if start <= end:
        return start, end

    # Wrapping window: pick the occurrence containing or following today.
    if today <= end:
        return _safe_date(today.year - 1, w.start_month, w.start_day), end
    return start, _safe_date(today.year + 1, w.end_month, w.end_day)


# --- internals ------------------------------------------------------------


def _safe_date(year: int, month: int, day: int) -> date:
    """Build a date, clamping the day to the month's length.

    `end_day: 31` is a natural way to write "end of the month" in the
    YAML, and it must not explode in February.
    """
    import calendar

    return date(year, month, min(day, calendar.monthrange(year, month)[1]))


def _evaluate_gate(
    gate: ProgramGate | None,
    climate: ClimateConfig,
    soil: SoilSnapshot | None,
) -> tuple[GateStatus, str]:
    if gate is None:
        return GateStatus.NONE, ""

    threshold = getattr(climate, gate.threshold_ref, None)
    if threshold is None:
        # Settings validation rejects unknown refs at load, so reaching
        # here means the config changed underneath us.
        log.warning("schedule.unknown_threshold_ref", ref=gate.threshold_ref)
        return GateStatus.UNKNOWN, f"threshold {gate.threshold_ref!r} not found"

    if soil is None or not soil.trailing_7d_4in_f:
        return GateStatus.UNKNOWN, "4-inch soil temperature unavailable"

    recent = soil.trailing_7d_4in_f[-gate.sustained_days :]
    if len(recent) < gate.sustained_days:
        return (
            GateStatus.UNKNOWN,
            f"need {gate.sustained_days} days of soil temperature, have {len(recent)}",
        )

    rising = gate.direction.value == "rising"
    met = all(v >= threshold for v in recent) if rising else all(v <= threshold for v in recent)
    arrow = "at or above" if rising else "at or below"
    latest = recent[-1]
    if met:
        return (
            GateStatus.SATISFIED,
            f"4-inch soil temp {arrow} {threshold}F for {gate.sustained_days} days "
            f"(latest {latest:.0f}F)",
        )
    return (
        GateStatus.UNMET,
        f"4-inch soil temp not yet {arrow} {threshold}F for "
        f"{gate.sustained_days} consecutive days (latest {latest:.0f}F)",
    )
