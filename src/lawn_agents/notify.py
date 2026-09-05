"""Output sinks — Phase 1 renders to the console; Phase 3 adds email/SMS.

The `Sink` Protocol keeps the orchestrator decoupled from any specific
output medium. Adding a new sink is a single class plus an entry in
`config.yaml > notify.sinks`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Protocol

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from lawn_agents.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date
    from typing import TextIO

    from lawn_agents.config import AppConfig, SinkName
    from lawn_agents.models import CalendarItem, Citation, Recommendation

log = get_logger(__name__)


class Sink(Protocol):
    """Anything that can render a `Recommendation` to some medium."""

    def emit(self, recommendation: Recommendation) -> None:
        """Render the recommendation. Must never raise."""


class ConsoleSink:
    """Pretty-prints a `Recommendation` via `rich`.

    Refusals are surfaced with a clear panel; normal recommendations
    show the headline, conditions summary, weekly + monthly actions
    with inline citation markers, and a numbered Sources block at the
    end so users can verify every chemical claim.
    """

    def __init__(self, *, file: TextIO | None = None) -> None:
        """Build a console.

        Args:
            file: Optional file-like sink for the rich `Console`. Tests
                pass `io.StringIO`; production leaves it None and rich
                writes to stdout.
        """
        self._console = Console(file=file, force_terminal=False, record=False)

    def emit(self, recommendation: Recommendation) -> None:
        """See `Sink.emit`."""
        try:
            if recommendation.refused:
                self._emit_refusal(recommendation)
            else:
                self._emit_recommendation(recommendation)
        except Exception as exc:  # pragma: no cover — boundary swallow
            log.error("notify.console.render_failed", error=str(exc))

    # --- internals ---

    def _emit_refusal(self, rec: Recommendation) -> None:
        body = rec.refusal_reason or "(no reason provided)"
        self._console.print(
            Panel(
                body,
                title="[red]Refused[/red]",
                title_align="left",
                border_style="red",
            )
        )

    def _emit_recommendation(self, rec: Recommendation) -> None:
        self._console.print(
            Panel(
                rec.headline,
                title="[green]Recommendation[/green]",
                title_align="left",
                border_style="green",
            )
        )
        if rec.conditions_summary:
            self._console.print(f"\n[bold]Conditions:[/bold] {rec.conditions_summary}")

        citations_by_index: dict[str, int] = {}
        self._emit_action_section("This week", rec.weekly_actions, citations_by_index)
        self._emit_action_section(
            _monthly_heading(rec.monthly_actions), rec.monthly_actions, citations_by_index
        )

        if rec.notes:
            self._console.print("\n[bold]Notes:[/bold]")
            for note in rec.notes:
                self._console.print(f"  • {note}")

        if citations_by_index:
            self._console.print("\n[bold]Sources:[/bold]")
            for citation, idx in sorted(citations_by_index.items(), key=lambda kv: kv[1]):
                self._console.print(f"  [{idx}] {citation}")

    def _emit_action_section(
        self,
        heading: str,
        actions: Sequence[CalendarItem],
        citations_by_index: dict[str, int],
    ) -> None:
        if not actions:
            return
        self._console.print(f"\n[bold]{heading}:[/bold]")
        for item in actions:
            markers = self._record_citations(item.citations, citations_by_index)
            window = _format_window(item.earliest, item.latest)
            conditional = (
                f"\n      [dim]gated by:[/dim] {item.conditional}" if item.conditional else ""
            )
            # rich treats `[word]` as a style tag, so render the category
            # outside markup with an em-dash separator.
            line = f"  • {item.category.value} — {item.action}{window}{markers}{conditional}"
            self._console.print(Markdown(line) if "**" in line else line)

    def _record_citations(
        self,
        citations: Sequence[Citation],
        citations_by_index: dict[str, int],
    ) -> str:
        if not citations:
            return ""
        # An item often cites the same document several times (different
        # snippets from one factsheet). They collapse to one entry in the
        # Sources block, so repeating the marker renders as "[1] [1] [1]"
        # — noise that reads like three separate sources.
        seen: set[int] = set()
        markers: list[str] = []
        for c in citations:
            key = _citation_key(c)
            if key not in citations_by_index:
                citations_by_index[key] = len(citations_by_index) + 1
            idx = citations_by_index[key]
            if idx not in seen:
                seen.add(idx)
                markers.append(f"[{idx}]")
        return " " + " ".join(markers)


def build_sinks(config: AppConfig) -> list[Sink]:
    """Construct the list of sinks named in `config.notify.sinks`.

    Args:
        config: Application configuration.

    Returns:
        Ordered list of `Sink` instances. Phase 1 wires `console` only;
        unknown sink names log a warning and are skipped so a typo in
        config doesn't crash the run.
    """
    sinks: list[Sink] = []
    for name in config.notify.sinks:
        sink = _build_one(name)
        if sink is not None:
            sinks.append(sink)
    if not sinks:
        # Always have something so the user sees output. Default to console.
        sinks.append(ConsoleSink())
    return sinks


def _build_one(name: SinkName) -> Sink | None:
    if name == "console":
        return ConsoleSink()
    log.warning("notify.unknown_sink", name=name)
    return None


# --- helpers --------------------------------------------------------------


def _citation_key(c: Citation) -> str:
    """Stable key for de-duping citations in the Sources block."""
    parts: list[str] = [c.source_title]
    if c.page is not None:
        parts.append(f"p.{c.page}")
    if c.url:
        parts.append(c.url)
    if c.auto_researched:
        parts.append("[auto-researched, unreviewed]")
    return " — ".join(parts)


MONTHLY_HORIZON_DAYS = 31
"""Past this many days out, `monthly_actions` is no longer 'this month'."""


def _monthly_heading(actions: Sequence[CalendarItem], *, today: date | None = None) -> str:
    """Label for the `monthly_actions` section, based on how far it reaches.

    `monthly_actions` is really "everything that isn't this week," and
    a target-date question can fill it with a schedule running months
    out. The 2026-09-02 acceptance run planned to a July 2027 birthday
    and printed all nine items under a flat "This month:" header.

    So the heading follows the data: if any item is dated beyond
    `MONTHLY_HORIZON_DAYS`, the section is a schedule, not a month.
    Undated items keep the original label — that's the ad-hoc case
    where "this month" was accurate all along.
    """
    ref = today or datetime.now(UTC).date()
    horizon = ref + timedelta(days=MONTHLY_HORIZON_DAYS)
    for item in actions:
        if any(d is not None and d > horizon for d in (item.earliest, item.latest)):
            return "Planned schedule"
    return "This month"


def _format_window(earliest: object, latest: object) -> str:
    if earliest is None and latest is None:
        return ""
    if earliest is not None and latest is not None:
        return f"  [dim]({earliest} → {latest})[/dim]"
    if earliest is not None:
        return f"  [dim](on/after {earliest})[/dim]"
    return f"  [dim](by {latest})[/dim]"
