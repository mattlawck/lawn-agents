"""CLI entry point: `lawn-agents --ask "..."`, `--scheduled`, `--plan-month`, etc."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

from lawn_agents import __version__, notify, orchestrator, planner
from lawn_agents.config import DEFAULT_CONFIG_PATH, Settings
from lawn_agents.logging import configure as configure_logging
from lawn_agents.logging import get_logger

EXIT_OK: Final = 0
EXIT_USAGE: Final = 2
EXIT_RUNTIME: Final = 1

log = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="lawn-agents",
        description="Multi-agent lawn-care advisory for Zeon Zoysia in coastal SC.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.yaml (default: ./config.yaml).",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ask", type=str, metavar="QUESTION", help="Ad-hoc question mode.")
    mode.add_argument(
        "--scheduled",
        action="store_true",
        help="Run the weekly scheduled check (used by launchd).",
    )
    mode.add_argument(
        "--plan-month",
        type=str,
        metavar="YYYY-MM",
        help="Generate a plan for the named month.",
    )
    mode.add_argument(
        "--plan-year",
        type=int,
        metavar="YYYY",
        help="Generate a forward-looking plan for the named year.",
    )
    mode.add_argument(
        "--review-additions",
        action="store_true",
        help="Interactively promote/reject auto-ingested research passages.",
    )

    return parser


def cli(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code.

    Args:
        argv: Optional argv override (used in tests). When `None`, reads
            from `sys.argv[1:]`.

    Returns:
        `EXIT_OK` on success, `EXIT_USAGE` on argument errors, or
        `EXIT_RUNTIME` on runtime failures.
    """
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        settings = Settings.load(args.config)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except Exception as exc:
        print(f"error loading config: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_USAGE

    sinks = notify.build_sinks(settings.app)

    try:
        return _dispatch(args, settings, sinks)
    except Exception as exc:
        log.error("cli.unexpected_failure", error=str(exc), error_type=type(exc).__name__)
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_RUNTIME


def _dispatch(
    args: argparse.Namespace,
    settings: Settings,
    sinks: list[notify.Sink],
) -> int:
    if args.ask:
        rec = orchestrator.answer(args.ask, settings)
    elif args.scheduled:
        rec = orchestrator.scheduled_check(settings)
    elif args.plan_month is not None:
        try:
            year, month = _parse_yyyy_mm(args.plan_month)
        except ValueError as exc:
            print(f"error: --plan-month {exc}", file=sys.stderr)
            return EXIT_USAGE
        rec = planner.plan_month(year, month, settings)
    elif args.plan_year is not None:
        rec = planner.plan_year(args.plan_year, settings)
    elif args.review_additions:
        print(
            "--review-additions: not yet implemented "
            "(lands with the research subagent PR — task #16).",
            file=sys.stderr,
        )
        return EXIT_RUNTIME
    else:  # pragma: no cover — argparse enforces required mutually-exclusive group
        return EXIT_USAGE

    for sink in sinks:
        sink.emit(rec)
    return EXIT_RUNTIME if rec.refused else EXIT_OK


def _parse_yyyy_mm(raw: str) -> tuple[int, int]:
    """Parse a `YYYY-MM` string into a `(year, month)` tuple."""
    parts = raw.split("-")
    if len(parts) != 2:
        msg = f"expected YYYY-MM, got {raw!r}"
        raise ValueError(msg)
    try:
        year = int(parts[0])
        month = int(parts[1])
    except ValueError as exc:
        msg = f"expected numeric YYYY-MM, got {raw!r}"
        raise ValueError(msg) from exc
    if not 1 <= month <= 12:
        msg = f"month must be 1..12, got {month}"
        raise ValueError(msg)
    return year, month


if __name__ == "__main__":  # pragma: no cover
    sys.exit(cli())
