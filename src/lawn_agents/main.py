"""CLI entry point: `lawn-agents --ask "..."`, `--scheduled`, `--plan-month`, etc."""

from __future__ import annotations

import argparse
import sys
from typing import Final

from lawn_agents import __version__
from lawn_agents.config import DEFAULT_CONFIG_PATH
from lawn_agents.logging import configure as configure_logging

EXIT_OK: Final = 0
EXIT_USAGE: Final = 2
EXIT_RUNTIME: Final = 1


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="lawn-agents",
        description="Multi-agent lawn-care advisory for Zeon Zoysia in coastal SC.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
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
    parser.parse_args(argv)
    # Phase 1 implementation lands in subsequent tasks; for now,
    # successful argparse is the only behavior under test.
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(cli())
