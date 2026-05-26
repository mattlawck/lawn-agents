"""CLI: ingest `data/corpus/*.pdf` + `seed_urls:` into the LanceDB index.

Run after editing `data/corpus/` or `seed_urls:` in `config.yaml`.
Idempotent — re-running with an unchanged source set is a no-op
because `Chunk.id` is content-addressed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lawn_agents.config import DEFAULT_CONFIG_PATH, Settings
from lawn_agents.ingest import IngestReport, ingest
from lawn_agents.logging import configure as configure_logging


def main() -> int:
    """Entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="lawn-agents-ingest",
        description=(
            "Ingest PDFs from data/corpus/ and seed URLs from config.yaml "
            "into the local LanceDB knowledge index."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.yaml (default: ./config.yaml).",
    )
    args = parser.parse_args()

    configure_logging()
    settings = Settings.load(args.config)
    report = ingest(settings.app)
    _print_report(report)
    return 0 if not report.sources_failed else 1


def _print_report(report: IngestReport) -> None:
    print(f"Sources seen:     {report.sources_seen}")
    print(f"Sources OK:       {report.sources_ok}")
    print(f"Sources failed:   {len(report.sources_failed)}")
    for src, reason in report.sources_failed:
        print(f"  - {src}: {reason}")
    print(f"Chunks added:     {report.chunks_added}")
    print(
        f"Store totals:     {report.chunks_total_after} chunks (was {report.chunks_total_before})"
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
