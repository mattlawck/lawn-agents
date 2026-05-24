"""CLI: ingest `data/corpus/*.pdf` + `seed_urls:` into the LanceDB index.

Run after editing `data/corpus/` or `seed_urls:` in `config.yaml`.
Idempotent — re-running with an unchanged source set is a no-op.
"""

from __future__ import annotations

import sys


def main() -> int:
    """Entry point. Returns a process exit code."""
    raise NotImplementedError


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
