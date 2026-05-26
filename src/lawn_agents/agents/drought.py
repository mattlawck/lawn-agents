"""US Drought Monitor client.

Fetches the current weekly D-level for the configured county FIPS via
the USDM CountyStatistics REST API. The API returns CSV by default;
we ask for JSON with `Accept: application/json`.

Phase 1 ships D-level only. NOAA CPC seasonal outlook (1-month and
3-month temperature / precipitation probability) integration is
deferred to a follow-up — the CPC outlooks aren't directly queryable
by point, and the annual planner already gets enough signal from the
current D-level plus RAG-derived climatological norms to make useful
recommendations.

`DroughtSnapshot.d_level` semantics:
- -1: no drought present (all D-fields are 0)
-  0: D0 — abnormally dry
-  1: D1 — moderate drought
-  2: D2 — severe drought
-  3: D3 — extreme drought
-  4: D4 — exceptional drought

The level reported is the highest D-class with any non-zero area
coverage in the county. For lawn-care decisions this is the
conservative choice — even a small fraction of the county at D3 is
worth flagging.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx

from lawn_agents.logging import get_logger
from lawn_agents.models import DroughtSnapshot

if TYPE_CHECKING:
    from collections.abc import Callable

    from lawn_agents.config import AppConfig

USDM_BASE_URL = "https://usdmdataservices.unl.edu/api"
LOOKBACK_WEEKS = 2  # USDM updates weekly; 2-week window guarantees a row

log = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class _CountyRow:
    """Pulled-apart USDM CountyStatistics row."""

    valid_date: date | None
    d_levels: tuple[float, float, float, float, float]  # d0..d4 as fractions

    def max_nonzero_level(self) -> int:
        """Highest D-class (0..4) with any non-zero coverage; -1 if none."""
        max_level = -1
        for idx, pct in enumerate(self.d_levels):
            if pct > 0:
                max_level = idx
        return max_level


def snapshot(config: AppConfig) -> DroughtSnapshot | None:
    """Fetch the current drought classification for the configured county.

    Args:
        config: Application configuration (uses `location.county_fips`).

    Returns:
        A populated `DroughtSnapshot`. Partial data is fine — fields
        that couldn't be fetched are `None` and the failure lands in
        `errors`. Returns `None` only when the HTTP client itself
        can't be built.
    """
    fetched_at = datetime.now(UTC)
    fips = config.location.county_fips
    errors: list[str] = []

    try:
        client = _build_client(config)
    except Exception as exc:
        log.error("drought.client_build_failed", error=str(exc))
        return None

    with client:
        rows = _try(
            lambda: _fetch_recent_county_stats(client, fips, weeks=LOOKBACK_WEEKS),
            "USDM CountyStatistics fetch",
            errors,
        )

    if not rows:
        if not errors:
            errors.append(f"USDM returned no rows for FIPS {fips}")
        return DroughtSnapshot(fetched_at=fetched_at, county_fips=fips, errors=errors)

    # API returns most-recent week first.
    most_recent = _parse_row(rows[0])
    return DroughtSnapshot(
        fetched_at=fetched_at,
        county_fips=fips,
        valid_date=most_recent.valid_date,
        d_level=most_recent.max_nonzero_level(),
        errors=errors,
    )


# --- internals -------------------------------------------------------------


def _build_user_agent(config: AppConfig) -> str:
    return f"lawn-agents (+https://github.com/mattlawck/lawn-agents, {config.http.contact_email})"


def _build_client(config: AppConfig) -> httpx.Client:
    return httpx.Client(
        base_url=USDM_BASE_URL,
        timeout=config.http.timeout_seconds,
        headers={
            "User-Agent": _build_user_agent(config),
            "Accept": "application/json",
        },
    )


def _fetch_recent_county_stats(
    client: httpx.Client, fips: str, *, weeks: int
) -> list[dict[str, Any]]:
    """Fetch CountyStatistics for the last `weeks` weeks (most-recent first)."""
    end = date.today()
    start = end - timedelta(weeks=weeks)
    response = client.get(
        "/CountyStatistics/GetDroughtSeverityStatisticsByAreaPercent",
        params={
            # USDM expects M/D/YYYY date strings.
            "aoi": fips,
            "startdate": start.strftime("%m/%d/%Y"),
            "enddate": end.strftime("%m/%d/%Y"),
            "statisticsType": 1,
        },
    )
    response.raise_for_status()
    payload: list[dict[str, Any]] = response.json()
    return payload


def _parse_row(row: dict[str, Any]) -> _CountyRow:
    """Coerce a USDM JSON row into a typed `_CountyRow`."""
    d_levels = tuple(_pct(row, key) for key in ("d0", "d1", "d2", "d3", "d4"))
    return _CountyRow(
        valid_date=_parse_iso_date(row.get("validStart")),
        d_levels=d_levels,  # type: ignore[arg-type]
    )


def _pct(row: dict[str, Any], key: str) -> float:
    v = row.get(key)
    return float(v) if isinstance(v, int | float) else 0.0


def _parse_iso_date(raw: object) -> date | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw.rstrip("Z")).date()
    except ValueError:
        return None


def _try[T](fn: Callable[[], T], label: str, errors: list[str]) -> T | None:
    """Run `fn`, return its result, or append a formatted error and return None."""
    try:
        return fn()
    except httpx.HTTPStatusError as exc:
        msg = f"{label} failed: HTTP {exc.response.status_code}"
        log.warning("drought.http_error", label=label, status=exc.response.status_code)
    except httpx.HTTPError as exc:
        msg = f"{label} failed: {type(exc).__name__}: {exc}"
        log.warning("drought.http_error", label=label, error=str(exc))
    except Exception as exc:
        # Boundary swallow: drought.snapshot never raises past this module.
        msg = f"{label} failed: {type(exc).__name__}: {exc}"
        log.warning("drought.unexpected_error", label=label, error=str(exc))
    errors.append(msg)
    return None
