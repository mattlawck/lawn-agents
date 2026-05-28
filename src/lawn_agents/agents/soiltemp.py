"""USDA-NRCS AWDB (SCAN) soil-temperature client.

Finds the nearest SCAN station to the configured lat/lon and fetches
2-inch and 4-inch soil temperatures. Values are already in Fahrenheit
per the AWDB `storedUnitCode=degF`.

Phase 1 leaves the Parton/Logan modeled fallback unimplemented
(ADR 0004). When no SCAN station is within `MAX_STATION_RADIUS_MI`,
the snapshot returns with an explicit error and `current_*_f` left
`None`. The modeled fallback will ship as its own follow-up that
borrows NWS air-temperature history from `weather.py`.

API reference: https://wcc.sc.egov.usda.gov/awdbRestApi/swagger-ui/index.html
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from math import asin, cos, radians, sin, sqrt
from typing import TYPE_CHECKING, Any

import httpx

from lawn_agents.logging import get_logger
from lawn_agents.models import SoilSnapshot

if TYPE_CHECKING:
    from collections.abc import Callable

    from lawn_agents.config import AppConfig

AWDB_BASE_URL = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1"
MAX_STATION_RADIUS_MI = 100.0
HISTORY_DAYS = 7
DEPTH_2IN = -2
DEPTH_4IN = -4

log = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class _Station:
    """Subset of AWDB station metadata we care about."""

    triplet: str
    name: str
    latitude: float
    longitude: float
    distance_mi: float


def snapshot(config: AppConfig) -> SoilSnapshot | None:
    """Fetch current and trailing-7d soil temperature at 2" and 4" depth.

    Args:
        config: Application configuration (used for lat/lon and HTTP knobs).

    Returns:
        A populated `SoilSnapshot`, or `None` only if the HTTP client
        itself can't be built. Partial data is fine — missing values
        stay `None` and the cause lands in `errors`.
    """
    fetched_at = datetime.now(UTC)
    errors: list[str] = []

    try:
        client = _build_client(config)
    except Exception as exc:
        log.error("soiltemp.client_build_failed", error=str(exc))
        return None

    with client:
        station = _try(
            lambda: _find_nearest_scan_station(
                client,
                config.location.latitude,
                config.location.longitude,
                MAX_STATION_RADIUS_MI,
            ),
            "nearest SCAN station lookup",
            errors,
        )
        if station is None:
            errors.append(
                f"no SCAN station within {MAX_STATION_RADIUS_MI:.0f} mi of "
                f"({config.location.latitude}, {config.location.longitude}); "
                "Parton/Logan fallback not yet implemented (ADR 0004)"
            )
            return SoilSnapshot(fetched_at=fetched_at, errors=errors)

        log.info(
            "soiltemp.station_selected",
            triplet=station.triplet,
            name=station.name,
            distance_mi=round(station.distance_mi, 1),
        )

        payload = _try(
            lambda: _fetch_sto_data(client, station.triplet),
            "soil temp data fetch",
            errors,
        )

    parsed = _parse_data(payload)

    return SoilSnapshot(
        fetched_at=fetched_at,
        station_id=station.triplet,
        modeled=False,
        current_2in_f=parsed.temp_2in,
        current_4in_f=parsed.temp_4in,
        trailing_7d_4in_f=parsed.trail_temp_4in,
        current_2in_moisture_pct=parsed.moisture_2in,
        current_4in_moisture_pct=parsed.moisture_4in,
        trailing_7d_4in_moisture_pct=parsed.trail_moisture_4in,
        errors=errors,
    )


# --- internals -------------------------------------------------------------


def _build_user_agent(config: AppConfig) -> str:
    return f"lawn-agents (+https://github.com/mattlawck/lawn-agents, {config.http.contact_email})"


def _build_client(config: AppConfig) -> httpx.Client:
    return httpx.Client(
        base_url=AWDB_BASE_URL,
        timeout=config.http.timeout_seconds,
        headers={
            "User-Agent": _build_user_agent(config),
            "Accept": "application/json",
        },
    )


def _find_nearest_scan_station(
    client: httpx.Client,
    lat: float,
    lon: float,
    max_miles: float,
) -> _Station | None:
    """Return the closest SCAN station within `max_miles`, or `None`.

    The AWDB `/stations` endpoint's `networkCds` filter is unreliable,
    so we fetch the full station list and filter client-side. ~4400
    stations at ~150 bytes each is a trivially small response.
    """
    response = client.get("/stations", params={"networkCds": "SCAN"})
    response.raise_for_status()
    rows: list[dict[str, Any]] = response.json()

    best: _Station | None = None
    best_dist = float("inf")
    for row in rows:
        if row.get("networkCode") != "SCAN":
            continue
        s_lat = row.get("latitude")
        s_lon = row.get("longitude")
        if not isinstance(s_lat, int | float) or not isinstance(s_lon, int | float):
            continue
        dist = _haversine_miles(lat, lon, float(s_lat), float(s_lon))
        if dist < best_dist:
            best_dist = dist
            best = _Station(
                triplet=str(row["stationTriplet"]),
                name=str(row.get("name", "")),
                latitude=float(s_lat),
                longitude=float(s_lon),
                distance_mi=dist,
            )
    if best is None or best.distance_mi > max_miles:
        return None
    return best


@dataclass(slots=True)
class _ParsedSoilData:
    """Decomposed soil-data block: temps at 2/4 inch + moisture at 2/4 inch.

    Mutable so the parser can fill it in one block at a time without
    threading a separate accumulator type.
    """

    temp_2in: float | None = None
    temp_4in: float | None = None
    trail_temp_4in: list[float] = field(default_factory=list)
    moisture_2in: float | None = None
    moisture_4in: float | None = None
    trail_moisture_4in: list[float] = field(default_factory=list)


def _fetch_sto_data(client: httpx.Client, triplet: str) -> list[dict[str, Any]]:
    """Fetch daily STO + SMS values at 2" and 4" depth for the last week.

    AWDB element format: `elementCode:heightDepth:ordinal`. We request
    ordinal=1 (primary sensor) for both Soil Temperature Observed (STO)
    and Soil Moisture (SMS). SMS is reported as volumetric water content
    percent (~0..50).
    """
    end = date.today()
    start = end - timedelta(days=HISTORY_DAYS)
    response = client.get(
        "/data",
        params={
            "stationTriplets": triplet,
            "elements": (
                f"STO:{DEPTH_2IN}:1,STO:{DEPTH_4IN}:1,SMS:{DEPTH_2IN}:1,SMS:{DEPTH_4IN}:1"
            ),
            "duration": "DAILY",
            "beginDate": start.isoformat(),
            "endDate": end.isoformat(),
        },
    )
    response.raise_for_status()
    payload: list[dict[str, Any]] = response.json()
    return payload


def _parse_data(payload: list[dict[str, Any]] | None) -> _ParsedSoilData:
    """Pull out current 2"/4" temps + moisture and trailing-7d at 4" depth."""
    parsed = _ParsedSoilData()
    if not payload:
        return parsed
    for block in payload[0].get("data", []) or []:
        _apply_block(parsed, block)
    return parsed


def _apply_block(parsed: _ParsedSoilData, block: dict[str, Any]) -> None:
    """Fold one stationElement block into `parsed`.

    Extracted from `_parse_data` so cognitive complexity stays under
    SonarCloud's threshold — the per-element/depth chain is local here.
    """
    elem = block.get("stationElement", {}) or {}
    code = elem.get("elementCode")
    depth = elem.get("heightDepth")
    values = block.get("values", []) or []
    if not values:
        return

    latest = _latest_value(values)
    if code == "STO" and depth == DEPTH_2IN:
        parsed.temp_2in = latest
    elif code == "STO" and depth == DEPTH_4IN:
        parsed.temp_4in = latest
        parsed.trail_temp_4in = _ordered_values(values)[-HISTORY_DAYS:]
    elif code == "SMS" and depth == DEPTH_2IN:
        parsed.moisture_2in = latest
    elif code == "SMS" and depth == DEPTH_4IN:
        parsed.moisture_4in = latest
        parsed.trail_moisture_4in = _ordered_values(values)[-HISTORY_DAYS:]


def _latest_value(values: list[dict[str, Any]]) -> float | None:
    for entry in reversed(values):
        v = entry.get("value")
        if isinstance(v, int | float):
            return float(v)
    return None


def _ordered_values(values: list[dict[str, Any]]) -> list[float]:
    """Return float values in original (date-ordered) order, skipping nulls."""
    out: list[float] = []
    for entry in values:
        v = entry.get("value")
        if isinstance(v, int | float):
            out.append(float(v))
    return out


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles between two lat/lon pairs."""
    r_mi = 3958.8
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    return 2 * r_mi * asin(sqrt(a))


def _try[T](fn: Callable[[], T], label: str, errors: list[str]) -> T | None:
    """Run `fn`, return its result, or append a formatted error and return None."""
    try:
        return fn()
    except httpx.HTTPStatusError as exc:
        msg = f"{label} failed: HTTP {exc.response.status_code}"
        log.warning("soiltemp.http_error", label=label, status=exc.response.status_code)
    except httpx.HTTPError as exc:
        msg = f"{label} failed: {type(exc).__name__}: {exc}"
        log.warning("soiltemp.http_error", label=label, error=str(exc))
    except Exception as exc:
        # Boundary swallow: soiltemp.snapshot never raises past this module.
        msg = f"{label} failed: {type(exc).__name__}: {exc}"
        log.warning("soiltemp.unexpected_error", label=label, error=str(exc))
    errors.append(msg)
    return None
