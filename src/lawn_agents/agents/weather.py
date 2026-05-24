"""NWS api.weather.gov client.

Resolves the configured lat/lon to a gridpoint, fetches the 7-day
forecast and the latest observations from the nearest station, and
returns a `WeatherSnapshot` or `None` on failure.

NWS asks for a descriptive User-Agent with contact info. Every fetch is
wrapped in try/except; failures are recorded in `WeatherSnapshot.errors`
and the corresponding fields stay `None`. The synthesizer reads the
errors list so it can degrade gracefully ("forecast unavailable today,
recommending based on N-day history instead").
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx

from lawn_agents.logging import get_logger
from lawn_agents.models import WeatherSnapshot

if TYPE_CHECKING:
    from collections.abc import Callable

    from lawn_agents.config import AppConfig

# Callable thunk used by `_try` for uniform exception handling.
type _Thunk[T] = "Callable[[], T]"

NWS_BASE_URL = "https://api.weather.gov"
FROST_THRESHOLD_F = 32.0
FORECAST_DAYS = 7

log = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class _Gridpoint:
    """Cached NWS gridpoint resolution for a lat/lon."""

    forecast_url: str
    observation_stations_url: str


def snapshot(config: AppConfig) -> WeatherSnapshot | None:
    """Fetch current weather + 7-day forecast for the configured location.

    Args:
        config: Application configuration (used for lat/lon and HTTP knobs).

    Returns:
        A populated `WeatherSnapshot`. Partial data is fine — fields that
        couldn't be fetched are `None` and the failure is appended to
        `errors`. Returns `None` only when the HTTP client itself can't
        be built (configuration error).
    """
    fetched_at = datetime.now(UTC)
    errors: list[str] = []

    try:
        client = _build_client(config)
    except Exception as exc:
        log.error("weather.client_build_failed", error=str(exc))
        return None

    with client:
        gridpoint = _try(
            lambda: _resolve_gridpoint(client, config.location.latitude, config.location.longitude),
            "gridpoint resolution",
            errors,
        )
        if gridpoint is None:
            return WeatherSnapshot(fetched_at=fetched_at, errors=errors)

        forecast_data = _try(
            lambda: _fetch_json(client, gridpoint.forecast_url),
            "forecast fetch",
            errors,
        )

        observation = _try(
            lambda: _fetch_latest_observation(client, gridpoint),
            "observation fetch",
            errors,
        )

        precip_24h_in: float | None = None
        if observation is not None:
            precip_24h_in = _try(
                lambda: _fetch_precip_window_in(client, observation.station_id, hours=24),
                "24h precip fetch",
                errors,
            )

    highs, lows, pop = _parse_forecast(forecast_data)
    frost_risk = any(t is not None and t <= FROST_THRESHOLD_F for t in lows)
    current_temp = observation.current_temp_f if observation is not None else None
    station_id = observation.station_id if observation is not None else None

    return WeatherSnapshot(
        fetched_at=fetched_at,
        station_id=station_id,
        current_temp_f=current_temp,
        last_24h_precip_in=precip_24h_in,
        last_7d_precip_in=None,
        forecast_high_7d_f=highs,
        forecast_low_7d_f=lows,
        forecast_precip_chance_7d=pop,
        frost_risk_next_7d=frost_risk,
        errors=errors,
    )


# --- internals -------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class _Observation:
    station_id: str
    current_temp_f: float | None


def _build_user_agent(config: AppConfig) -> str:
    """Build the descriptive User-Agent NWS requests."""
    return f"(lawn-agents +https://github.com/mattlawck/lawn-agents, {config.http.contact_email})"


def _build_client(config: AppConfig) -> httpx.Client:
    return httpx.Client(
        base_url=NWS_BASE_URL,
        timeout=config.http.timeout_seconds,
        headers={
            "User-Agent": _build_user_agent(config),
            "Accept": "application/geo+json",
        },
    )


def _fetch_json(client: httpx.Client, url: str) -> dict[str, Any]:
    response = client.get(url)
    response.raise_for_status()
    data: dict[str, Any] = response.json()
    return data


def _resolve_gridpoint(client: httpx.Client, lat: float, lon: float) -> _Gridpoint:
    data = _fetch_json(client, f"/points/{lat},{lon}")
    props = data["properties"]
    return _Gridpoint(
        forecast_url=props["forecast"],
        observation_stations_url=props["observationStations"],
    )


def _fetch_latest_observation(client: httpx.Client, gridpoint: _Gridpoint) -> _Observation | None:
    stations_data = _fetch_json(client, gridpoint.observation_stations_url)
    features = stations_data.get("features", [])
    if not features:
        return None

    station_id = str(features[0]["properties"]["stationIdentifier"])
    obs_data = _fetch_json(client, f"/stations/{station_id}/observations/latest")
    temp_props = obs_data["properties"].get("temperature", {}) or {}
    temp_value = temp_props.get("value")
    current_f = _c_to_f(temp_value) if isinstance(temp_value, (int, float)) else None
    return _Observation(station_id=station_id, current_temp_f=current_f)


def _fetch_precip_window_in(client: httpx.Client, station_id: str, *, hours: int) -> float | None:
    end = datetime.now(UTC)
    start = end - timedelta(hours=hours)
    response = client.get(
        f"/stations/{station_id}/observations",
        params={"start": start.isoformat(), "end": end.isoformat()},
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    total_mm = 0.0
    seen_any = False
    for feature in payload.get("features", []):
        props = feature.get("properties", {})
        precip = props.get("precipitationLastHour") or {}
        value = precip.get("value")
        if isinstance(value, int | float):
            total_mm += value
            seen_any = True
    if not seen_any:
        return None
    return _mm_to_in(total_mm)


def _parse_forecast(
    forecast_data: dict[str, Any] | None,
) -> tuple[list[float], list[float], list[float]]:
    """Return (highs, lows, precip-chance) lists of length up to FORECAST_DAYS."""
    if forecast_data is None:
        return [], [], []
    periods = forecast_data.get("properties", {}).get("periods", [])
    if not periods:
        return [], [], []

    by_date_high: dict[str, float] = defaultdict(lambda: float("-inf"))
    by_date_low: dict[str, float] = defaultdict(lambda: float("inf"))
    by_date_pop: dict[str, float] = defaultdict(float)
    seen_pop: dict[str, bool] = defaultdict(bool)
    ordered_dates: list[str] = []

    for period in periods:
        start = period.get("startTime")
        temp = period.get("temperature")
        unit = period.get("temperatureUnit", "F")
        if not isinstance(start, str) or not isinstance(temp, int | float):
            continue
        date_key = start[:10]
        if date_key not in ordered_dates:
            ordered_dates.append(date_key)

        temp_f = float(temp) if unit == "F" else _c_to_f(float(temp))
        if period.get("isDaytime"):
            by_date_high[date_key] = max(by_date_high[date_key], temp_f)
        else:
            by_date_low[date_key] = min(by_date_low[date_key], temp_f)

        pop_value = (period.get("probabilityOfPrecipitation") or {}).get("value")
        if isinstance(pop_value, int | float):
            # Take the day's max so a 70% pop window isn't washed out by an
            # adjacent 0% window in the same date bucket.
            by_date_pop[date_key] = max(by_date_pop[date_key], float(pop_value))
            seen_pop[date_key] = True

    highs: list[float] = []
    lows: list[float] = []
    pop: list[float] = []
    for date_key in ordered_dates[:FORECAST_DAYS]:
        high = by_date_high.get(date_key, float("-inf"))
        low = by_date_low.get(date_key, float("inf"))
        if high != float("-inf"):
            highs.append(round(high, 1))
        if low != float("inf"):
            lows.append(round(low, 1))
        if seen_pop.get(date_key):
            pop.append(round(by_date_pop[date_key], 1))
    return highs, lows, pop


def _try[T](fn: _Thunk[T], label: str, errors: list[str]) -> T | None:
    """Run `fn`, return its result, or append a formatted error and return None."""
    try:
        return fn()
    except httpx.HTTPStatusError as exc:
        msg = f"{label} failed: HTTP {exc.response.status_code}"
        log.warning("weather.http_error", label=label, status=exc.response.status_code)
    except httpx.HTTPError as exc:
        msg = f"{label} failed: {type(exc).__name__}: {exc}"
        log.warning("weather.http_error", label=label, error=str(exc))
    except Exception as exc:
        # Boundary swallow: weather.py never raises past its public functions.
        # The failure is recorded in `errors` so the synthesizer can degrade.
        msg = f"{label} failed: {type(exc).__name__}: {exc}"
        log.warning("weather.unexpected_error", label=label, error=str(exc))
    errors.append(msg)
    return None


def _c_to_f(celsius: float) -> float:
    return celsius * 9.0 / 5.0 + 32.0


def _mm_to_in(mm: float) -> float:
    return mm / 25.4
