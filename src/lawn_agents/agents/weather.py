"""NWS api.weather.gov client.

Resolves the configured lat/lon to a gridpoint and fetches the 7-day
forecast, hourly forecast, and recent observations. Returns a
`WeatherSnapshot` or `None` on failure.

NWS asks for a descriptive User-Agent with contact info. See ADR 0001
and the `http.contact_email` field in `config.yaml`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lawn_agents.config import AppConfig
    from lawn_agents.models import WeatherSnapshot


def snapshot(config: AppConfig) -> WeatherSnapshot | None:
    """Fetch current weather + 7-day forecast for the configured location.

    Args:
        config: Application configuration (used for lat/lon and HTTP knobs).

    Returns:
        A populated `WeatherSnapshot`, or `None` if the fetch failed.
        Implementations must never raise — log and return `None` instead.
    """
    raise NotImplementedError
