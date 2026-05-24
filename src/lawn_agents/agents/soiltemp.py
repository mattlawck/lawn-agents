"""USDA-NRCS AWDB (SCAN) soil-temperature client.

Finds the nearest SCAN station to the configured lat/lon and queries
2"/4" soil temperature. Falls back to a Parton/Logan-style model from
NWS air-temperature history when no station is within the configured
radius. See ADR 0004.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lawn_agents.config import AppConfig
    from lawn_agents.models import SoilSnapshot


def snapshot(config: AppConfig) -> SoilSnapshot | None:
    """Fetch current and trailing-7d soil temperature at 2" and 4" depth.

    Args:
        config: Application configuration (used for lat/lon and station radius).

    Returns:
        A populated `SoilSnapshot`, or `None` if both the AWDB query and
        the modeled fallback fail. Implementations must never raise.
    """
    raise NotImplementedError
