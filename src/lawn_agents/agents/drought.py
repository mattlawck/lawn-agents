"""US Drought Monitor + NOAA CPC outlook client.

Pulls current Drought Monitor D-level for the configured county FIPS,
plus 1-month and 3-month NOAA CPC temperature and precipitation
outlooks. Feeds the annual planner so it can emit drought-conditional
gates ("defer high-N if D2+ persists through May").
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lawn_agents.config import AppConfig
    from lawn_agents.models import DroughtSnapshot


def snapshot(config: AppConfig) -> DroughtSnapshot | None:
    """Fetch current drought classification + seasonal outlooks.

    Args:
        config: Application configuration (used for `location.county_fips`).

    Returns:
        A populated `DroughtSnapshot`, or `None` if the fetch failed.
    """
    raise NotImplementedError
