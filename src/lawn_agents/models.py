"""Domain models used across the pipeline.

Every value that crosses a module boundary is a Pydantic model — fetchers
return snapshots, knowledge retrieval returns passages, the synthesizer
returns a validated recommendation. The `Recommendation` model carries
the never-guess guardrail (see ADR 0003): chemical-category calendar
items must have at least one citation, enforced by a validator.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, model_validator


class ChemicalCategory(StrEnum):
    """Recommendation categories that require at least one citation.

    The "never guess" guardrail (ADR 0003) treats these as safety-critical:
    a `CalendarItem` whose category is in this enum must carry one or more
    `Citation` entries or it fails Pydantic validation.
    """

    FERTILIZER = "fertilizer"
    MICRONUTRIENT = "micronutrient"
    HERBICIDE = "herbicide"
    INSECTICIDE = "insecticide"
    FUNGICIDE = "fungicide"


class GeneralCategory(StrEnum):
    """Recommendation categories that do not require a citation by default."""

    MOWING = "mowing"
    IRRIGATION = "irrigation"
    AERATION = "aeration"
    DETHATCHING = "dethatching"
    MONITORING = "monitoring"
    GENERAL = "general"


# Discriminated string type kept simple; callers use the enums above when
# constructing values. The synthesizer is permitted to emit any of these.
ItemCategory = ChemicalCategory | GeneralCategory


class Citation(BaseModel):
    """A pointer to the source passage that backs a recommendation.

    Every chemical recommendation must carry one or more of these. The
    `url`, `page`, and `snippet` are surfaced in rendered output so the
    user can verify the claim against the original source.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(description="Stable identifier for the source document.")
    source_title: str
    url: str | None = None
    page: int | None = None
    snippet: str = Field(description="The quoted passage backing the claim.")
    auto_researched: bool = Field(
        default=False,
        description="True if the source was added by the research subagent and not yet reviewed.",
    )


class Passage(BaseModel):
    """A retrieved chunk from the knowledge base.

    Returned by `agents.knowledge.retrieve()`. Carries enough provenance
    to construct a `Citation` directly.
    """

    model_config = ConfigDict(extra="forbid")

    content: str
    score: NonNegativeFloat
    source_id: str
    source_title: str
    url: str | None = None
    page: int | None = None
    fetched_at: datetime | None = None
    auto_ingested: bool = False
    requires_review: bool = False

    def to_citation(self) -> Citation:
        """Build a `Citation` from this passage."""
        return Citation(
            source_id=self.source_id,
            source_title=self.source_title,
            url=self.url,
            page=self.page,
            snippet=self.content[:280],
            auto_researched=self.auto_ingested and self.requires_review,
        )


class WeatherSnapshot(BaseModel):
    """Current and short-term forecast conditions from NWS api.weather.gov."""

    model_config = ConfigDict(extra="forbid")

    fetched_at: datetime
    station_id: str | None = None
    current_temp_f: float | None = None
    last_24h_precip_in: float | None = None
    last_7d_precip_in: float | None = None
    forecast_high_7d_f: list[float] = Field(default_factory=list)
    forecast_low_7d_f: list[float] = Field(default_factory=list)
    forecast_precip_chance_7d: list[float] = Field(default_factory=list)
    forecast_max_wind_mph_7d: list[float] = Field(
        default_factory=list,
        description="Per-day max wind speed in mph (max across day + night periods).",
    )
    frost_risk_next_7d: bool = False
    errors: list[str] = Field(default_factory=list)


class SoilSnapshot(BaseModel):
    """Soil temperature + moisture from USDA-NRCS AWDB (SCAN) or modeled fallback."""

    model_config = ConfigDict(extra="forbid")

    fetched_at: datetime
    station_id: str | None = None
    modeled: bool = Field(
        default=False,
        description="True when no SCAN station was available and we used the Parton/Logan model.",
    )
    current_2in_f: float | None = None
    current_4in_f: float | None = None
    trailing_7d_4in_f: list[float] = Field(default_factory=list)
    # SCAN reports SMS as volumetric water content percent (0..50ish).
    current_2in_moisture_pct: float | None = None
    current_4in_moisture_pct: float | None = None
    trailing_7d_4in_moisture_pct: list[float] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class DroughtSnapshot(BaseModel):
    """Current US Drought Monitor classification + optional NOAA CPC outlooks."""

    model_config = ConfigDict(extra="forbid")

    fetched_at: datetime
    county_fips: str
    valid_date: date | None = None
    d_level: int | None = Field(
        default=None,
        ge=-1,
        le=4,
        description="-1 = none, 0 = D0 (abnormally dry) ... 4 = D4 (exceptional drought).",
    )
    cpc_temp_outlook_1mo: str | None = None
    cpc_precip_outlook_1mo: str | None = None
    cpc_temp_outlook_3mo: str | None = None
    cpc_precip_outlook_3mo: str | None = None
    errors: list[str] = Field(default_factory=list)


class Conditions(BaseModel):
    """Aggregate snapshot of every live input the synthesizer can see."""

    model_config = ConfigDict(extra="forbid")

    weather: WeatherSnapshot | None = None
    soil: SoilSnapshot | None = None
    drought: DroughtSnapshot | None = None
    as_of: datetime


class CalendarItem(BaseModel):
    """A single recommended action with optional date window and citations.

    Chemical-category items must carry one or more citations. The
    `requires_citation` model-validator enforces ADR 0003.
    """

    model_config = ConfigDict(extra="forbid")

    category: ItemCategory
    action: str
    earliest: date | None = None
    latest: date | None = None
    conditional: str | None = Field(
        default=None,
        description=(
            "Optional gating condition, e.g. "
            "'only if 4-inch soil temp >= 65F for 3 consecutive days'."
        ),
    )
    citations: list[Citation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _requires_citation_for_chemicals(self) -> Self:
        if isinstance(self.category, ChemicalCategory) and not self.citations:
            msg = (
                f"CalendarItem in chemical category {self.category!r} requires "
                "at least one citation (never-guess guardrail, ADR 0003)."
            )
            raise ValueError(msg)
        return self


class Recommendation(BaseModel):
    """The synthesizer's final output."""

    model_config = ConfigDict(extra="forbid")

    headline: str
    conditions_summary: str
    weekly_actions: list[CalendarItem] = Field(default_factory=list)
    monthly_actions: list[CalendarItem] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    refused: bool = Field(
        default=False,
        description="True when the synthesizer refused to recommend due to missing citations.",
    )
    refusal_reason: str | None = None
