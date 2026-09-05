"""Source trust tiers (ADR 0009).

Every passage in `<sources>` used to read as equally authoritative to
the synthesizer. The 2026-09-02 acceptance run recommended three Yard
Mastery SKUs cited to a Lawn Care Nut guide — real citations to a real
document, and a vendor recommending its own line. The never-guess
guardrail (ADR 0003) can't catch that: it only checks that a citation
exists, not what kind of source it points at.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lawn_agents.agents.knowledge import classify_source, format_sources
from lawn_agents.config import Settings, SourceTiersConfig
from lawn_agents.models import Passage, SourceTier


def _passage(
    *,
    url: str | None = None,
    source_id: str = "s",
    source_title: str = "S",
    content: str = "Apply at 55F.",
    score: float = 0.8,
    requires_review: bool = False,
) -> Passage:
    return Passage(
        content=content,
        score=score,
        source_id=source_id,
        source_title=source_title,
        url=url,
        requires_review=requires_review,
    )


class TestClassifySource:
    @pytest.fixture
    def tiers(self) -> SourceTiersConfig:
        return SourceTiersConfig()

    def test_clemson_is_extension(self, tiers: SourceTiersConfig) -> None:
        p = _passage(url="https://hgic.clemson.edu/factsheet/zoysiagrass/")
        assert classify_source(p, tiers) is SourceTier.EXTENSION

    def test_ncsu_is_extension(self, tiers: SourceTiersConfig) -> None:
        p = _passage(url="https://content.ces.ncsu.edu/large-patch-in-turf")
        assert classify_source(p, tiers) is SourceTier.EXTENSION

    def test_cdms_is_label(self, tiers: SourceTiersConfig) -> None:
        p = _passage(url="https://www.cdms.net/ldat/ld8CF003.pdf")
        assert classify_source(p, tiers) is SourceTier.LABEL

    def test_supersod_is_vendor(self, tiers: SourceTiersConfig) -> None:
        p = _passage(url="https://info.supersod.com/zeon-zoysia-specs")
        assert classify_source(p, tiers) is SourceTier.VENDOR

    def test_unmatched_is_unknown(self, tiers: SourceTiersConfig) -> None:
        p = _passage(url="https://some-random-blog.example/lawn-tips")
        assert classify_source(p, tiers) is SourceTier.UNKNOWN

    def test_label_wins_over_vendor_host(self, tiers: SourceTiersConfig) -> None:
        """Trust travels with the document, not the host serving it.

        A manufacturer label mirrored on a retailer's domain is still a
        label — the memory-documented cross-match rule for retailer
        pages depends on that distinction.
        """
        p = _passage(
            url="https://www.trianglecc.com/wp-content/uploads/Bayer-Celcius-WG-Label.pdf",
            source_title="Bayer Celsius WG Label",
        )
        assert classify_source(p, tiers) is SourceTier.LABEL

    def test_local_pdf_tiered_by_title(self) -> None:
        """Corpus PDFs have no URL, so title/source_id must be matchable."""
        tiers = SourceTiersConfig(vendor=["Warm Season E-Guide"])
        p = _passage(
            url=None,
            source_id="pdf:2025 Warm Season E-Guide",
            source_title="2025 Warm Season E-Guide",
        )
        assert classify_source(p, tiers) is SourceTier.VENDOR

    def test_matching_is_case_insensitive(self) -> None:
        tiers = SourceTiersConfig(extension=["HGIC.Clemson.EDU"])
        p = _passage(url="https://hgic.clemson.edu/x")
        assert classify_source(p, tiers) is SourceTier.EXTENSION

    def test_empty_patterns_yield_unknown(self) -> None:
        tiers = SourceTiersConfig(label=[], extension=[], vendor=[])
        p = _passage(url="https://hgic.clemson.edu/x")
        assert classify_source(p, tiers) is SourceTier.UNKNOWN


class TestFormatSources:
    @pytest.fixture
    def tiers(self) -> SourceTiersConfig:
        return SourceTiersConfig()

    def test_empty_passages(self, tiers: SourceTiersConfig) -> None:
        assert format_sources([], tiers) == "(no relevant passages retrieved)"

    def test_tier_marker_is_present(self, tiers: SourceTiersConfig) -> None:
        out = format_sources([_passage(url="https://hgic.clemson.edu/x")], tiers)
        assert "tier=extension" in out

    def test_vendor_and_extension_are_distinguishable(self, tiers: SourceTiersConfig) -> None:
        out = format_sources(
            [
                _passage(url="https://hgic.clemson.edu/x", source_id="a"),
                _passage(url="https://info.supersod.com/y", source_id="b"),
            ],
            tiers,
        )
        assert "tier=extension" in out
        assert "tier=vendor" in out

    def test_preserves_existing_provenance_fields(self, tiers: SourceTiersConfig) -> None:
        """Tiering is additive — the unreviewed flag and score still render."""
        out = format_sources(
            [_passage(url="https://hgic.clemson.edu/x", requires_review=True)],
            tiers,
        )
        assert "[unreviewed]" in out
        assert "score=0.800" in out
        assert "Apply at 55F." in out


class TestSourceTiersConfigLoading:
    def test_defaults_apply_when_config_omits_the_section(self, config_yaml_path: Path) -> None:
        """Existing configs keep working and still get sensible tiering."""
        settings = Settings.load(config_yaml_path)
        tiers = settings.app.knowledge.source_tiers
        p = _passage(url="https://hgic.clemson.edu/factsheet/zoysiagrass/")
        assert classify_source(p, tiers) is SourceTier.EXTENSION
