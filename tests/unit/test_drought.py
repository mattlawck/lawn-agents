"""Unit tests for the US Drought Monitor client.

No network — `httpx.MockTransport` only.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest

from lawn_agents.agents import drought
from lawn_agents.config import Settings

# --- canned USDM CountyStatistics responses --------------------------------

# Real shape from the live API for Charleston County SC during the
# 2026 spring drought (D3 — extreme).
CHARLESTON_D3 = [
    {
        "mapDate": "2026-05-19T00:00:00",
        "fips": "45019",
        "county": "Charleston County",
        "state": "SC",
        "none": 0.00,
        "d0": 100.00,
        "d1": 100.00,
        "d2": 100.00,
        "d3": 100.00,
        "d4": 0.00,
        "validStart": "2026-05-19T00:00:00",
        "validEnd": "2026-05-25T23:59:59",
        "statisticFormatID": 1,
    },
    {
        "mapDate": "2026-05-12T00:00:00",
        "fips": "45019",
        "county": "Charleston County",
        "state": "SC",
        "none": 0.00,
        "d0": 100.00,
        "d1": 100.00,
        "d2": 100.00,
        "d3": 100.00,
        "d4": 0.00,
        "validStart": "2026-05-12T00:00:00",
        "validEnd": "2026-05-18T23:59:59",
        "statisticFormatID": 1,
    },
]

NO_DROUGHT = [
    {
        "mapDate": "2026-05-19T00:00:00",
        "fips": "45079",
        "county": "Richland County",
        "state": "SC",
        "none": 100.00,
        "d0": 0.00,
        "d1": 0.00,
        "d2": 0.00,
        "d3": 0.00,
        "d4": 0.00,
        "validStart": "2026-05-19T00:00:00",
        "validEnd": "2026-05-25T23:59:59",
        "statisticFormatID": 1,
    }
]

D1_PARTIAL = [
    {
        "validStart": "2026-05-19T00:00:00",
        "none": 20.0,
        "d0": 80.0,
        "d1": 40.0,
        "d2": 0.0,
        "d3": 0.0,
        "d4": 0.0,
    }
]


# --- helpers ---------------------------------------------------------------


def _make_handler(
    *, payload: list[dict[str, Any]] | int = CHARLESTON_D3
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        if not request.url.path.endswith("GetDroughtSeverityStatisticsByAreaPercent"):
            return httpx.Response(404, json={"error": "unmocked"})
        if isinstance(payload, int):
            return httpx.Response(payload, json={"error": "test failure"})
        return httpx.Response(200, json=payload)

    return handler


def _install_mock_client(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> None:
    def fake_build(_config: Any) -> httpx.Client:
        return httpx.Client(
            base_url=drought.USDM_BASE_URL,
            transport=httpx.MockTransport(handler),
            headers={"User-Agent": "lawn-agents-tests"},
        )

    monkeypatch.setattr(drought, "_build_client", fake_build)


@pytest.fixture
def settings(config_yaml_path: Path) -> Settings:
    return Settings.load(config_yaml_path)


# --- tests -----------------------------------------------------------------


class TestSnapshotHappyPath:
    def test_extreme_drought_returns_d_level_3(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_mock_client(monkeypatch, _make_handler())
        snap = drought.snapshot(settings.app)
        assert snap is not None
        assert snap.errors == []
        assert snap.county_fips == settings.app.location.county_fips
        assert snap.d_level == 3  # D3 — extreme drought
        assert snap.valid_date == date(2026, 5, 19)

    def test_no_drought_returns_minus_one(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_mock_client(monkeypatch, _make_handler(payload=NO_DROUGHT))
        snap = drought.snapshot(settings.app)
        assert snap is not None
        assert snap.d_level == -1  # no drought present

    def test_partial_coverage_uses_max_nonzero_level(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_mock_client(monkeypatch, _make_handler(payload=D1_PARTIAL))
        snap = drought.snapshot(settings.app)
        assert snap is not None
        # D0 and D1 both have non-zero coverage; max is D1 → level 1.
        assert snap.d_level == 1


class TestFailureModes:
    def test_empty_response_records_error(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_mock_client(monkeypatch, _make_handler(payload=[]))
        snap = drought.snapshot(settings.app)
        assert snap is not None
        assert snap.d_level is None
        assert any("no rows" in e.lower() for e in snap.errors)

    def test_500_response_records_error(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_mock_client(monkeypatch, _make_handler(payload=500))
        snap = drought.snapshot(settings.app)
        assert snap is not None
        assert snap.d_level is None
        assert any("USDM CountyStatistics fetch failed" in e for e in snap.errors)


class TestUserAgent:
    def test_user_agent_includes_contact_email(self, settings: Settings) -> None:
        ua = drought._build_user_agent(settings.app)
        assert "lawn-agents" in ua
        assert settings.app.http.contact_email in ua


class TestMaxNonzeroLevelEdges:
    """Edge cases on the D-level reduction logic."""

    @pytest.mark.parametrize(
        ("d_pcts", "expected"),
        [
            ((0.0, 0.0, 0.0, 0.0, 0.0), -1),
            ((100.0, 0.0, 0.0, 0.0, 0.0), 0),
            ((100.0, 50.0, 0.0, 0.0, 0.0), 1),
            ((100.0, 100.0, 100.0, 100.0, 0.0), 3),
            ((100.0, 100.0, 100.0, 100.0, 100.0), 4),
            ((0.0, 0.0, 0.0, 0.0, 0.01), 4),  # tiny coverage still counts
        ],
    )
    def test_max_nonzero_level(
        self, d_pcts: tuple[float, float, float, float, float], expected: int
    ) -> None:
        row = drought._CountyRow(valid_date=None, d_levels=d_pcts)
        assert row.max_nonzero_level() == expected
