"""Unit tests for the USDA-NRCS AWDB soil-temperature client.

No network — `httpx.MockTransport` only.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest

from lawn_agents.agents import soiltemp
from lawn_agents.config import Settings

# --- canned AWDB-shaped responses -----------------------------------------

# Columbia, SC config.example.yaml location: 34.00, -81.03.
# Place one station ~5 mi away (clearly the closest) and one ~600 mi
# away (rejected). The nearby station's triplet becomes the one we
# query for /data.
NEAR_STATION = {
    "stationTriplet": "9999:SC:SCAN",
    "stationId": "9999",
    "stateCode": "SC",
    "networkCode": "SCAN",
    "name": "Mock Near Station",
    "latitude": 34.05,
    "longitude": -81.0,
}

FAR_STATION = {
    "stationTriplet": "1111:CA:SCAN",
    "stationId": "1111",
    "stateCode": "CA",
    "networkCode": "SCAN",
    "name": "Mock Far Station",
    "latitude": 37.0,
    "longitude": -122.0,
}

# A non-SCAN station that must be filtered out even though it's nearby.
NEAR_NON_SCAN_STATION = {
    "stationTriplet": "8888:SC:SNTL",
    "networkCode": "SNTL",
    "name": "Mock Near Non-SCAN",
    "latitude": 34.01,
    "longitude": -81.05,
}


def _values_block(
    depth: int,
    values: list[dict[str, Any]],
    *,
    element_code: str = "STO",
    stored_unit: str = "degF",
) -> dict[str, Any]:
    return {
        "stationElement": {
            "elementCode": element_code,
            "ordinal": 1,
            "heightDepth": depth,
            "durationName": "DAILY",
            "storedUnitCode": stored_unit,
        },
        "values": values,
    }


DATA_HAPPY_PATH = [
    {
        "stationTriplet": "9999:SC:SCAN",
        "data": [
            _values_block(
                depth=-2,
                values=[
                    {"date": "2026-05-17", "value": 71},
                    {"date": "2026-05-18", "value": 72},
                    {"date": "2026-05-19", "value": 73},
                    {"date": "2026-05-20", "value": 74},
                    {"date": "2026-05-21", "value": 75},
                    {"date": "2026-05-22", "value": 76},
                    {"date": "2026-05-23", "value": 77},
                    {"date": "2026-05-24", "value": 78},
                ],
            ),
            _values_block(
                depth=-4,
                values=[
                    {"date": "2026-05-17", "value": 68},
                    {"date": "2026-05-18", "value": 69},
                    {"date": "2026-05-19", "value": 70},
                    {"date": "2026-05-20", "value": 71},
                    {"date": "2026-05-21", "value": 72},
                    {"date": "2026-05-22", "value": 73},
                    {"date": "2026-05-23", "value": 74},
                    {"date": "2026-05-24", "value": 75},
                ],
            ),
            _values_block(
                depth=-2,
                element_code="SMS",
                stored_unit="pct",
                values=[
                    {"date": "2026-05-23", "value": 12.5},
                    {"date": "2026-05-24", "value": 11.0},
                ],
            ),
            _values_block(
                depth=-4,
                element_code="SMS",
                stored_unit="pct",
                values=[
                    {"date": "2026-05-17", "value": 18.0},
                    {"date": "2026-05-18", "value": 17.5},
                    {"date": "2026-05-19", "value": 17.0},
                    {"date": "2026-05-20", "value": 16.5},
                    {"date": "2026-05-21", "value": 16.0},
                    {"date": "2026-05-22", "value": 15.5},
                    {"date": "2026-05-23", "value": 15.0},
                    {"date": "2026-05-24", "value": 14.5},
                ],
            ),
        ],
    }
]


# --- helpers ---------------------------------------------------------------


def _make_handler(
    *,
    stations: list[dict[str, Any]] | int | None = None,
    data: list[dict[str, Any]] | int | None = None,
) -> Callable[[httpx.Request], httpx.Response]:
    """Map AWDB URLs to canned responses.

    Pass an int to make an endpoint return that status code (failure
    mode). Pass `None` to use the default canned response.
    """

    default_stations = [NEAR_STATION, NEAR_NON_SCAN_STATION, FAR_STATION]
    default_data = DATA_HAPPY_PATH

    def _respond(payload: Any) -> httpx.Response:
        if isinstance(payload, int):
            return httpx.Response(payload, json={"error": "test failure"})
        return httpx.Response(200, json=payload)

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/stations"):
            return _respond(stations if stations is not None else default_stations)
        if path.endswith("/data"):
            return _respond(data if data is not None else default_data)
        return httpx.Response(404, json={"error": f"unmocked {path}"})

    return handler


def _install_mock_client(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> None:
    """Patch `soiltemp._build_client` to return a MockTransport-backed client."""

    def fake_build(_config: Any) -> httpx.Client:
        return httpx.Client(
            base_url=soiltemp.AWDB_BASE_URL,
            transport=httpx.MockTransport(handler),
            headers={"User-Agent": "lawn-agents-tests"},
        )

    monkeypatch.setattr(soiltemp, "_build_client", fake_build)


@pytest.fixture
def settings(config_yaml_path: Any) -> Settings:
    return Settings.load(config_yaml_path)


# --- tests -----------------------------------------------------------------


class TestSnapshotHappyPath:
    """End-to-end snapshot with nearby SCAN station and full data."""

    def test_returns_populated_snapshot(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_mock_client(monkeypatch, _make_handler())
        snap = soiltemp.snapshot(settings.app)
        assert snap is not None
        assert snap.errors == []
        assert snap.station_id == "9999:SC:SCAN"
        assert snap.modeled is False
        assert snap.current_2in_f == 78.0
        assert snap.current_4in_f == 75.0
        # Trailing window: last 7 daily values at 4-inch depth.
        assert snap.trailing_7d_4in_f == [69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0]
        # Soil moisture from SMS elements.
        assert snap.current_2in_moisture_pct == 11.0
        assert snap.current_4in_moisture_pct == 14.5
        assert snap.trailing_7d_4in_moisture_pct == [
            17.5,
            17.0,
            16.5,
            16.0,
            15.5,
            15.0,
            14.5,
        ]


class TestStationFiltering:
    """The nearest-SCAN selection filters by network and respects max radius."""

    def test_non_scan_stations_are_ignored(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_mock_client(
            monkeypatch,
            _make_handler(stations=[NEAR_NON_SCAN_STATION, FAR_STATION]),
        )
        snap = soiltemp.snapshot(settings.app)
        assert snap is not None
        # Only candidate was the far SCAN station (~2000 mi); past radius.
        assert snap.station_id is None
        assert snap.current_2in_f is None
        assert any("no SCAN station within" in err for err in snap.errors)

    def test_no_stations_in_response_returns_no_data(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_mock_client(monkeypatch, _make_handler(stations=[]))
        snap = soiltemp.snapshot(settings.app)
        assert snap is not None
        assert snap.station_id is None
        assert any("no SCAN station within" in err for err in snap.errors)


class TestPartialFailures:
    """Either endpoint can fail without taking the whole snapshot down."""

    def test_stations_endpoint_500_returns_error(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_mock_client(monkeypatch, _make_handler(stations=500))
        snap = soiltemp.snapshot(settings.app)
        assert snap is not None
        assert snap.station_id is None
        assert snap.current_2in_f is None
        assert any("nearest SCAN station lookup" in err for err in snap.errors)

    def test_data_endpoint_503_keeps_station_id(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_mock_client(monkeypatch, _make_handler(data=503))
        snap = soiltemp.snapshot(settings.app)
        assert snap is not None
        # Station was found before the data call failed.
        assert snap.station_id == "9999:SC:SCAN"
        assert snap.current_2in_f is None
        assert snap.current_4in_f is None
        assert snap.trailing_7d_4in_f == []
        assert any("soil temp data fetch" in err for err in snap.errors)


class TestDataParsing:
    """The values parser tolerates nulls and missing depths."""

    def test_null_values_are_skipped(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data_with_nulls = [
            {
                "stationTriplet": "9999:SC:SCAN",
                "data": [
                    _values_block(
                        depth=-4,
                        values=[
                            {"date": "2026-05-22", "value": None},
                            {"date": "2026-05-23", "value": 72},
                            {"date": "2026-05-24", "value": None},
                        ],
                    )
                ],
            }
        ]
        _install_mock_client(monkeypatch, _make_handler(data=data_with_nulls))
        snap = soiltemp.snapshot(settings.app)
        assert snap is not None
        # Latest non-null at 4-inch is 72.
        assert snap.current_4in_f == 72.0
        # Trailing window only contains the non-null values.
        assert snap.trailing_7d_4in_f == [72.0]
        # No 2-inch element in this payload.
        assert snap.current_2in_f is None

    def test_empty_data_returns_no_values(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_mock_client(monkeypatch, _make_handler(data=[]))
        snap = soiltemp.snapshot(settings.app)
        assert snap is not None
        assert snap.station_id == "9999:SC:SCAN"
        assert snap.current_2in_f is None
        assert snap.current_4in_f is None
        assert snap.trailing_7d_4in_f == []


class TestHaversine:
    """Sanity check on the distance helper."""

    def test_zero_distance(self) -> None:
        assert soiltemp._haversine_miles(34.0, -81.0, 34.0, -81.0) == pytest.approx(0.0)

    def test_columbia_to_charleston_is_roughly_100_miles(self) -> None:
        # Columbia, SC (~34.00, -81.03) to Charleston, SC (~32.78, -79.93).
        # Great-circle distance is ~105 mi.
        d = soiltemp._haversine_miles(34.0, -81.03, 32.78, -79.93)
        assert 100 < d < 115


class TestUserAgent:
    def test_user_agent_includes_contact_email(self, settings: Settings) -> None:
        ua = soiltemp._build_user_agent(settings.app)
        assert "lawn-agents" in ua
        assert settings.app.http.contact_email in ua
