"""Unit tests for the NWS weather client (no network — `httpx.MockTransport`)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest

from lawn_agents.agents import weather
from lawn_agents.config import Settings

# --- canned NWS-shaped responses ------------------------------------------

POINTS = {
    "properties": {
        "gridId": "CAE",
        "gridX": 80,
        "gridY": 50,
        "forecast": "https://api.weather.gov/gridpoints/CAE/80,50/forecast",
        "forecastHourly": ("https://api.weather.gov/gridpoints/CAE/80,50/forecast/hourly"),
        "observationStations": ("https://api.weather.gov/gridpoints/CAE/80,50/stations"),
    }
}

FORECAST = {
    "properties": {
        "periods": [
            {
                "name": "Today",
                "startTime": "2026-05-23T10:00:00-04:00",
                "isDaytime": True,
                "temperature": 78,
                "temperatureUnit": "F",
                "probabilityOfPrecipitation": {"value": 20},
            },
            {
                "name": "Tonight",
                "startTime": "2026-05-23T18:00:00-04:00",
                "isDaytime": False,
                "temperature": 62,
                "temperatureUnit": "F",
                "probabilityOfPrecipitation": {"value": 10},
            },
            {
                "name": "Tomorrow",
                "startTime": "2026-05-24T06:00:00-04:00",
                "isDaytime": True,
                "temperature": 82,
                "temperatureUnit": "F",
                "probabilityOfPrecipitation": {"value": 70},
            },
            {
                "name": "Tomorrow Night",
                "startTime": "2026-05-24T18:00:00-04:00",
                "isDaytime": False,
                # Cold snap — should trip the frost flag.
                "temperature": 30,
                "temperatureUnit": "F",
                "probabilityOfPrecipitation": {"value": 80},
            },
        ]
    }
}

STATIONS = {
    "features": [
        {"properties": {"stationIdentifier": "KCUB"}},
        {"properties": {"stationIdentifier": "KCAE"}},
    ]
}

# 23.5°C → 74.3°F current air temp.
OBSERVATION_LATEST = {
    "properties": {
        "temperature": {"value": 23.5, "unitCode": "wmoUnit:degC"},
    }
}

# Three obs in the 24h window; 2.5 + 1.3 = 3.8 mm → ~0.1496 in.
OBSERVATION_WINDOW = {
    "features": [
        {"properties": {"precipitationLastHour": {"value": 2.5}}},
        {"properties": {"precipitationLastHour": {"value": None}}},
        {"properties": {"precipitationLastHour": {"value": 1.3}}},
    ]
}


# --- helpers ---------------------------------------------------------------


def _make_handler(
    *,
    points: dict[str, Any] | int = POINTS,
    forecast: dict[str, Any] | int = FORECAST,
    stations: dict[str, Any] | int = STATIONS,
    observation_latest: dict[str, Any] | int = OBSERVATION_LATEST,
    observation_window: dict[str, Any] | int = OBSERVATION_WINDOW,
) -> Callable[[httpx.Request], httpx.Response]:
    """Build a request handler that maps NWS URLs to canned responses.

    Pass an int to make a given endpoint return that status code (for
    failure-mode tests).
    """

    def _respond(payload: dict[str, Any] | int) -> httpx.Response:
        if isinstance(payload, int):
            return httpx.Response(payload, json={"error": "test failure"})
        return httpx.Response(200, json=payload)

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.startswith("/points/"):
            return _respond(points)
        if path.endswith("/forecast"):
            return _respond(forecast)
        if path.endswith("/stations"):
            return _respond(stations)
        if path.endswith("/observations/latest"):
            return _respond(observation_latest)
        if path.endswith("/observations"):
            return _respond(observation_window)
        return httpx.Response(404, json={"error": f"unmocked path {path}"})

    return handler


def _install_mock_client(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> None:
    """Patch `weather._build_client` to return an httpx client with our transport."""

    def fake_build(_config: Any) -> httpx.Client:
        return httpx.Client(
            base_url=weather.NWS_BASE_URL,
            transport=httpx.MockTransport(handler),
            headers={"User-Agent": "lawn-agents-tests"},
        )

    monkeypatch.setattr(weather, "_build_client", fake_build)


@pytest.fixture
def settings(config_yaml_path: Any) -> Settings:
    return Settings.load(config_yaml_path)


# --- tests -----------------------------------------------------------------


class TestSnapshotHappyPath:
    """End-to-end snapshot with every endpoint responding."""

    def test_returns_populated_snapshot(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_mock_client(monkeypatch, _make_handler())
        snap = weather.snapshot(settings.app)
        assert snap is not None
        assert snap.errors == []
        assert snap.station_id == "KCUB"
        # 23.5 C -> 74.3 F.
        assert snap.current_temp_f is not None
        assert round(snap.current_temp_f, 1) == 74.3
        # 3.8 mm -> ~0.1496 in.
        assert snap.last_24h_precip_in is not None
        assert round(snap.last_24h_precip_in, 3) == 0.15
        # Two days in the forecast fixture: highs 78, 82; lows 62, 30; PoP 20, 80.
        assert snap.forecast_high_7d_f == [78.0, 82.0]
        assert snap.forecast_low_7d_f == [62.0, 30.0]
        assert snap.forecast_precip_chance_7d == [20.0, 80.0]
        # The 30 F low trips frost risk.
        assert snap.frost_risk_next_7d is True


class TestPartialFailures:
    """Each endpoint can fail independently without taking the whole run down."""

    def test_forecast_503_keeps_observation(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_mock_client(monkeypatch, _make_handler(forecast=503))
        snap = weather.snapshot(settings.app)
        assert snap is not None
        assert snap.current_temp_f is not None  # observation still came through
        assert snap.forecast_high_7d_f == []
        assert snap.forecast_low_7d_f == []
        assert snap.frost_risk_next_7d is False
        assert any("forecast fetch failed" in err for err in snap.errors)

    def test_observation_500_keeps_forecast(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_mock_client(monkeypatch, _make_handler(observation_latest=500))
        snap = weather.snapshot(settings.app)
        assert snap is not None
        assert snap.current_temp_f is None
        assert snap.station_id is None
        assert snap.forecast_high_7d_f == [78.0, 82.0]
        assert any("observation fetch failed" in err for err in snap.errors)

    def test_no_stations_returns_no_observation(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_mock_client(monkeypatch, _make_handler(stations={"features": []}))
        snap = weather.snapshot(settings.app)
        assert snap is not None
        assert snap.station_id is None
        assert snap.current_temp_f is None
        assert snap.last_24h_precip_in is None
        # Forecast and frost still parsed.
        assert snap.forecast_high_7d_f == [78.0, 82.0]
        # No error here — empty stations is a valid response.
        assert all("observation fetch" not in e for e in snap.errors)

    def test_gridpoint_failure_returns_empty_snapshot(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_mock_client(monkeypatch, _make_handler(points=502))
        snap = weather.snapshot(settings.app)
        assert snap is not None
        assert snap.current_temp_f is None
        assert snap.forecast_high_7d_f == []
        assert snap.station_id is None
        assert any("gridpoint resolution failed" in err for err in snap.errors)


class TestForecastParsing:
    """The forecast parser must group day/night periods by date."""

    def test_no_frost_when_lows_above_threshold(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mild = {
            "properties": {
                "periods": [
                    {
                        "name": "Day",
                        "startTime": "2026-05-23T10:00:00-04:00",
                        "isDaytime": True,
                        "temperature": 78,
                        "temperatureUnit": "F",
                        "probabilityOfPrecipitation": {"value": 0},
                    },
                    {
                        "name": "Night",
                        "startTime": "2026-05-23T18:00:00-04:00",
                        "isDaytime": False,
                        "temperature": 55,
                        "temperatureUnit": "F",
                        "probabilityOfPrecipitation": {"value": 0},
                    },
                ]
            }
        }
        _install_mock_client(monkeypatch, _make_handler(forecast=mild))
        snap = weather.snapshot(settings.app)
        assert snap is not None
        assert snap.frost_risk_next_7d is False
        assert snap.forecast_low_7d_f == [55.0]

    def test_celsius_temperatures_are_converted(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        metric = {
            "properties": {
                "periods": [
                    {
                        "name": "Day",
                        "startTime": "2026-05-23T10:00:00-04:00",
                        "isDaytime": True,
                        "temperature": 25,
                        "temperatureUnit": "C",
                        "probabilityOfPrecipitation": {"value": 0},
                    },
                ]
            }
        }
        _install_mock_client(monkeypatch, _make_handler(forecast=metric))
        snap = weather.snapshot(settings.app)
        assert snap is not None
        # 25 C -> 77 F.
        assert snap.forecast_high_7d_f == [77.0]


class TestUserAgent:
    """NWS asks for a contact-info User-Agent."""

    def test_user_agent_includes_contact_email(self, settings: Settings) -> None:
        ua = weather._build_user_agent(settings.app)
        assert "lawn-agents" in ua
        assert settings.app.http.contact_email in ua
