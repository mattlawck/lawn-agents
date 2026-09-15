"""The weekly unattended check.

The property under test throughout is **silence**. A watchdog that
speaks every week gets muted, and a muted watchdog is worse than none —
it leaves the user believing they'd have been told.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from lawn_agents import todoist, watchdog
from lawn_agents.config import Settings
from lawn_agents.models import SoilSnapshot


@pytest.fixture
def settings(config_yaml_path: Path, repo_root: Path) -> Settings:
    """Example config with Todoist enabled and the real program loaded."""
    loaded = Settings.load(config_yaml_path)
    from lawn_agents.models import ProgramConfig

    program = ProgramConfig.model_validate(
        yaml.safe_load((repo_root / "data" / "calendar.yaml").read_text())
    )
    app = loaded.app.model_copy(
        update={
            "todoist": loaded.app.todoist.model_copy(
                update={"enabled": True, "project_id": "proj-1"}
            )
        }
    )
    # conftest injects the LLM keys but not this one.
    from pydantic import SecretStr

    return loaded.model_copy(
        update={
            "app": app,
            "program": program,
            "todoist_api_token": SecretStr("test-todoist-token"),
        }
    )


def _fake_client(rows: list[dict[str, Any]], created: list[dict[str, Any]]) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"results": rows})
        import json

        body = json.loads(request.content)
        created.append(body)
        return httpx.Response(
            200,
            json={
                "id": f"new-{len(created)}",
                "content": body["content"],
                "description": body.get("description", ""),
                "due": {"date": body["due_date"]} if body.get("due_date") else None,
                "labels": body.get("labels", []),
            },
        )

    return todoist.TodoistClient(
        "tok",
        "proj-1",
        client=httpx.Client(base_url=todoist.API_BASE, transport=httpx.MockTransport(handler)),
    )


def _no_soil(monkeypatch: pytest.MonkeyPatch) -> None:
    from lawn_agents.agents import soiltemp

    monkeypatch.setattr(soiltemp, "snapshot", lambda _c: None)


def _soil(monkeypatch: pytest.MonkeyPatch, trailing: list[float]) -> None:
    from datetime import UTC, datetime

    from lawn_agents.agents import soiltemp

    snap = SoilSnapshot(
        fetched_at=datetime(2026, 9, 11, tzinfo=UTC),
        station_id="x",
        current_4in_f=trailing[-1],
        trailing_7d_4in_f=trailing,
    )
    monkeypatch.setattr(soiltemp, "snapshot", lambda _c: snap)


class TestSilence:
    def test_quiet_week_renders_nothing(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deep winter: nothing critical approaching, nothing overdue."""
        _no_soil(monkeypatch)
        created: list[dict[str, Any]] = []
        result = watchdog.run(settings, today=date(2026, 7, 20), client=_fake_client([], created))
        assert result.spoke is False
        assert watchdog.render(result) == ""
        assert created == []

    def test_routine_items_alone_do_not_speak(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Scouting and fertility are routine — they don't earn a 7am ping."""
        _no_soil(monkeypatch)
        created: list[dict[str, Any]] = []
        # Mid-October: routine items are open, criticals already tracked.
        rows = [
            {
                "id": "a",
                "content": f"Fall pre-emergent {todoist.marker('preemergent-fall', 2026)}",
                "description": "",
                "due": {"date": "2026-10-20"},
                "labels": [],
            },
            {
                "id": "b",
                "content": (
                    f"Preventive fungicide {todoist.marker('fungicide-large-patch-fall', 2026)}"
                ),
                "description": "",
                "due": {"date": "2026-10-25"},
                "labels": [],
            },
        ]
        result = watchdog.run(
            settings, today=date(2026, 10, 10), client=_fake_client(rows, created)
        )
        assert created == [], "routine work must not be filed unprompted"
        assert result.spoke is False


class TestSpeaksWhenItShould:
    def test_files_a_task_for_an_approaching_critical_window(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _no_soil(monkeypatch)
        created: list[dict[str, Any]] = []
        result = watchdog.run(settings, today=date(2026, 9, 11), client=_fake_client([], created))
        titles = [c["content"] for c in created]
        assert any("preemergent-fall/2026" in t for t in titles)
        assert result.spoke is True
        assert "attention" in watchdog.render(result)

    def test_does_not_duplicate_a_tracked_item(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _no_soil(monkeypatch)
        created: list[dict[str, Any]] = []
        rows = [
            {
                "id": "a",
                "content": f"Fall pre-emergent {todoist.marker('preemergent-fall', 2026)}",
                "description": "",
                "due": {"date": "2026-09-15"},
                "labels": [],
            }
        ]
        watchdog.run(settings, today=date(2026, 9, 11), client=_fake_client(rows, created))
        assert not any("preemergent-fall" in c["content"] for c in created)

    def test_overdue_task_is_surfaced(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A slipped hand-written task is the plan diverging from reality."""
        _no_soil(monkeypatch)
        created: list[dict[str, Any]] = []
        rows = [
            {
                "id": "a",
                "content": "Spray Sedge Ender (sulfentrazone) on nutsedge",
                "description": "",
                "due": {"date": "2026-09-14"},
                "labels": [],
            }
        ]
        result = watchdog.run(settings, today=date(2026, 9, 25), client=_fake_client(rows, created))
        assert result.spoke is True
        assert "Past due" in watchdog.render(result)
        assert "Sedge Ender" in watchdog.render(result)


class TestDegradation:
    def test_runs_without_soil_data(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Window arithmetic needs no station; the nudge must survive an outage."""
        _no_soil(monkeypatch)
        created: list[dict[str, Any]] = []
        result = watchdog.run(settings, today=date(2026, 9, 11), client=_fake_client([], created))
        assert result.soil is None
        assert created, "an unreachable station must not silence the watchdog"

    def test_soil_data_is_used_when_present(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _soil(monkeypatch, [83, 84, 78, 81, 80, 81, 82])
        created: list[dict[str, Any]] = []
        result = watchdog.run(settings, today=date(2026, 9, 11), client=_fake_client([], created))
        assert result.soil is not None
        assert result.soil.current_4in_f == 82


class TestConfigurationGuards:
    def test_disabled_todoist_skips_cleanly(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _no_soil(monkeypatch)
        off = settings.model_copy(
            update={
                "app": settings.app.model_copy(
                    update={"todoist": settings.app.todoist.model_copy(update={"enabled": False})}
                )
            }
        )
        result = watchdog.run(off, today=date(2026, 9, 11))
        assert result.skipped_reason is not None
        assert "not enabled" in result.skipped_reason

    def test_missing_project_id_is_reported(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _no_soil(monkeypatch)
        broken = settings.model_copy(
            update={
                "app": settings.app.model_copy(
                    update={"todoist": settings.app.todoist.model_copy(update={"project_id": None})}
                )
            }
        )
        result = watchdog.run(broken, today=date(2026, 9, 11))
        assert result.skipped_reason is not None
        assert "project_id" in result.skipped_reason


class TestDryRun:
    def test_create_false_evaluates_without_writing(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _no_soil(monkeypatch)
        created: list[dict[str, Any]] = []
        result = watchdog.run(
            settings,
            today=date(2026, 9, 11),
            client=_fake_client([], created),
            create=False,
        )
        assert created == []
        assert result.plan is not None
        assert result.plan.proposals, "it should still have found work to propose"
