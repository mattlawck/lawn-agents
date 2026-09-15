"""Todoist client and task-ownership marking.

The lawn project is mixed-use — mower servicing, azaleas, oak spikes and
hand-written lawn notes alongside anything generated. Ownership is
therefore load-bearing: the system must recognise its own tasks exactly,
and must never modify one the user wrote.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import httpx
import pytest

from lawn_agents import todoist


def _client(handler: Any) -> todoist.TodoistClient:
    return todoist.TodoistClient(
        "test-token",
        "proj-1",
        client=httpx.Client(base_url=todoist.API_BASE, transport=httpx.MockTransport(handler)),
    )


def _row(content: str, **over: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": "t1",
        "content": content,
        "description": "",
        "due": None,
        "labels": [],
    }
    row.update(over)
    return row


class TestMarker:
    def test_round_trip(self) -> None:
        m = todoist.marker("preemergent-fall", 2026)
        assert m == "[preemergent-fall/2026]"
        assert todoist.parse_marker(f"Fall pre-emergent {m}") == ("preemergent-fall", 2026)

    @pytest.mark.parametrize(
        "title",
        [
            "Apply Pre Emergnent",
            "Fertilize 7-0-20",
            "Spray Sedge Ender (sulfentrazone) on nutsedge",
            "Bag Mow Leaves",
            "[not-a-marker]",  # no year
            "Fall pre-emergent [preemergent-fall/26]",  # short year
        ],
    )
    def test_hand_written_titles_are_never_claimed(self, title: str) -> None:
        """A near-miss must not be adopted — these are the user's tasks."""
        assert todoist.parse_marker(title) is None

    def test_marker_must_be_at_the_end(self) -> None:
        assert todoist.parse_marker("[preemergent-fall/2026] leading") is None

    def test_year_distinguishes_annual_recurrences(self) -> None:
        """2026's fall pre-emergent is not 2027's."""
        assert todoist.marker("preemergent-fall", 2026) != todoist.marker("preemergent-fall", 2027)


class TestOpenTasks:
    def test_parses_ownership_due_and_labels(self) -> None:
        rows = [
            _row(
                "Fall pre-emergent [preemergent-fall/2026]",
                id="a",
                due={"date": "2026-09-15"},
                labels=["lawn-agents"],
            ),
            _row("Apply Pre Emergnent", id="b"),
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.params["project_id"] == "proj-1"
            return httpx.Response(200, json={"results": rows})

        with _client(handler) as client:
            tasks = client.open_tasks()

        owned, manual = tasks
        assert owned.owned is True
        assert owned.item_id == "preemergent-fall"
        assert owned.year == 2026
        assert owned.due == date(2026, 9, 15)
        assert owned.labels == ("lawn-agents",)
        assert manual.owned is False
        assert manual.due is None

    def test_handles_bare_list_payload(self) -> None:
        """The API has returned both a bare list and a {"results": ...} wrapper."""

        def handler(_r: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[_row("x")])

        with _client(handler) as client:
            assert len(client.open_tasks()) == 1

    def test_datetime_due_is_reduced_to_a_date(self) -> None:
        def handler(_r: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[_row("x", due={"date": "2026-09-15T09:00:00"})])

        with _client(handler) as client:
            assert client.open_tasks()[0].due == date(2026, 9, 15)

    def test_http_failure_raises_todoist_error(self) -> None:
        def handler(_r: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "boom"})

        with _client(handler) as client, pytest.raises(todoist.TodoistError, match="list"):
            client.open_tasks()


class TestOverdue:
    def test_past_due_is_overdue(self) -> None:
        task = todoist.Task("a", "x", "", date(2026, 9, 10), (), None, None)
        assert task.is_overdue(date(2026, 9, 11)) is True

    def test_due_today_is_not_overdue(self) -> None:
        task = todoist.Task("a", "x", "", date(2026, 9, 11), (), None, None)
        assert task.is_overdue(date(2026, 9, 11)) is False

    def test_undated_is_never_overdue(self) -> None:
        task = todoist.Task("a", "x", "", None, (), None, None)
        assert task.is_overdue(date(2030, 1, 1)) is False


class TestWrites:
    def test_create_sends_project_due_and_labels(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            seen.update(json.loads(request.content))
            return httpx.Response(200, json=_row(seen["content"], id="new"))

        with _client(handler) as client:
            client.create_task(
                content="Fall pre-emergent [preemergent-fall/2026]",
                description="why",
                due=date(2026, 9, 15),
                labels=["lawn-agents"],
            )

        assert seen["project_id"] == "proj-1"
        assert seen["due_date"] == "2026-09-15"
        assert seen["labels"] == ["lawn-agents"]

    def test_delete_uses_delete_not_close(self) -> None:
        """Retiring a task must not record it as done.

        Completion is this system's evidence that work happened. Closing
        a task nobody did would poison the signal the weekly loop reads.
        """
        calls: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append((request.method, request.url.path))
            return httpx.Response(204)

        with _client(handler) as client:
            client.delete_task("t9")

        assert calls == [("DELETE", "/api/v1/tasks/t9")]

    def test_close_marks_complete(self) -> None:
        calls: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append((request.method, request.url.path))
            return httpx.Response(204)

        with _client(handler) as client:
            client.close_task("t9")

        assert calls == [("POST", "/api/v1/tasks/t9/close")]

    def test_update_skips_the_call_when_nothing_changed(self) -> None:
        def handler(_r: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("should not have issued a request")

        with _client(handler) as client:
            client.update_task("t1")
