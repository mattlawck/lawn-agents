"""Tests for the CLI entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lawn_agents import notify, orchestrator, planner
from lawn_agents.main import EXIT_OK, EXIT_RUNTIME, EXIT_USAGE, build_parser, cli
from lawn_agents.models import Recommendation


def _good_rec() -> Recommendation:
    return Recommendation(
        headline="Apply pre-emergent this weekend.",
        conditions_summary="4-inch soil temp 56F.",
    )


def _refusal_rec() -> Recommendation:
    return Recommendation(
        headline="Unable to produce a recommendation.",
        conditions_summary="",
        refused=True,
        refusal_reason="Out of scope.",
    )


@pytest.fixture
def cwd_with_example_config(monkeypatch: pytest.MonkeyPatch, repo_root: Path) -> Path:
    """cd into a tmp dir that contains the committed example config."""
    monkeypatch.chdir(repo_root)
    return repo_root


# --- argparse plumbing ----------------------------------------------------


class TestParser:
    def test_requires_a_mode(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_accepts_ask(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--ask", "should I fertilize this week?"])
        assert args.ask == "should I fertilize this week?"

    def test_accepts_scheduled(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--scheduled"])
        assert args.scheduled is True

    def test_accepts_plan_month(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--plan-month", "2026-07"])
        assert args.plan_month == "2026-07"

    def test_accepts_plan_year(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--plan-year", "2026"])
        assert args.plan_year == 2026


# --- cli dispatch ---------------------------------------------------------


class TestCliDispatch:
    def test_ask_invokes_orchestrator_answer_and_emits(
        self,
        cwd_with_example_config: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        emitted: list[Recommendation] = []
        monkeypatch.setattr(orchestrator, "answer", lambda _q, _s: _good_rec())
        monkeypatch.setattr(
            notify,
            "build_sinks",
            lambda _c: [_RecordingSink(emitted)],
        )

        exit_code = cli(["--config", "config.example.yaml", "--ask", "any question"])
        assert exit_code == EXIT_OK
        assert len(emitted) == 1
        assert emitted[0].headline == "Apply pre-emergent this weekend."

    def test_ask_refusal_exits_nonzero(
        self,
        cwd_with_example_config: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        emitted: list[Recommendation] = []
        monkeypatch.setattr(orchestrator, "answer", lambda _q, _s: _refusal_rec())
        monkeypatch.setattr(notify, "build_sinks", lambda _c: [_RecordingSink(emitted)])

        exit_code = cli(["--config", "config.example.yaml", "--ask", "trees"])
        assert exit_code == EXIT_RUNTIME
        assert emitted[0].refused is True

    def test_scheduled_invokes_scheduled_check(
        self,
        cwd_with_example_config: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        emitted: list[Recommendation] = []
        called: dict[str, Any] = {}

        def fake_scheduled(settings: Any) -> Recommendation:
            called["settings"] = settings
            return _good_rec()

        monkeypatch.setattr(orchestrator, "scheduled_check", fake_scheduled)
        monkeypatch.setattr(notify, "build_sinks", lambda _c: [_RecordingSink(emitted)])

        exit_code = cli(["--config", "config.example.yaml", "--scheduled"])
        assert exit_code == EXIT_OK
        assert "settings" in called
        assert len(emitted) == 1

    def test_plan_month_dispatches_to_planner(
        self,
        cwd_with_example_config: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        emitted: list[Recommendation] = []
        called: dict[str, Any] = {}

        def fake_plan_month(year: int, month: int, _settings: Any) -> Recommendation:
            called["year"] = year
            called["month"] = month
            return _good_rec()

        monkeypatch.setattr(planner, "plan_month", fake_plan_month)
        monkeypatch.setattr(notify, "build_sinks", lambda _c: [_RecordingSink(emitted)])

        exit_code = cli(["--config", "config.example.yaml", "--plan-month", "2026-07"])
        assert exit_code == EXIT_OK
        assert called == {"year": 2026, "month": 7}
        assert len(emitted) == 1

    def test_plan_month_bad_format_returns_usage_error(
        self,
        cwd_with_example_config: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        exit_code = cli(["--config", "config.example.yaml", "--plan-month", "nope"])
        assert exit_code == EXIT_USAGE
        err = capsys.readouterr().err
        assert "--plan-month" in err

    def test_plan_year_dispatches_to_planner(
        self,
        cwd_with_example_config: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        emitted: list[Recommendation] = []
        called: dict[str, Any] = {}

        def fake_plan_year(year: int, _settings: Any) -> Recommendation:
            called["year"] = year
            return _good_rec()

        monkeypatch.setattr(planner, "plan_year", fake_plan_year)
        monkeypatch.setattr(notify, "build_sinks", lambda _c: [_RecordingSink(emitted)])

        exit_code = cli(["--config", "config.example.yaml", "--plan-year", "2026"])
        assert exit_code == EXIT_OK
        assert called == {"year": 2026}
        assert len(emitted) == 1

    def test_missing_config_returns_usage_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.chdir(tmp_path)
        exit_code = cli(["--config", "missing.yaml", "--ask", "hi"])
        assert exit_code == EXIT_USAGE
        err = capsys.readouterr().err
        assert "config" in err.lower()


# --- helpers --------------------------------------------------------------


class _RecordingSink:
    """Captures emitted Recommendations into a shared list."""

    def __init__(self, sink: list[Recommendation]) -> None:
        self._sink = sink

    def emit(self, recommendation: Recommendation) -> None:
        self._sink.append(recommendation)
