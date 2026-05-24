"""Tests for the CLI entry point."""

from __future__ import annotations

import pytest

from lawn_agents.main import EXIT_OK, build_parser, cli


def test_parser_requires_a_mode() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_accepts_ask() -> None:
    parser = build_parser()
    args = parser.parse_args(["--ask", "should I fertilize this week?"])
    assert args.ask == "should I fertilize this week?"


def test_parser_accepts_scheduled() -> None:
    parser = build_parser()
    args = parser.parse_args(["--scheduled"])
    assert args.scheduled is True


def test_cli_runs_with_ask() -> None:
    assert cli(["--ask", "is it too early for pre-emergent?"]) == EXIT_OK
