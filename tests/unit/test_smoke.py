"""Smoke tests — every module imports cleanly."""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    "lawn_agents",
    "lawn_agents.config",
    "lawn_agents.llm",
    "lawn_agents.logging",
    "lawn_agents.models",
    "lawn_agents.main",
    "lawn_agents.orchestrator",
    "lawn_agents.notify",
    "lawn_agents.agents",
    "lawn_agents.agents.weather",
    "lawn_agents.agents.soiltemp",
    "lawn_agents.agents.drought",
    "lawn_agents.agents.knowledge",
    "lawn_agents.agents.research",
    "lawn_agents.subjects",
    "lawn_agents.subjects.lawn",
    "lawn_agents.subjects.tree",
    "lawn_agents.subjects.shrub",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_import(module_name: str) -> None:
    importlib.import_module(module_name)
