"""Tree subject — Phase 2 placeholder.

Future scope: oaks (young + mature), palms. Implementing this module
requires (a) adding a `subjects:` list to `config.yaml`, (b) extending
the router to dispatch on subject, and (c) seeding a tree-specific
corpus. Not loaded by Phase 1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lawn_agents.config import AppConfig


def system_prompt_fragment(config: AppConfig) -> str:
    """Phase 2 — not yet implemented."""
    raise NotImplementedError("tree subject is a Phase 2 placeholder")
