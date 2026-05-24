"""Lawn subject: cultivar-specific prompt fragments and retrieval filters.

Phase 1's primary subject. Holds the Zeon-Zoysia-specific copy that the
synthesizer composes into its system prompt, plus any retrieval filters
the knowledge layer should apply when answering lawn questions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lawn_agents.config import AppConfig


def system_prompt_fragment(config: AppConfig) -> str:
    """Return the lawn-specific block injected into the synthesizer's system prompt.

    Args:
        config: Application configuration (cultivar, climate thresholds).

    Returns:
        Markdown text describing the subject for the synthesizer.
    """
    raise NotImplementedError
