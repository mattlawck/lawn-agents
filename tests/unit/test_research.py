"""Unit tests for the research subagent (no network — mocked search + HTTP)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from lawn_agents.agents import research
from lawn_agents.agents.knowledge import LanceDBStore
from lawn_agents.config import Settings

# Reuse the fake embedder from the knowledge tests.
from tests.unit.test_knowledge import FakeEmbeddings

# --- helpers ---------------------------------------------------------------

_GOOD_HTML = """
<html><body>
  <article>
    <h1>Zeon Zoysia July care</h1>
    <p>In coastal SC, apply 1 lb N per 1000 sqft in July.</p>
    <p>Mow at 1.5 inches weekly.</p>
  </article>
</body></html>
"""


def _mock_client(html_by_url: dict[str, str | int]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = html_by_url.get(str(request.url))
        if payload is None:
            return httpx.Response(404, text="not in fixture")
        if isinstance(payload, int):
            return httpx.Response(payload, text="boom")
        return httpx.Response(200, text=payload, headers={"content-type": "text/html"})

    return httpx.Client(transport=httpx.MockTransport(handler))


def _fake_search(urls: list[str]) -> Callable[[str, list[str], int], list[str]]:
    def search(_query: str, _allowlist: list[str], _max: int) -> list[str]:
        return urls

    return search


@pytest.fixture
def settings(config_yaml_path: Path) -> Settings:
    return Settings.load(config_yaml_path)


# --- search_and_ingest -----------------------------------------------------


class TestSearchAndIngest:
    def test_disabled_returns_empty(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings.app.research, "enabled", False)
        result = research.search_and_ingest("any query", settings.app)
        assert result == []

    def test_empty_allowlist_returns_empty(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings.app.research, "domain_allowlist", [])
        result = research.search_and_ingest("any query", settings.app)
        assert result == []

    def test_ingests_fetched_pages_with_review_flags(
        self,
        settings: Settings,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        url = "https://info.supersod.com/lawn-care/zeon-zoysia-lawn-maintenance"
        store = LanceDBStore(index_dir=tmp_path / "idx", vector_dim=4)
        embed = FakeEmbeddings(dim=4)
        client = _mock_client({url: _GOOD_HTML})

        # Point knowledge.retrieve at our tmp store too so the
        # `search_and_ingest` post-ingest retrieve hits the same data.
        monkeypatch.setattr("lawn_agents.agents.knowledge._build_store", lambda _c: store)
        monkeypatch.setattr("lawn_agents.agents.knowledge._build_embeddings", lambda _c: embed)

        passages = research.search_and_ingest(
            "Zeon Zoysia July care",
            settings.app,
            embeddings=embed,
            store=store,
            http_client=client,
            search_fn=_fake_search([url]),
        )
        # At least one passage came back from the post-ingest retrieve.
        assert len(passages) >= 1
        # Every newly-stored passage carries the review flags so the
        # synthesizer/UI can mark it `[unreviewed]`.
        assert all(p.requires_review for p in passages)
        assert all(p.auto_ingested for p in passages)

    def test_off_allowlist_url_is_dropped_before_fetch(
        self,
        settings: Settings,
        tmp_path: Path,
    ) -> None:
        # The fake search returns the off-allowlist URL; the allowlist
        # filter in _ddg_search would drop it normally, but our injected
        # search_fn skips that. In real usage, the URL would never
        # arrive — here we assert the fetch failure path is silent and
        # the run returns []. With no successful fetches, no chunks.
        off_url = "https://example.com/zoysia"  # not in allowlist
        store = LanceDBStore(index_dir=tmp_path / "idx", vector_dim=4)
        embed = FakeEmbeddings(dim=4)
        # Mock returns 404 for any URL — simulates a non-cooperating page.
        client = _mock_client({})

        passages = research.search_and_ingest(
            "any",
            settings.app,
            embeddings=embed,
            store=store,
            http_client=client,
            search_fn=_fake_search([off_url]),
        )
        # All fetches failed → no chunks ingested → post-ingest retrieve
        # against an empty store returns nothing.
        assert passages == []

    def test_search_exception_is_swallowed(
        self,
        settings: Settings,
        tmp_path: Path,
    ) -> None:
        def boom(_q: str, _a: list[str], _n: int) -> list[str]:
            raise RuntimeError("ddgs is being rate-limited")

        result = research.search_and_ingest(
            "any",
            settings.app,
            embeddings=FakeEmbeddings(dim=4),
            store=LanceDBStore(index_dir=tmp_path / "idx", vector_dim=4),
            http_client=_mock_client({}),
            search_fn=boom,
        )
        assert result == []


# --- allowlist matching ----------------------------------------------------


class TestMatchesAllowlist:
    @pytest.mark.parametrize(
        ("url", "allowlist", "expected"),
        [
            (
                "https://info.supersod.com/lawn-care",
                ["info.supersod.com"],
                True,
            ),
            (
                "https://hgic.clemson.edu/factsheet/zoysia",
                ["clemson.edu"],
                True,  # subdomain matches root
            ),
            (
                "https://evil.example.com/info.supersod.com",
                ["info.supersod.com"],
                False,  # path contains domain, but host doesn't match
            ),
            (
                "https://infosupersod.com",  # no separator
                ["info.supersod.com"],
                False,
            ),
            (
                "not-a-url",
                ["clemson.edu"],
                False,
            ),
            (
                "https://hgic.clemson.edu",
                [],
                False,
            ),
        ],
    )
    def test_match(self, url: str, allowlist: list[str], expected: bool) -> None:
        assert research._matches_allowlist(url, allowlist) is expected


class TestUserAgent:
    def test_user_agent_includes_contact_email(self, settings: Settings) -> None:
        ua = research._build_user_agent(settings.app)
        assert "lawn-agents-research" in ua
        assert settings.app.http.contact_email in ua
