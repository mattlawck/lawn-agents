"""Allowlist enforcement for the research subagent (ADR 0005).

The subagent fetches pages the *model* asked for, from search results,
and stores them in the RAG where they later get cited as authoritative.
That makes the allowlist a security boundary, not a politeness setting:
content that lands in the index inherits an ADR 0009 trust tier from its
URL and is treated as supporting evidence by ADR 0010's grounding check.
"""

from __future__ import annotations

import httpx
import pytest

from lawn_agents.agents.research import _matches_allowlist
from lawn_agents.ingest import (
    IngestSource,
    RedirectedOffAllowlistError,
    _host_allowed,
    fetch_url_text,
)

ALLOWLIST = ["hgic.clemson.edu", "content.ces.ncsu.edu"]


class TestHostMatching:
    @pytest.mark.parametrize(
        "url",
        [
            "https://hgic.clemson.edu/factsheet/zoysiagrass/",
            "https://content.ces.ncsu.edu/large-patch-in-turf",
            # Subdomains of an allowlisted host are in scope.
            "https://sub.hgic.clemson.edu/x",
        ],
    )
    def test_allowed(self, url: str) -> None:
        assert _host_allowed(url, ALLOWLIST) is True

    @pytest.mark.parametrize(
        ("url", "why"),
        [
            ("https://evil.example/x", "unrelated host"),
            ("https://hgic.clemson.edu.evil.example/x", "suffix-confusion domain"),
            ("https://notclemson.edu/x", "substring but not a domain match"),
            ("http://hgic.clemson.edu/x", "http is not https"),
            ("file:///etc/passwd", "non-web scheme"),
            ("ftp://hgic.clemson.edu/x", "non-web scheme on an allowed host"),
            ("", "empty"),
        ],
    )
    def test_rejected(self, url: str, why: str) -> None:
        assert _host_allowed(url, ALLOWLIST) is False, why

    def test_userinfo_cannot_smuggle_a_host(self) -> None:
        """`netloc` includes userinfo; `hostname` does not.

        The old check compared against `netloc`, so credentials in the
        URL became part of the string being matched.
        """
        assert _host_allowed("https://hgic.clemson.edu@evil.example/x", ALLOWLIST) is False

    def test_port_does_not_break_a_legitimate_match(self) -> None:
        """The old netloc check rejected allowed hosts carrying a port."""
        assert _host_allowed("https://hgic.clemson.edu:8443/x", ALLOWLIST) is True

    def test_research_prefilter_uses_the_same_rule(self) -> None:
        """Pre-fetch and post-redirect checks must not drift apart."""
        for url in ("https://hgic.clemson.edu/x", "https://evil.example/x"):
            assert _matches_allowlist(url, ALLOWLIST) == _host_allowed(url, ALLOWLIST)


class TestRedirectGuard:
    """The bypass this module exists to close.

    `_matches_allowlist` validated the search result's URL, then the fetch
    ran with `follow_redirects=True`. An allowlisted host redirecting
    off-allowlist got its content ingested under the *original*
    allowlisted source_id.
    """

    def _source(self) -> IngestSource:
        return IngestSource(
            kind="url",
            location="https://hgic.clemson.edu/factsheet/zoysiagrass/",
            source_id="url:https://hgic.clemson.edu/factsheet/zoysiagrass/",
            source_title="hgic.clemson.edu/factsheet/zoysiagrass",
        )

    def test_offsite_redirect_is_rejected(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "hgic.clemson.edu":
                return httpx.Response(302, headers={"Location": "https://evil.example/payload"})
            return httpx.Response(200, html="<html><body>malicious copy</body></html>")

        client = httpx.Client(transport=httpx.MockTransport(handler))
        with pytest.raises(RedirectedOffAllowlistError, match=r"evil\.example"):
            fetch_url_text(self._source(), client, allowed_hosts=ALLOWLIST)

    def test_onsite_redirect_is_allowed(self) -> None:
        """Legitimate redirects (trailing slash, canonicalization) still work."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/zoysiagrass/"):
                return httpx.Response(
                    301,
                    headers={"Location": "https://hgic.clemson.edu/factsheet/zoysiagrass"},
                )
            return httpx.Response(
                200,
                html=(
                    "<html><body><article><p>"
                    + ("Zoysiagrass maintenance guidance for warm season lawns. " * 12)
                    + "</p></article></body></html>"
                ),
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        text = fetch_url_text(self._source(), client, allowed_hosts=ALLOWLIST)
        assert "Zoysiagrass" in text

    def test_seed_urls_are_exempt(self) -> None:
        """Curated seed URLs pass allowed_hosts=None and redirect freely.

        The user hand-picks those; the allowlist governs what the *model*
        can pull in unattended.
        """

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "hgic.clemson.edu":
                return httpx.Response(302, headers={"Location": "https://cdn.example/mirror"})
            return httpx.Response(
                200,
                html=(
                    "<html><body><article><p>"
                    + ("Mirrored extension content about zoysiagrass care. " * 12)
                    + "</p></article></body></html>"
                ),
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        text = fetch_url_text(self._source(), client)
        assert "Mirrored" in text
