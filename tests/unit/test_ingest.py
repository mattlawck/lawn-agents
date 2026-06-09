"""Unit tests for the corpus ingester."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from lawn_agents.agents.knowledge import LanceDBStore
from lawn_agents.config import Settings
from lawn_agents.ingest import (
    IngestSource,
    chunk_text,
    discover_pdfs,
    fetch_pdf_pages,
    fetch_pdf_url_pages,
    fetch_url_text,
    ingest,
    make_url_sources,
)

# Reuse the fake embedder from the knowledge tests.
from tests.unit.test_knowledge import FakeEmbeddings

# --- helpers ---------------------------------------------------------------


def _write_fixture_pdf(path: Path, pages: list[str]) -> None:
    """Generate a small PDF with one given text per page via pypdf."""
    from pypdf import PdfWriter
    from pypdf.generic import (
        ArrayObject,
        DecodedStreamObject,
        DictionaryObject,
        NameObject,
        NumberObject,
    )

    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=612, height=792)
        # Minimal text stream — use a built-in font (Helvetica) and a
        # single Tj operator. pypdf's extract_text reads this back.
        stream = DecodedStreamObject()
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream.set_data(f"BT /F1 12 Tf 50 700 Td ({escaped}) Tj ET".encode("latin-1"))
        page[NameObject("/Contents")] = stream
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {
                        NameObject("/F1"): DictionaryObject(
                            {
                                NameObject("/Type"): NameObject("/Font"),
                                NameObject("/Subtype"): NameObject("/Type1"),
                                NameObject("/BaseFont"): NameObject("/Helvetica"),
                            }
                        )
                    }
                ),
                NameObject("/ProcSet"): ArrayObject([NameObject("/PDF"), NameObject("/Text")]),
            }
        )
        _ = NumberObject  # keep import used; line otherwise unused
    with path.open("wb") as f:
        writer.write(f)


@pytest.fixture
def fixture_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "sample.pdf"
    _write_fixture_pdf(pdf_path, ["Hello first page about zoysia care.", "Page two content."])
    return pdf_path


# --- discovery -------------------------------------------------------------


class TestDiscoverPdfs:
    def test_missing_dir_returns_empty(self, tmp_path: Path) -> None:
        assert discover_pdfs(tmp_path / "does-not-exist") == []

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        assert discover_pdfs(tmp_path) == []

    def test_finds_pdfs_and_ignores_others(self, tmp_path: Path, fixture_pdf: Path) -> None:
        (tmp_path / "notes.txt").write_text("not a pdf", encoding="utf-8")
        (tmp_path / "other.PDF").write_text("uppercase ext", encoding="utf-8")  # not .pdf
        sources = discover_pdfs(tmp_path)
        assert len(sources) == 1
        assert sources[0].kind == "pdf"
        assert sources[0].source_id == f"pdf:{fixture_pdf.name}"
        assert sources[0].source_title == fixture_pdf.stem


class TestMakeUrlSources:
    def test_builds_titles_from_domain_and_path(self) -> None:
        sources = make_url_sources(
            [
                "https://hgic.clemson.edu/factsheet/zoysiagrass/",
                "https://info.supersod.com/lawn-care/zeon-zoysia-lawn-maintenance",
                "https://example.com",
            ]
        )
        assert [s.kind for s in sources] == ["url", "url", "url"]
        assert sources[0].source_title == "hgic.clemson.edu/factsheet/zoysiagrass"
        assert sources[1].source_title.endswith("/zeon-zoysia-lawn-maintenance")
        assert sources[2].source_title == "example.com/"
        assert all(s.source_id.startswith("url:") for s in sources)

    @pytest.mark.parametrize(
        "url",
        [
            "https://www.trianglecc.com/wp-content/uploads/2022/08/Bayer-Celcius-WG-Label.pdf",
            "https://example.com/Foo.PDF",  # uppercase extension
            "https://example.com/path/to/file.pdf?download=1&v=2",  # query string
            "https://example.com/file.pdf#page=3",  # fragment
            "https://bynder.envu.com/m/26dea06330c25990/original/Celsius-WG_NA_US_EN.pdf",
        ],
    )
    def test_pdf_urls_tagged_pdf_url_kind(self, url: str) -> None:
        sources = make_url_sources([url])
        assert sources[0].kind == "pdf_url"
        assert sources[0].location == url

    @pytest.mark.parametrize(
        "url",
        [
            "https://hgic.clemson.edu/factsheet/zoysiagrass/",
            "https://example.com/page.html",
            "https://example.com",
            "https://example.com/article",
            "https://example.com/path.pdfish",  # similar but not actually .pdf
        ],
    )
    def test_non_pdf_urls_stay_url_kind(self, url: str) -> None:
        sources = make_url_sources([url])
        assert sources[0].kind == "url"


# --- chunking --------------------------------------------------------------


class TestChunkText:
    def test_empty_text(self) -> None:
        assert (
            chunk_text(
                "",
                source_id="s",
                source_title="S",
                chunk_size_chars=2800,
                chunk_overlap_chars=400,
            )
            == []
        )

    def test_single_short_paragraph_one_chunk(self) -> None:
        chunks = chunk_text(
            "A short paragraph about pre-emergent timing.",
            source_id="s",
            source_title="S",
            chunk_size_chars=2800,
            chunk_overlap_chars=400,
        )
        assert len(chunks) == 1
        assert chunks[0].content == "A short paragraph about pre-emergent timing."

    def test_multiple_paragraphs_split_at_size_limit(self) -> None:
        para = "x" * 1000
        text = "\n\n".join([para] * 4)  # ~4000 chars total
        chunks = chunk_text(
            text,
            source_id="s",
            source_title="S",
            chunk_size_chars=2500,
            chunk_overlap_chars=200,
        )
        assert len(chunks) >= 2
        for c in chunks:
            assert len(c.content) <= 2500 + 200  # rough upper bound w/ overlap

    def test_chunk_ids_are_deterministic(self) -> None:
        kwargs: dict[str, Any] = {
            "source_id": "s",
            "source_title": "S",
            "chunk_size_chars": 2800,
            "chunk_overlap_chars": 400,
            "page": 3,
        }
        a = chunk_text("hello\n\nworld", **kwargs)
        b = chunk_text("hello\n\nworld", **kwargs)
        assert [c.id for c in a] == [c.id for c in b]

    def test_oversize_single_paragraph_splits_by_char_window(self) -> None:
        """Regression: a paragraph larger than chunk_size_chars must split.

        Pre-fix, HGIC factsheets extracted as a single 19k-char blob
        produced exactly one mega-chunk per page, hiding the chemistry
        section from retrieval. The chunker now splits oversize
        paragraphs by character window as a last resort.
        """
        single_blob = "x" * 8000  # no \n\n boundaries at all
        chunks = chunk_text(
            single_blob,
            source_id="t",
            source_title="t",
            chunk_size_chars=2800,
            chunk_overlap_chars=400,
        )
        assert len(chunks) >= 3  # 8000 chars / 2800 ~ 3 minimum
        # Each chunk's content stays bounded: size_chars + overlap + a
        # tiny slack for the "\n\n" paragraph separator.
        for c in chunks:
            assert len(c.content) <= 2800 + 400 + 16

    def test_chunk_carries_provenance(self) -> None:
        chunks = chunk_text(
            "para",
            source_id="hgic-1207",
            source_title="Clemson HGIC 1207",
            chunk_size_chars=2800,
            chunk_overlap_chars=400,
            url="https://hgic.clemson.edu/x",
            page=4,
        )
        assert len(chunks) == 1
        c = chunks[0]
        assert c.source_id == "hgic-1207"
        assert c.url == "https://hgic.clemson.edu/x"
        assert c.page == 4


# --- PDF + URL fetching ----------------------------------------------------


class TestFetchPdfPages:
    def test_round_trip_two_pages(self, fixture_pdf: Path) -> None:
        source = IngestSource(
            kind="pdf",
            location=str(fixture_pdf),
            source_id="pdf:sample.pdf",
            source_title="sample",
        )
        pages = fetch_pdf_pages(source)
        assert len(pages) == 2
        assert pages[0][0] == 1
        assert "zoysia" in pages[0][1].lower()
        assert pages[1][0] == 2
        assert "page two" in pages[1][1].lower()


class TestFetchPdfUrlPages:
    """Regression: `.pdf` URLs in seed_urls must reach pypdf, not trafilatura."""

    def test_round_trip_via_httpx(self, fixture_pdf: Path) -> None:
        pdf_bytes = fixture_pdf.read_bytes()
        url = "https://example.com/zoysia-guide.pdf"

        def handler(request: httpx.Request) -> httpx.Response:
            assert str(request.url) == url
            return httpx.Response(
                200, content=pdf_bytes, headers={"content-type": "application/pdf"}
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        source = IngestSource(
            kind="pdf_url",
            location=url,
            source_id=f"url:{url}",
            source_title="example.com/zoysia-guide.pdf",
        )
        pages = fetch_pdf_url_pages(source, client)
        assert len(pages) == 2
        assert pages[0][0] == 1
        assert "zoysia" in pages[0][1].lower()
        assert pages[1][0] == 2
        assert "page two" in pages[1][1].lower()

    def test_http_error_raises(self) -> None:
        url = "https://example.com/missing.pdf"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="not found")

        client = httpx.Client(transport=httpx.MockTransport(handler))
        source = IngestSource(
            kind="pdf_url",
            location=url,
            source_id=f"url:{url}",
            source_title="example.com/missing.pdf",
        )
        with pytest.raises(httpx.HTTPStatusError):
            fetch_pdf_url_pages(source, client)


class TestFetchUrlText:
    def test_extracts_main_content(self) -> None:
        html = """
        <html>
          <head><title>Zoysia Care</title></head>
          <body>
            <nav>Skip me</nav>
            <article>
              <h1>Zoysia Lawn Care</h1>
              <p>Apply pre-emergent when 4-inch soil temperature hits 55F consistently.</p>
              <p>Fertilize with 1 to 3 pounds of nitrogen per 1000 sqft annually.</p>
            </article>
            <footer>Copyright</footer>
          </body>
        </html>
        """

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=html, headers={"content-type": "text/html"})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        source = IngestSource(
            kind="url",
            location="https://example.com/zoysia",
            source_id="url:https://example.com/zoysia",
            source_title="example.com/zoysia",
        )
        text = fetch_url_text(source, client)
        assert "pre-emergent" in text.lower()
        assert "skip me" not in text.lower()  # nav stripped
        assert "copyright" not in text.lower()  # footer stripped

    def test_preserves_paragraph_boundaries_for_multi_section_pages(self) -> None:
        """Regression: HGIC-style multi-section pages must yield multiple paragraphs.

        Pre-fix, trafilatura's plain-text output collapsed multi-section
        pages into a single line-broken blob with no `\\n\\n`. That
        defeated `chunk_text`'s paragraph splitter and produced one
        mega-chunk. Markdown output preserves real paragraph + heading
        boundaries.
        """
        html = """
        <html>
          <body>
            <article>
              <h1>White Grub Management</h1>
              <p>White grubs are the larval stage of several scarab beetles.</p>
              <h2>Insect Life Cycle</h2>
              <p>All of these beetles go through four distinct forms.</p>
              <h2>Chemical Control</h2>
              <p>Several insecticides are labeled for use, including chlorantraniliprole.</p>
            </article>
          </body>
        </html>
        """

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=html, headers={"content-type": "text/html"})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        source = IngestSource(
            kind="url",
            location="https://example.com/grubs",
            source_id="url:https://example.com/grubs",
            source_title="example.com/grubs",
        )
        text = fetch_url_text(source, client)
        # Markdown output puts real \n\n between sections and ## on h2's.
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        assert len(paragraphs) >= 4  # h1 + 2 h2 + ≥3 body paragraphs
        assert "chlorantraniliprole" in text.lower()
        assert "## " in text  # h2 headings preserved as markdown


# --- end-to-end ingest -----------------------------------------------------


class TestIngest:
    @pytest.fixture
    def settings(self, config_yaml_path: Path) -> Settings:
        return Settings.load(config_yaml_path)

    def _override_corpus_and_seed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        settings: Settings,
        corpus_dir: Path,
        seed_urls: list[str],
    ) -> None:
        monkeypatch.setattr(settings.app.knowledge, "corpus_dir", corpus_dir)
        monkeypatch.setattr(settings.app, "seed_urls", seed_urls)

    @staticmethod
    def _mock_client(html_by_url: dict[str, str | int]) -> httpx.Client:
        def handler(request: httpx.Request) -> httpx.Response:
            payload = html_by_url.get(str(request.url))
            if payload is None:
                return httpx.Response(404, text="not in fixture")
            if isinstance(payload, int):
                return httpx.Response(payload, text="boom")
            return httpx.Response(200, text=payload, headers={"content-type": "text/html"})

        return httpx.Client(transport=httpx.MockTransport(handler))

    @staticmethod
    def _mock_client_mixed(payloads: dict[str, str | bytes]) -> httpx.Client:
        """HTML for `str` payloads, PDF bytes for `bytes` payloads."""

        def handler(request: httpx.Request) -> httpx.Response:
            payload = payloads.get(str(request.url))
            if payload is None:
                return httpx.Response(404, text="not in fixture")
            if isinstance(payload, bytes):
                return httpx.Response(
                    200, content=payload, headers={"content-type": "application/pdf"}
                )
            return httpx.Response(200, text=payload, headers={"content-type": "text/html"})

        return httpx.Client(transport=httpx.MockTransport(handler))

    def test_ingest_pdf_and_url_into_store(
        self,
        settings: Settings,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        fixture_pdf: Path,
    ) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "sample.pdf").write_bytes(fixture_pdf.read_bytes())

        url = "https://example.com/zoysia"
        html = (
            "<html><body><article><h1>Zoysia</h1>"
            "<p>Apply pre-emergent at 55F.</p></article></body></html>"
        )
        self._override_corpus_and_seed(monkeypatch, settings, corpus, [url])

        store = LanceDBStore(index_dir=tmp_path / "idx", vector_dim=4)
        embeddings = FakeEmbeddings(dim=4)
        client = self._mock_client({url: html})

        report = ingest(settings.app, embeddings=embeddings, store=store, http_client=client)

        assert report.sources_seen == 2
        assert report.sources_ok == 2
        assert report.sources_failed == []
        assert report.chunks_added > 0
        assert report.chunks_total_after == report.chunks_added
        assert store.count() == report.chunks_added

    def test_ingest_pdf_url_into_store(
        self,
        settings: Settings,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        fixture_pdf: Path,
    ) -> None:
        """Regression: a `.pdf` URL must be parsed via pypdf, not trafilatura.

        Before the fix, `make_url_sources` tagged every URL `kind="url"`,
        and `.pdf` URLs went through `fetch_url_text` → trafilatura,
        which returned empty on PDF bytes and silently dropped the
        source as `ingest.source_empty`. This test fixes that regression.
        """
        corpus = tmp_path / "corpus"
        corpus.mkdir()

        pdf_url = "https://example.com/zoysia-guide.pdf"
        self._override_corpus_and_seed(monkeypatch, settings, corpus, [pdf_url])

        store = LanceDBStore(index_dir=tmp_path / "idx", vector_dim=4)
        embeddings = FakeEmbeddings(dim=4)
        client = self._mock_client_mixed({pdf_url: fixture_pdf.read_bytes()})

        report = ingest(settings.app, embeddings=embeddings, store=store, http_client=client)

        assert report.sources_seen == 1
        assert report.sources_ok == 1
        assert report.sources_failed == []
        assert report.chunks_added >= 2  # one chunk per fixture page
        assert store.count() >= 2
        # Citation provenance: at least one stored chunk must carry the
        # PDF URL + page number so the synthesizer can cite both.
        passages = store.search(embeddings.embed_query("zoysia"), k=10)
        assert any(p.url == pdf_url and p.page == 1 for p in passages)
        assert any(p.page == 2 for p in passages)

    def test_ingest_is_idempotent(
        self,
        settings: Settings,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        fixture_pdf: Path,
    ) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "sample.pdf").write_bytes(fixture_pdf.read_bytes())
        self._override_corpus_and_seed(monkeypatch, settings, corpus, [])

        store = LanceDBStore(index_dir=tmp_path / "idx", vector_dim=4)
        embeddings = FakeEmbeddings(dim=4)
        client = self._mock_client({})

        first = ingest(settings.app, embeddings=embeddings, store=store, http_client=client)
        second = ingest(settings.app, embeddings=embeddings, store=store, http_client=client)

        assert first.chunks_added > 0
        # Second run sees the same content; upsert by chunk_id means
        # the store size doesn't grow.
        assert second.chunks_total_before == first.chunks_total_after
        assert second.chunks_total_after == first.chunks_total_after

    def test_failing_url_is_reported_and_run_continues(
        self,
        settings: Settings,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        fixture_pdf: Path,
    ) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "sample.pdf").write_bytes(fixture_pdf.read_bytes())

        bad_url = "https://broken.example.com/oops"
        self._override_corpus_and_seed(monkeypatch, settings, corpus, [bad_url])

        store = LanceDBStore(index_dir=tmp_path / "idx", vector_dim=4)
        embeddings = FakeEmbeddings(dim=4)
        # 500 on the URL fetch.
        client = self._mock_client({bad_url: 500})

        report = ingest(settings.app, embeddings=embeddings, store=store, http_client=client)
        assert report.sources_seen == 2
        assert report.sources_ok == 1  # the PDF
        assert len(report.sources_failed) == 1
        assert report.sources_failed[0][0] == f"url:{bad_url}"
        # PDF chunks still landed.
        assert store.count() > 0
