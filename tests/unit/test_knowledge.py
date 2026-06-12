"""Unit tests for the RAG core (Embeddings + VectorStore + retrieve)."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from lawn_agents.agents import knowledge
from lawn_agents.agents.knowledge import (
    Chunk,
    LanceDBStore,
    chunk_id,
    is_weak,
    retrieve,
)
from lawn_agents.config import Settings
from lawn_agents.models import Passage

# --- fakes ----------------------------------------------------------------


class FakeEmbeddings:
    """Deterministic embedder for tests; no model download.

    Hashes the text and converts the first `dim` bytes to floats in
    [0, 1]. Identical texts produce identical vectors, so we can predict
    which chunks rank highest.
    """

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_query(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in h[: self._dim]]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


def _hgic_chunk(idx: int = 0) -> Chunk:
    return Chunk(
        id=f"hgic-{idx}",
        content="Apply pre-emergent when 4-inch soil temp hits 55F.",
        source_id="hgic-1207",
        source_title="Clemson HGIC 1207",
        url="https://hgic.clemson.edu/zoysia",
        page=3,
        fetched_at=datetime(2026, 5, 22, 12, 0, tzinfo=UTC),
        auto_ingested=False,
        requires_review=False,
    )


def _supersod_chunk(idx: int = 0) -> Chunk:
    return Chunk(
        id=f"supersod-{idx}",
        content="Zeon zoysia tolerates 1 to 3 lb N per 1000 sqft per year.",
        source_id="supersod-zeon",
        source_title="Super-Sod Zeon Maintenance",
        url="https://info.supersod.com/lawn-care/zeon-zoysia-lawn-maintenance",
        page=None,
        fetched_at=datetime(2026, 5, 22, 12, 0, tzinfo=UTC),
        auto_ingested=False,
        requires_review=False,
    )


# --- tests -----------------------------------------------------------------


class TestChunkId:
    """`chunk_id` is content-addressed and stable."""

    def test_same_inputs_same_id(self) -> None:
        a = chunk_id("src", 1, 0, "hello world")
        b = chunk_id("src", 1, 0, "hello world")
        assert a == b

    @pytest.mark.parametrize(
        ("a_args", "b_args"),
        [
            (("src", 1, 0, "a"), ("src", 1, 0, "b")),
            (("src1", 1, 0, "a"), ("src2", 1, 0, "a")),
            (("src", 1, 0, "a"), ("src", 2, 0, "a")),
            (("src", 1, 0, "a"), ("src", 1, 1, "a")),
            (("src", None, 0, "a"), ("src", 1, 0, "a")),
        ],
    )
    def test_distinct_inputs_distinct_ids(
        self,
        a_args: tuple[str, int | None, int, str],
        b_args: tuple[str, int | None, int, str],
    ) -> None:
        assert chunk_id(*a_args) != chunk_id(*b_args)

    def test_id_is_hex_sha256(self) -> None:
        cid = chunk_id("src", 1, 0, "hello")
        assert len(cid) == 64
        int(cid, 16)  # raises if not hex


class TestIsWeak:
    """`is_weak` gates the research subagent."""

    @pytest.fixture
    def settings(self, config_yaml_path: Path) -> Settings:
        return Settings.load(config_yaml_path)

    def _passage(self, score: float) -> Passage:
        return Passage(
            content="x",
            score=score,
            source_id="s",
            source_title="S",
        )

    def test_empty_is_weak(self, settings: Settings) -> None:
        assert is_weak([], settings.app) is True

    def test_above_threshold_is_strong(self, settings: Settings) -> None:
        threshold = settings.app.knowledge.retrieval.weak_score_threshold
        assert is_weak([self._passage(threshold + 0.1)], settings.app) is False

    def test_below_threshold_is_weak(self, settings: Settings) -> None:
        threshold = settings.app.knowledge.retrieval.weak_score_threshold
        assert is_weak([self._passage(threshold - 0.1)], settings.app) is True

    def test_only_top_passage_matters(self, settings: Settings) -> None:
        threshold = settings.app.knowledge.retrieval.weak_score_threshold
        passages = [
            self._passage(threshold + 0.1),
            self._passage(0.01),
        ]
        assert is_weak(passages, settings.app) is False


class TestIsWeakTiered:
    """Tiered relevance check (PR-tba): high / medium / low confidence bands.

    `is_weak` now treats the BGE-small score as a *confidence band*. Top
    scores at or above `strong_score_threshold` (default 0.70) skip
    extra checks; scores between weak and strong run a lexical-overlap
    check, optionally escalating to an LLM relevance gate. Scores below
    `weak_score_threshold` (default 0.55) are weak as before.

    Probed motivation in PR #32: the Japanese-clover query returned
    sedge factsheets at score ~0.696 — squarely medium — and was
    falsely marked strong by the old single-threshold check.
    """

    @pytest.fixture
    def settings(self, config_yaml_path: Path) -> Settings:
        return Settings.load(config_yaml_path)

    def _passage(self, score: float, content: str = "generic content") -> Passage:
        return Passage(
            content=content,
            score=score,
            source_id="s",
            source_title="S",
        )

    def test_strong_band_skips_checks(self, settings: Settings) -> None:
        """Top score at the strong threshold short-circuits without lexical/gate."""
        strong = settings.app.knowledge.retrieval.strong_score_threshold
        gate_calls: list[tuple[str, Passage]] = []

        def gate(q: str, p: Passage) -> bool:  # pragma: no cover - shouldn't fire
            gate_calls.append((q, p))
            return False

        assert (
            is_weak(
                [self._passage(strong + 0.01, content="irrelevant filler")],
                settings.app,
                query="Japanese clover control",
                extra_terms=["annual lespedeza"],
                relevance_gate=gate,
            )
            is False
        )
        assert gate_calls == []

    def test_medium_band_lexical_overlap_marks_strong(self, settings: Settings) -> None:
        """Medium-band score + lexical overlap with the query → strong, no gate."""
        weak = settings.app.knowledge.retrieval.weak_score_threshold
        strong = settings.app.knowledge.retrieval.strong_score_threshold
        mid_score = (weak + strong) / 2
        gate_calls: list[tuple[str, Passage]] = []

        def gate(q: str, p: Passage) -> bool:  # pragma: no cover - shouldn't fire
            gate_calls.append((q, p))
            return False

        # Passage shares "clover" with the query → lexical overlap → strong.
        passage = self._passage(mid_score, content="control of clover in lawns")
        assert (
            is_weak(
                [passage],
                settings.app,
                query="how do I treat clover in my zoysia?",
                relevance_gate=gate,
            )
            is False
        )
        assert gate_calls == []

    def test_medium_band_lexical_miss_escalates_to_gate_relevant(self, settings: Settings) -> None:
        """Lexical miss → escalate to gate; gate says relevant → strong."""
        weak = settings.app.knowledge.retrieval.weak_score_threshold
        strong = settings.app.knowledge.retrieval.strong_score_threshold
        mid_score = (weak + strong) / 2
        gate_calls: list[tuple[str, Passage]] = []

        def gate(q: str, p: Passage) -> bool:
            gate_calls.append((q, p))
            return True

        # No token overlap between question and content.
        passage = self._passage(mid_score, content="unrelated turf maintenance prose")
        assert (
            is_weak(
                [passage],
                settings.app,
                query="how do I treat clover?",
                relevance_gate=gate,
            )
            is False
        )
        assert len(gate_calls) == 1

    def test_medium_band_lexical_miss_gate_irrelevant_is_weak(self, settings: Settings) -> None:
        """Lexical miss + gate disagrees → weak."""
        weak = settings.app.knowledge.retrieval.weak_score_threshold
        strong = settings.app.knowledge.retrieval.strong_score_threshold
        mid_score = (weak + strong) / 2

        def gate(q: str, p: Passage) -> bool:
            return False

        passage = self._passage(mid_score, content="totally unrelated content")
        assert (
            is_weak(
                [passage],
                settings.app,
                query="treat dollarweed",
                relevance_gate=gate,
            )
            is True
        )

    def test_japanese_clover_canonical_failure_now_caught(self, settings: Settings) -> None:
        """Regression test for the exact PR #32 false-positive case.

        Query: "I have Japanese clover in the yard, what should I use?"
        Top hit (pre-fix): sedge factsheet at score 0.696.
        Sedge content shares no tokens with "japanese clover" or the
        weed-bridge aliases (annual lespedeza / Lespedeza striata /
        Kummerowia striata). Lexical miss → escalates to gate. With a
        gate that declines, this is now `weak` and the research
        subagent fires.
        """

        def gate(q: str, p: Passage) -> bool:
            return False

        passage = self._passage(
            0.696,
            content=(
                "Sedges have edges. Yellow nutsedge can be controlled with "
                "halosulfuron-methyl applied in the early summer..."
            ),
        )
        assert (
            is_weak(
                [passage],
                settings.app,
                query="I have Japanese clover in the yard, what should I use?",
                extra_terms=[
                    "Annual lespedeza",
                    "Lespedeza striata",
                    "Kummerowia striata",
                ],
                relevance_gate=gate,
            )
            is True
        )

    def test_low_band_is_weak_regardless(self, settings: Settings) -> None:
        weak = settings.app.knowledge.retrieval.weak_score_threshold

        def gate(q: str, p: Passage) -> bool:  # pragma: no cover - shouldn't fire
            return True

        passage = self._passage(weak - 0.05, content="clover")
        assert (
            is_weak(
                [passage],
                settings.app,
                query="clover",
                relevance_gate=gate,
            )
            is True
        )

    def test_gate_exception_treated_as_weak(self, settings: Settings) -> None:
        """A failing gate must not crash the pipeline; conservatively mark weak."""
        weak = settings.app.knowledge.retrieval.weak_score_threshold
        strong = settings.app.knowledge.retrieval.strong_score_threshold
        mid_score = (weak + strong) / 2

        def gate(q: str, p: Passage) -> bool:
            raise RuntimeError("router model down")

        passage = self._passage(mid_score, content="unrelated content")
        assert (
            is_weak(
                [passage],
                settings.app,
                query="japanese clover",
                relevance_gate=gate,
            )
            is True
        )

    def test_no_gate_provided_treats_lexical_miss_as_weak(self, settings: Settings) -> None:
        """Without a gate, lexical miss → weak (no escalation path)."""
        weak = settings.app.knowledge.retrieval.weak_score_threshold
        strong = settings.app.knowledge.retrieval.strong_score_threshold
        mid_score = (weak + strong) / 2

        passage = self._passage(mid_score, content="unrelated content")
        assert (
            is_weak(
                [passage],
                settings.app,
                query="japanese clover",
            )
            is True
        )

    def test_missing_query_falls_back_to_single_threshold(self, settings: Settings) -> None:
        """Legacy callers without query argument get the old behavior."""
        weak = settings.app.knowledge.retrieval.weak_score_threshold
        strong = settings.app.knowledge.retrieval.strong_score_threshold
        mid_score = (weak + strong) / 2

        # No query passed → fall through medium band as "not weak"
        # (legacy behavior preserved for tests that haven't migrated).
        assert is_weak([self._passage(mid_score, content="x")], settings.app) is False


class TestLanceDBStore:
    """LanceDBStore round-trips chunks through an actual file-backed index."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> LanceDBStore:
        return LanceDBStore(index_dir=tmp_path / "idx", vector_dim=4)

    def test_empty_count_is_zero(self, store: LanceDBStore) -> None:
        assert store.count() == 0

    def test_add_then_search_returns_passages(self, store: LanceDBStore) -> None:
        chunks = [_hgic_chunk(0), _supersod_chunk(0)]
        vectors = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
        store.add(chunks, vectors)
        assert store.count() == 2

        # Query close to the HGIC vector.
        results = store.search([0.99, 0.01, 0.0, 0.0], k=2)
        assert len(results) == 2
        assert results[0].source_id == "hgic-1207"
        assert results[0].score >= results[1].score
        assert results[0].score > 0.5
        assert "pre-emergent" in results[0].content

    def test_add_is_idempotent_by_id(self, store: LanceDBStore) -> None:
        c = _hgic_chunk(0)
        store.add([c], [[1.0, 0.0, 0.0, 0.0]])
        store.add([c], [[1.0, 0.0, 0.0, 0.0]])  # same id, upsert
        assert store.count() == 1

    def test_mismatched_lengths_raise(self, store: LanceDBStore) -> None:
        with pytest.raises(ValueError, match="must be the same length"):
            store.add([_hgic_chunk(0)], [])

    def test_add_empty_is_noop(self, store: LanceDBStore) -> None:
        store.add([], [])
        assert store.count() == 0

    def test_passage_provenance_round_trip(self, store: LanceDBStore) -> None:
        c = _hgic_chunk(0)
        store.add([c], [[1.0, 0.0, 0.0, 0.0]])
        results = store.search([1.0, 0.0, 0.0, 0.0], k=1)
        assert results[0].url == c.url
        assert results[0].page == c.page
        assert results[0].source_title == c.source_title
        assert results[0].fetched_at == c.fetched_at

    def test_reopen_existing_index_does_not_raise(self, tmp_path: Path) -> None:
        index_dir = tmp_path / "idx"
        first = LanceDBStore(index_dir=index_dir, vector_dim=4)
        first.add([_hgic_chunk(0)], [[1.0, 0.0, 0.0, 0.0]])
        assert first.count() == 1

        second = LanceDBStore(index_dir=index_dir, vector_dim=4)
        assert second.count() == 1
        results = second.search([1.0, 0.0, 0.0, 0.0], k=1)
        assert results[0].source_id == "hgic-1207"


class TestRetrieveEndToEnd:
    """`retrieve()` glues embedder + store. Tests inject fakes."""

    @pytest.fixture
    def settings(self, config_yaml_path: Path) -> Settings:
        return Settings.load(config_yaml_path)

    def test_returns_top_k_passages(
        self,
        settings: Settings,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Inject fakes that share dimension=4 with the store.
        embeddings = FakeEmbeddings(dim=4)
        store = LanceDBStore(index_dir=tmp_path / "idx", vector_dim=4)

        # Seed the store with two distinct chunks at known vectors.
        seed_chunks = [_hgic_chunk(0), _supersod_chunk(0)]
        seed_vectors = embeddings.embed_documents([c.content for c in seed_chunks])
        store.add(seed_chunks, seed_vectors)

        monkeypatch.setattr(knowledge, "_build_embeddings", lambda _c: embeddings)
        monkeypatch.setattr(knowledge, "_build_store", lambda _c: store)

        # Query closely matching the HGIC chunk's content.
        results = retrieve(
            "When do I apply pre-emergent for zoysia?",
            settings.app,
            top_k=2,
        )
        assert len(results) == 2
        assert {p.source_id for p in results} == {"hgic-1207", "supersod-zeon"}
        # Scores monotonic non-increasing.
        for a, b in pairwise(results):
            assert a.score >= b.score

    def test_uses_config_default_k_when_unset(
        self,
        settings: Settings,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        embeddings = FakeEmbeddings(dim=4)
        store = LanceDBStore(index_dir=tmp_path / "idx", vector_dim=4)
        chunks = [
            Chunk(
                id=f"c-{i}",
                content=f"chunk {i}",
                source_id="src",
                source_title="Src",
            )
            for i in range(10)
        ]
        vectors = embeddings.embed_documents([c.content for c in chunks])
        store.add(chunks, vectors)

        monkeypatch.setattr(knowledge, "_build_embeddings", lambda _c: embeddings)
        monkeypatch.setattr(knowledge, "_build_store", lambda _c: store)

        results = retrieve("any query", settings.app)
        assert len(results) == settings.app.knowledge.retrieval.rerank_top_k


class TestBgeSmallSmoke:
    """Verify the bge-small wrapper constructs and exposes the right dimension."""

    def test_constructor_does_not_load_model(self) -> None:
        embed = knowledge.BgeSmall()
        assert embed.dimension == knowledge.DEFAULT_VECTOR_DIM
        # Model attribute is lazy; constructor should not pay the import cost.
        assert embed._model is None

    def test_factory_returns_bge_small_by_default(self, config_yaml_path: Path) -> None:
        settings = Settings.load(config_yaml_path)
        embed: Any = knowledge._build_embeddings(settings.app)
        assert isinstance(embed, knowledge.BgeSmall)

    def test_store_factory_returns_lancedb_store(self, config_yaml_path: Path) -> None:
        settings = Settings.load(config_yaml_path)
        store: Any = knowledge._build_store(settings.app)
        assert isinstance(store, LanceDBStore)
