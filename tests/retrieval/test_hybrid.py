"""Phase-4 R4 tests: RRF hybrid fusion + unified retrieval API.

Per docs/RETRIEVAL_DESIGN.md sections 7 (hybrid) and 9 (unified
``retrieve(query, top_k, mode)`` + ``get_chunk``). Synthetic corpora
plus a read-only BM25-equivalence check against the frozen
``data/retrieval/chunks.json`` inventory. Semantic coverage uses a
deterministic stub embedding model, so this module never downloads
weights and never reads ground_truth/ or canonical/ content.
"""

import ast
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import retrieval.hybrid as H  # noqa: E402
from retrieval.bm25 import BM25Index  # noqa: E402
from retrieval import embeddings as E  # noqa: E402

CHUNKS_PATH = ROOT / "data" / "retrieval" / "chunks.json"
META_PATH = ROOT / "data" / "retrieval" / "index_meta.json"
HIT_KEYS = {"chunk_id", "score", "rank", "source"}


class StubModel:
    """Deterministic stub embedding model (no download, no torch)."""

    def __init__(self, dim=8):
        self.dim = dim

    def eval(self):
        return None

    def encode(self, texts, **kwargs):
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            row = np.array([b / 255.0 for b in digest[:self.dim]],
                           dtype=float)
            norm = float(np.linalg.norm(row))
            rows.append(row / norm if norm else row)
        return np.stack(rows)


def make_record(chunk_id, text, **extra):
    record = {"chunk_id": chunk_id, "text": text}
    record.update(extra)
    return record


def make_retriever(pairs, dim=8, **record_extra):
    records = [make_record(cid, text, **record_extra)
               for cid, text in pairs]
    stub = StubModel(dim=dim)
    vectors = E.build_vectors(records, model=stub)
    by_id = {r["chunk_id"]: r for r in records}
    return H.UnifiedRetriever(records, vectors=vectors, model=stub), by_id


def write_artifacts(tmp_path, pairs, model_id=E.MODEL_ID,
                    revision="a" * 40, stub_dim=8):
    """Stage a complete tmp artifact dir (chunks + vectors + meta)."""
    records = [make_record(cid, text) for cid, text in pairs]
    stub = StubModel(dim=stub_dim)
    vectors = E.build_vectors(records, model=stub)
    (tmp_path / "chunks.json").write_text(
        json.dumps(records, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "vectors.json").write_text(
        json.dumps(vectors), encoding="utf-8")
    (tmp_path / "index_meta.json").write_text(
        json.dumps({"embedding_model": {"id": model_id,
                                        "revision": revision}}),
        encoding="utf-8")
    return records, vectors, stub


@pytest.fixture(scope="module")
def real_chunks():
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. Frozen constants
# ---------------------------------------------------------------------------
def test_rrf_constant_frozen():
    assert H.RRF_K == 60
    assert H.SOURCE == "hybrid"
    assert tuple(H.MODES) == ("bm25", "semantic", "hybrid")
    assert (H.TOP_K_MIN, H.TOP_K_MAX) == (1, 100)


# ---------------------------------------------------------------------------
# 2. RRF formula over full-depth ranks
# ---------------------------------------------------------------------------
def test_rrf_single_side_formula():
    bm25 = [{"chunk_id": "CHK-b", "score": 9.9, "rank": 2, "source": "bm25"},
            {"chunk_id": "CHK-a", "score": 1.1, "rank": 1, "source": "bm25"}]
    hits = H.rrf_fuse(bm25, [], 10)
    assert [h["chunk_id"] for h in hits] == ["CHK-a", "CHK-b"]
    assert math.isclose(hits[0]["score"], 1.0 / (60 + 1), rel_tol=1e-12)
    assert math.isclose(hits[1]["score"], 1.0 / (60 + 2), rel_tol=1e-12)
    # Input per-side scores are ignored: ranks only.
    assert hits[0]["score"] > hits[1]["score"]


def test_rrf_combines_both_sides_missing_contributes_nothing():
    bm25 = [{"chunk_id": "CHK-only-bm25", "score": 5.0, "rank": 1,
             "source": "bm25"},
            {"chunk_id": "CHK-both", "score": 4.0, "rank": 2, "source": "bm25"}]
    sem = [{"chunk_id": "CHK-both", "score": 0.9, "rank": 1,
            "source": "semantic"},
           {"chunk_id": "CHK-only-sem", "score": 0.8, "rank": 2,
            "source": "semantic"}]
    hits = H.rrf_fuse(bm25, sem, 10)
    by_id = {h["chunk_id"]: h for h in hits}
    assert math.isclose(by_id["CHK-both"]["score"],
                        1.0 / 62 + 1.0 / 61, rel_tol=1e-12)
    assert math.isclose(by_id["CHK-only-bm25"]["score"], 1.0 / 61,
                        rel_tol=1e-12)
    assert math.isclose(by_id["CHK-only-sem"]["score"], 1.0 / 62,
                        rel_tol=1e-12)
    assert hits[0]["chunk_id"] == "CHK-both"
    assert [h["rank"] for h in hits] == [1, 2, 3]


def test_rrf_deep_ranks_beyond_topk_cap():
    """Ranks past the 100 per-mode cap fuse with the same formula."""
    bm25 = [{"chunk_id": "CHK-deep", "score": 0.5, "rank": 150,
             "source": "bm25"}]
    sem = [{"chunk_id": "CHK-deep", "score": 0.7, "rank": 120,
            "source": "semantic"}]
    (hit,) = H.rrf_fuse(bm25, sem, 10)
    assert math.isclose(hit["score"], 1.0 / 210 + 1.0 / 180, rel_tol=1e-12)
    assert hit["rank"] == 1 and hit["source"] == "hybrid"


def test_rrf_overlap_dedup_and_tie_break():
    bm25 = [{"chunk_id": "CHK-002", "score": 2.0, "rank": 1,
             "source": "bm25"}]
    sem = [{"chunk_id": "CHK-001", "score": 0.9, "rank": 1,
            "source": "semantic"}]
    hits = H.rrf_fuse(bm25, sem, 10)
    # Equal fused scores -> chunk_id ascending.
    assert [h["chunk_id"] for h in hits] == ["CHK-001", "CHK-002"]
    assert hits[0]["score"] == hits[1]["score"]
    assert [h["rank"] for h in hits] == [1, 2]
    # Overlap appears exactly once.
    both = H.rrf_fuse(
        [{"chunk_id": "CHK-x", "score": 1.0, "rank": 3, "source": "bm25"}],
        [{"chunk_id": "CHK-x", "score": 0.5, "rank": 4, "source": "semantic"}],
        10)
    assert len(both) == 1 and both[0]["chunk_id"] == "CHK-x"


def test_rrf_empty_sides():
    assert H.rrf_fuse([], [], 10) == []


def test_rrf_top_k_truncation_and_shape():
    bm25 = [{"chunk_id": "CHK-%03d" % i, "score": float(50 - i), "rank": i,
             "source": "bm25"} for i in range(1, 6)]
    hits = H.rrf_fuse(bm25, [], 2)
    assert len(hits) == 2 and [h["rank"] for h in hits] == [1, 2]
    for hit in hits:
        assert set(hit) == HIT_KEYS
        assert isinstance(hit["score"], float)
        assert hit["source"] == "hybrid"


@pytest.mark.parametrize("bad_top_k", [0, -1, 101, 1000, "5", 2.5, None,
                                       True, False])
def test_rrf_top_k_bounds_enforced(bad_top_k):
    with pytest.raises(TypeError):
        H.rrf_fuse([], [], bad_top_k)


def test_rrf_malformed_rankings_raise():
    good = [{"chunk_id": "CHK-a", "score": 1.0, "rank": 1, "source": "bm25"}]
    for bad in (None, "x", {}, {"chunk_id": "CHK-a"}):
        with pytest.raises(TypeError):
            H.rrf_fuse(bad, [], 5)
        with pytest.raises(TypeError):
            H.rrf_fuse([], bad, 5)
    with pytest.raises(TypeError):
        H.rrf_fuse([{"chunk_id": "CHK-a", "rank": 0}], [], 5)
    with pytest.raises(TypeError):
        H.rrf_fuse([{"chunk_id": 7, "rank": 1}], [], 5)
    with pytest.raises(TypeError):
        H.rrf_fuse([{"chunk_id": "CHK-a", "rank": True}], [], 5)
    with pytest.raises(TypeError):
        H.rrf_fuse(good + [{"chunk_id": "CHK-a", "rank": 2}], [], 5)
    with pytest.raises(TypeError):
        H.rrf_fuse([], [], 5, k=0)
    with pytest.raises(TypeError):
        H.rrf_fuse([], [], 5, k="60")


# ---------------------------------------------------------------------------
# 3. Unified retrieve: mode delegation preserves R2/R3 exactly
# ---------------------------------------------------------------------------
def test_unified_bm25_mode_matches_r2_directly():
    pairs = [("CHK-000001", "invoice INV-2023-0106 total 500"),
             ("CHK-000002", "invoice INV-2023-0107 total 700"),
             ("CHK-000003", "annual report summary")]
    retriever, by_id = make_retriever(pairs)
    direct = BM25Index(
        [make_record(cid, text) for cid, text in pairs]).retrieve(
            "INV-2023-0106", top_k=3, chunks_by_id=by_id)
    assert (retriever.retrieve("INV-2023-0106", top_k=3, mode="bm25")
            == direct)


def test_unified_semantic_mode_matches_r3_directly():
    pairs = [("CHK-000001", "The Jeddah branch reopened after refit works."),
             ("CHK-000002", "Quantum chromodynamics gauge symmetry.")]
    retriever, by_id = make_retriever(pairs)
    query = "Jeddah store closure for renovation"
    direct = E.retrieve(query, 2, retriever._vectors, by_id,
                        model=retriever._model)
    assert retriever.retrieve(query, top_k=2, mode="semantic") == direct


def test_unified_hybrid_end_to_end_synthetic():
    pairs = [("CHK-000001", "invoice INV-2023-0106 total 500"),
             ("CHK-000002", "invoice INV-2023-0107 total 700"),
             ("CHK-000003", "annual report summary")]
    retriever, by_id = make_retriever(pairs)
    hits = retriever.retrieve("INV-2023-0106 total", top_k=3, mode="hybrid")
    assert hits
    assert [h["rank"] for h in hits] == list(range(1, len(hits) + 1))
    for hit in hits:
        assert set(hit) == HIT_KEYS
        assert hit["source"] == "hybrid"
        assert isinstance(hit["score"], float)
        # Provenance preserved: every hit resolves via get_chunk.
        assert retriever.get_chunk(hit["chunk_id"])["chunk_id"] == hit["chunk_id"]
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)
    # Determinism: repeated calls agree.
    assert retriever.retrieve("INV-2023-0106 total", top_k=3, mode="hybrid") == hits


def test_unified_hybrid_uses_full_depth_not_topk_truncated():
    """120 matching chunks: full BM25/semantic rankings exceed the 100 cap."""
    pairs = [("CHK-%04d" % i, "alpha filler document number %d" % i)
             for i in range(120)]
    retriever, _ = make_retriever(pairs)
    full_bm25 = retriever._full_bm25_ranking("alpha", None)
    full_sem = retriever._full_semantic_ranking("alpha", None)
    assert len(full_bm25) == 120
    assert len(full_sem) == 120
    hits = retriever.retrieve("alpha", top_k=100, mode="hybrid")
    assert len(hits) == 100
    assert [h["rank"] for h in hits] == list(range(1, 101))


def test_hybrid_deep_bm25_contribution_flips_public_order():
    """MINOR 2 recovery: a rank-105 contribution provably moves the PUBLIC order.

    141-chunk adversarial corpus with fully forced per-side ranks: BM25
    ranks come from a strict doc-length ladder (query "alpha", tf=1
    wherever it appears), semantic ranks from hand-built 2-D vectors
    around the stub query vector. Z sits at BM25 rank 105 (past the 100
    cap) + semantic rank 1; R is dual (60, 30). Full RRF puts Z first
    (1/165 + 1/61 > 1/120 + 1/90); a top-100-truncated fusion would drop
    Z's deep term and put R first (1/120 + 1/90 > 1/61). Z-first through
    the public retrieve() therefore proves depth >100 fused end to end.
    """
    zid, rid = "CHK-Z-999", "CHK-R-060"
    records, vectors = [], {}
    for i in range(59):
        records.append(make_record("CHK-B-%03d" % i, "alpha" + " x" * (i + 1)))
    records.append(make_record(rid, "alpha" + " x" * 100))
    for i in range(44):
        records.append(make_record("CHK-M-%03d" % i, "alpha" + " x" * (101 + i)))
    records.append(make_record(zid, "alpha" + " x" * 1000))
    for i in range(1, 37):
        records.append(make_record("CHK-S-%02d" % i,
                                   "semantic only filler words number %d" % i))
    stub = StubModel(dim=2)
    qx, qy = stub.encode(["alpha"])[0]
    ex, ey = -qy, qx  # perpendicular unit vector (q is unit norm)

    def vec_at(degrees):
        t = math.radians(degrees)
        return [math.cos(t) * qx + math.sin(t) * ex,
                math.cos(t) * qy + math.sin(t) * ey]

    vectors[zid] = vec_at(0)
    vectors[rid] = vec_at(29)
    for i in range(1, 37):
        vectors["CHK-S-%02d" % i] = vec_at(i if i <= 28 else i + 1)
    retriever = H.UnifiedRetriever(records, vectors=vectors, model=stub)
    bm_ranks = {h["chunk_id"]: h["rank"]
                for h in retriever._full_bm25_ranking("alpha", None)}
    sem_ranks = {h["chunk_id"]: h["rank"]
                 for h in retriever._full_semantic_ranking("alpha", None)}
    assert bm_ranks[zid] == 105 and bm_ranks[rid] == 60
    assert sem_ranks[zid] == 1 and sem_ranks[rid] == 30
    # The flip arithmetic: full Z > full R > truncated Z.
    assert 1 / 165 + 1 / 61 > 1 / 120 + 1 / 90 > 1 / 61
    hits = retriever.retrieve("alpha", top_k=10, mode="hybrid")
    assert [h["chunk_id"] for h in hits[:2]] == [zid, rid]
    assert math.isclose(hits[0]["score"], 1 / 165 + 1 / 61, rel_tol=1e-12)


def test_real_corpus_unified_bm25_matches_r2(real_chunks):
    by_id = {r["chunk_id"]: r for r in real_chunks}
    retriever = H.UnifiedRetriever(real_chunks)
    direct = BM25Index(real_chunks).retrieve(
        "invoice INV-2023-0106 total", top_k=5, chunks_by_id=by_id)
    assert (retriever.retrieve("invoice INV-2023-0106 total", top_k=5,
                               mode="bm25") == direct)
    assert by_id[direct[0]["chunk_id"]]["document_id"] == "DEV-003"


# ---------------------------------------------------------------------------
# 4. Guards: empty/blank, top_k, mode, query, validation-before-blank
# ---------------------------------------------------------------------------
def test_empty_blank_queries_return_empty_all_modes():
    retriever, _ = make_retriever([("CHK-000001", "hello world")])
    for mode in ("bm25", "semantic", "hybrid"):
        assert retriever.retrieve("", top_k=5, mode=mode) == []
        assert retriever.retrieve("   ", top_k=5, mode=mode) == []
        assert retriever.retrieve("\t\n ", top_k=5, mode=mode) == []


def test_blank_query_never_loads_model():
    retriever, _ = make_retriever([("CHK-000001", "hello world")])
    retriever._model = None
    retriever._vectors = {"CHK-000001": [1.0] * 8}
    assert retriever.retrieve("   ", top_k=5, mode="hybrid") == []
    assert retriever._model is None


@pytest.mark.parametrize("bad_top_k", [0, -1, 101, 1000, "5", 2.5, None,
                                       True, False])
def test_unified_top_k_bounds_enforced(bad_top_k):
    retriever, _ = make_retriever([("CHK-000001", "hello world")])
    for mode in ("bm25", "semantic", "hybrid"):
        with pytest.raises(TypeError):
            retriever.retrieve("hello", top_k=bad_top_k, mode=mode)


@pytest.mark.parametrize("bad_mode", ["BM25", "", "HYBRID", "rank", None, 123,
                                       True, ["hybrid"]])
def test_unified_mode_guards(bad_mode):
    retriever, _ = make_retriever([("CHK-000001", "hello world")])
    with pytest.raises(TypeError):
        retriever.retrieve("hello", top_k=5, mode=bad_mode)


@pytest.mark.parametrize("bad_query", [None, 123, ["x"], {"q": 1}, b"bytes"])
def test_unified_query_guards(bad_query):
    retriever, _ = make_retriever([("CHK-000001", "hello world")])
    with pytest.raises(TypeError):
        retriever.retrieve(bad_query, top_k=5, mode="hybrid")


def test_validates_before_blank_shortcut():
    retriever, by_id = make_retriever([("CHK-000001", "hello world")])
    with pytest.raises(TypeError):
        retriever.retrieve("", top_k=0, mode="hybrid")
    with pytest.raises(TypeError):
        retriever.retrieve("   ", top_k=5, mode="nope")
    with pytest.raises(TypeError):
        retriever.retrieve("", top_k=5, mode="hybrid",
                           chunks_by_id=["not", "a", "dict"])
    with pytest.raises(TypeError):
        retriever.retrieve(None, top_k=5, mode="hybrid")
    assert retriever.retrieve("", top_k=5, mode="hybrid") == []


def test_chunks_by_id_filter_restricts_bm25_and_hybrid():
    # Named precisely: bm25 restricts to keys (R2) and hybrid restricts
    # both sides to keys (R4). The semantic path is intentionally NOT
    # listed here: it validates the mapping but scores the full store
    # per accepted R3 (pinned by the test below).
    pairs = [("CHK-000001", "alpha one"),
             ("CHK-000002", "alpha two"),
             ("CHK-000003", "alpha three")]
    retriever, by_id = make_retriever(pairs)
    subset = {"CHK-000003": by_id["CHK-000003"]}
    for mode in ("bm25", "hybrid"):
        hits = retriever.retrieve("alpha", top_k=5, mode=mode,
                                  chunks_by_id=subset)
        assert [h["chunk_id"] for h in hits] == ["CHK-000003"]


def test_semantic_mapping_validated_but_scoring_ignores_values_per_r3():
    """Pin accepted R3 validate-but-ignore behavior on the semantic path.

    Well-formed mappings with junk VALUES score identically to the full
    inventory (values ignored); malformed mappings raise (see the
    recovery tests below for blank/nonblank + no-premature-load).
    """
    pairs = [("CHK-000001", "The Jeddah branch reopened after refit works."),
             ("CHK-000002", "Quantum chromodynamics gauge symmetry.")]
    retriever, by_id = make_retriever(pairs)
    query = "Jeddah store closure for renovation"
    full = retriever.retrieve(query, top_k=2, mode="semantic")
    junk_values = {cid: {} for cid in by_id}
    assert (retriever.retrieve(query, top_k=2, mode="semantic",
                               chunks_by_id=junk_values) == full)


@pytest.mark.parametrize("bad_mapping", [{"CHK-x": "not-a-dict"},
                                         {"CHK-x": None},
                                         {"CHK-x": []},
                                         {123: {}}])
def test_semantic_mapping_entries_rejected_before_blank_shortcut(
        bad_mapping):
    """MAJOR 3 recovery: R3 validates mapping entries before blank."""
    retriever, _ = make_retriever([("CHK-000001", "hello world")])
    with pytest.raises(TypeError):
        retriever.retrieve("", top_k=5, mode="semantic",
                           chunks_by_id=bad_mapping)
    with pytest.raises(TypeError):
        retriever.retrieve("   ", top_k=5, mode="semantic",
                           chunks_by_id=bad_mapping)


def test_semantic_mapping_rejected_before_model_load(monkeypatch):
    """MAJOR 3 recovery: malformed mapping fails before any model load."""
    retriever, _ = make_retriever([("CHK-000001", "hello world")])
    retriever._model = None
    calls = []

    def boom(model_id=None, revision=None, cache=True):
        calls.append((model_id, revision))
        raise AssertionError("model must not load before mapping rejection")

    monkeypatch.setattr(H.emb, "load_model", boom)
    with pytest.raises(TypeError):
        retriever.retrieve("hello", top_k=5, mode="semantic",
                           chunks_by_id={"CHK-x": "not-a-dict"})
    assert calls == [] and retriever._model is None


# ---------------------------------------------------------------------------
# 5. get_chunk: provenance preserved, unknown/malformed guarded
# ---------------------------------------------------------------------------
def test_get_chunk_returns_provenance_record():
    record = make_record(
        "CHK-000001", "Revenue rose.", document_id="DEV-001",
        filename="f.pdf", file_type="pdf", language="en", page=2,
        sheet=None, kind="text", section="Results", element_ids=["e1"],
        spans=[{"element_id": "e1", "start": 0, "end": 13}], length=13)
    retriever_full = H.UnifiedRetriever(
        [record], vectors=E.build_vectors([record], model=StubModel()),
        model=StubModel())
    got = retriever_full.get_chunk("CHK-000001")
    assert got["document_id"] == "DEV-001"
    assert got["text"] == "Revenue rose."
    assert got["element_ids"] == ["e1"]
    assert got["section"] == "Results"
    assert got["spans"] == [{"element_id": "e1", "start": 0, "end": 13}]


def test_get_chunk_unknown_and_malformed():
    retriever, _ = make_retriever([("CHK-000001", "hello")])
    with pytest.raises(KeyError):
        retriever.get_chunk("CHK-does-not-exist")
    for bad in (None, 123, ["CHK-000001"], {"id": 1}):
        with pytest.raises(TypeError):
            retriever.get_chunk(bad)


def test_hit_ids_never_fabricated():
    pairs = [("CHK-000001", "invoice total due"),
             ("CHK-000002", "balance sheet assets"),
             ("CHK-000003", "revenue summary")]
    retriever, by_id = make_retriever(pairs)
    for mode in ("bm25", "semantic", "hybrid"):
        for hit in retriever.retrieve("invoice total", top_k=3, mode=mode):
            assert hit["chunk_id"] in by_id


def test_missing_vectors_semantic_and_hybrid_raise():
    retriever = H.UnifiedRetriever([make_record("CHK-000001", "hello")],
                                   vectors=None)
    with pytest.raises(RuntimeError):
        retriever.retrieve("hello", top_k=5, mode="semantic")
    with pytest.raises(RuntimeError):
        retriever.retrieve("hello", top_k=5, mode="hybrid")
    # BM25 works without vectors.
    assert retriever.retrieve("hello", top_k=5, mode="bm25")


def test_empty_vector_store_returns_empty_without_model():
    retriever = H.UnifiedRetriever([make_record("CHK-000001", "hello")],
                                   vectors={}, model=None)
    assert retriever.retrieve("hello", top_k=5, mode="semantic") == []
    assert retriever._model is None


def test_orphan_vector_ids_rejected_at_construction():
    """MAJOR 1 recovery: vector IDs absent from the inventory are rejected.

    Such an inventory could otherwise return hits that get_chunk()
    cannot resolve. Chunks without vectors stay allowed.
    """
    records = [make_record("CHK-000001", "hello world"),
               make_record("CHK-000002", "goodbye world")]
    stub = StubModel()
    vectors = E.build_vectors(records, model=stub)
    vectors["CHK-orphan-999"] = [1.0] * 8
    with pytest.raises(TypeError) as exc:
        H.UnifiedRetriever(records, vectors=vectors, model=stub)
    assert "CHK-orphan-999" in str(exc.value)
    # Inventory-side extras (chunks without vectors) are fine.
    clean_vectors = E.build_vectors(records, model=stub)
    extra = records + [make_record("CHK-000003", "no vector here")]
    ok = H.UnifiedRetriever(extra, vectors=clean_vectors, model=stub)
    assert ok.get_chunk("CHK-000003")["text"] == "no vector here"


def test_real_artifacts_inventory_matches_vectors(real_chunks):
    """MAJOR 1 preservation: the frozen 4,388 artifacts stay constructible."""
    vectors = json.loads(
        (ROOT / "data" / "retrieval" / "vectors.json").read_text(
            encoding="utf-8"))
    assert set(vectors) == {r["chunk_id"] for r in real_chunks}
    retriever = H.UnifiedRetriever(real_chunks, vectors=vectors,
                                   model=StubModel())
    assert len(retriever._chunks_by_id) == 4388


# ---------------------------------------------------------------------------
# 6. Artifacts: missing files raise FileNotFoundError; roundtrip loads
# ---------------------------------------------------------------------------
def test_missing_artifacts_raise_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        H.load_chunks(tmp_path / "chunks.json")
    with pytest.raises(FileNotFoundError):
        H.load_vectors(tmp_path / "vectors.json")
    with pytest.raises(FileNotFoundError):
        H.load_index_meta(tmp_path / "index_meta.json")
    with pytest.raises(FileNotFoundError):
        H.UnifiedRetriever.from_artifacts(data_dir=tmp_path)
    # Chunks present but vectors missing still raises.
    (tmp_path / "chunks.json").write_text(
        json.dumps([make_record("CHK-000001", "hi")]), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        H.UnifiedRetriever.from_artifacts(data_dir=tmp_path)


def test_from_artifacts_requires_index_meta(tmp_path):
    """MAJOR 2 recovery: chunks+vectors WITHOUT metadata are refused."""
    pairs = [("CHK-000001", "invoice total due"),
             ("CHK-000002", "balance sheet assets")]
    records = [make_record(cid, text) for cid, text in pairs]
    vectors = E.build_vectors(records, model=StubModel())
    (tmp_path / "chunks.json").write_text(
        json.dumps(records, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "vectors.json").write_text(
        json.dumps(vectors), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        H.UnifiedRetriever.from_artifacts(data_dir=tmp_path,
                                          model=StubModel())


def test_from_artifacts_roundtrip_with_stub_model(tmp_path):
    pairs = [("CHK-000001", "invoice total due"),
             ("CHK-000002", "balance sheet assets")]
    records, _, stub = write_artifacts(tmp_path, pairs)
    retriever = H.UnifiedRetriever.from_artifacts(data_dir=tmp_path,
                                                  model=stub)
    hits = retriever.retrieve("invoice total", top_k=2, mode="hybrid")
    assert {h["chunk_id"] for h in hits} <= {"CHK-000001", "CHK-000002"}
    assert retriever.get_chunk("CHK-000001")["text"] == "invoice total due"


def test_recorded_model_and_revision_passed_to_loader(tmp_path, monkeypatch):
    """MAJOR 2 recovery: lazy encoding uses the recorded id + revision."""
    pairs = [("CHK-000001", "invoice total due"),
             ("CHK-000002", "balance sheet assets")]
    write_artifacts(tmp_path, pairs, model_id="recorded-model-id",
                    revision="b" * 40)
    captured = {}

    def fake_load(model_id=None, revision=None, cache=True):
        captured["model_id"] = model_id
        captured["revision"] = revision
        return StubModel()

    monkeypatch.setattr(H.emb, "load_model", fake_load)
    retriever = H.UnifiedRetriever.from_artifacts(data_dir=tmp_path)
    assert retriever._model_id == "recorded-model-id"
    assert retriever._model_revision == "b" * 40
    retriever.retrieve("invoice total", top_k=2, mode="semantic")
    assert captured == {"model_id": "recorded-model-id",
                        "revision": "b" * 40}
    hybrid_loader = H.UnifiedRetriever.from_artifacts(data_dir=tmp_path)
    hybrid_loader.retrieve("invoice total", top_k=2, mode="hybrid")
    assert captured == {"model_id": "recorded-model-id",
                        "revision": "b" * 40}


def test_pending_and_malformed_meta_rejected(tmp_path):
    """MAJOR 2 recovery: explicit failures, never silent fallback."""
    pairs = [("CHK-000001", "hello world")]
    records = [make_record(cid, text) for cid, text in pairs]
    vectors = E.build_vectors(records, model=StubModel())
    (tmp_path / "chunks.json").write_text(
        json.dumps(records), encoding="utf-8")
    (tmp_path / "vectors.json").write_text(
        json.dumps(vectors), encoding="utf-8")
    meta_path = tmp_path / "index_meta.json"

    def write_meta(payload):
        meta_path.write_text(json.dumps(payload), encoding="utf-8")

    # R1-only pending metadata: incompatible for pinned-weight use.
    write_meta({"embedding_model": {"id": "pending",
                                    "revision": "pending"}})
    with pytest.raises(RuntimeError):
        H.UnifiedRetriever.from_artifacts(data_dir=tmp_path,
                                          model=StubModel())
    # Malformed shapes.
    for bad in ([],
                {},
                {"embedding_model": "pending"},
                {"embedding_model": {}},
                {"embedding_model": {"id": None, "revision": "a" * 40}},
                {"embedding_model": {"id": E.MODEL_ID, "revision": None}},
                {"embedding_model": {"id": E.MODEL_ID,
                                     "revision": "not-a-hex-hash"}},
                {"embedding_model": {"id": E.MODEL_ID, "revision": "abc"}}):
        write_meta(bad)
        with pytest.raises(TypeError):
            H.UnifiedRetriever.from_artifacts(data_dir=tmp_path,
                                              model=StubModel())


def test_real_index_meta_records_pins_used_by_loader():
    """MAJOR 2 preservation: frozen pins consumed (never re-pinned here)."""
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    section = meta["embedding_model"]
    assert section["id"] == E.MODEL_ID
    assert re.fullmatch(r"[0-9a-f]{40}", section["revision"])
    retriever = H.UnifiedRetriever.from_artifacts(model=StubModel())
    assert retriever._model_id == section["id"]
    assert retriever._model_revision == section["revision"]


# ---------------------------------------------------------------------------
# 7. Latency measured per call; determinism
# ---------------------------------------------------------------------------
def test_latency_measured_per_call():
    retriever, _ = make_retriever([("CHK-000001", "hello world"),
                                   ("CHK-000002", "goodbye world")])
    assert retriever.last_latency_ms is None
    for mode in ("bm25", "semantic", "hybrid"):
        retriever.retrieve("hello", top_k=2, mode=mode)
        assert isinstance(retriever.last_latency_ms, float)
        assert retriever.last_latency_ms >= 0.0
        assert retriever.last_mode == mode
        assert retriever.last_top_k == 2
    retriever.retrieve("", top_k=2, mode="hybrid")
    assert isinstance(retriever.last_latency_ms, float)


def test_determinism_across_instances_and_calls():
    pairs = [("CHK-000002", "identical words here"),
             ("CHK-000001", "identical words here"),
             ("CHK-000003", "something else entirely")]
    first, _ = make_retriever(pairs)
    second, _ = make_retriever(pairs)
    for mode in ("bm25", "semantic", "hybrid"):
        assert (first.retrieve("identical words", top_k=3, mode=mode)
                == second.retrieve("identical words", top_k=3, mode=mode))


def test_hybrid_determinism_tie_break():
    # Identical texts tie-break to rank 1/2 on BOTH sides, so the fused
    # scores correctly differ (1/61+1/61 vs 1/62+1/62); ordering stays
    # deterministic with sequential ranks. Exact fused-score ties are
    # covered at the rrf_fuse level above.
    retriever, _ = make_retriever(
        [("CHK-000002", "identical words here"),
         ("CHK-000001", "identical words here")])
    hits = retriever.retrieve("identical words", top_k=2, mode="hybrid")
    assert [h["chunk_id"] for h in hits] == ["CHK-000001", "CHK-000002"]
    assert [h["rank"] for h in hits] == [1, 2]
    assert (hits
            == retriever.retrieve("identical words", top_k=2, mode="hybrid"))


# ---------------------------------------------------------------------------
# 8. Surface hygiene: no services/rerankers/heavy imports at top level
# ---------------------------------------------------------------------------
def test_no_service_or_rerank_imports():
    tree = ast.parse((ROOT / "src" / "retrieval" / "hybrid.py")
                     .read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(part.name.split(".")[0] for part in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.split(".")[0])
    banned = {"torch", "transformers", "sentence_transformers", "requests",
              "qdrant_client", "rank_bm25", "sklearn", "scipy", "urllib",
              "socket", "http", "httpx"}
    assert not (imported & banned), f"banned imports: {imported & banned}"
    assert imported <= {"time", "pathlib", "json", "re", "numpy",
                        "retrieval"}, (
        "unexpected imports: %s" % (imported,))
