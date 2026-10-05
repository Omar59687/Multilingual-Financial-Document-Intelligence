"""Phase-4 R2 tests: deterministic stdlib BM25 keyword retriever.

Per docs/RETRIEVAL_DESIGN.md sections 5 (BM25) and 9 (API shape for mode
``bm25``). Uses synthetic chunk records plus read-only targeting lookups
against ``data/retrieval/chunks.json``. Never reads ground_truth/ or
canonical/ content.
"""

import ast
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import retrieval.bm25 as bm25mod  # noqa: E402
from retrieval.bm25 import BM25Index, tokenize  # noqa: E402
from retrieval.chunking import normalize_text  # noqa: E402

CHUNKS_PATH = ROOT / "data" / "retrieval" / "chunks.json"


def make_record(chunk_id, text):
    return {"chunk_id": chunk_id, "text": text}


def make_index(pairs):
    records = [make_record(cid, text) for cid, text in pairs]
    return BM25Index().build(records), {r["chunk_id"]: r for r in records}


@pytest.fixture(scope="module")
def real_corpus():
    records = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    by_id = {record["chunk_id"]: record for record in records}
    return BM25Index().build(records), by_id


# ---------------------------------------------------------------------------
# 1. Tokenizer reuse (single owner: chunking.normalize_text)
# ---------------------------------------------------------------------------
def test_tokenize_matches_chunking_normalizer():
    samples = [
        "Invoice INV-2023-0106 Total",
        "  Mixed CASE  ",
        "رقم الفاتورة ١٢٣٤، الإجمالي؛ هل؟",
        "a\tb\nc",
        "",
        "،؛؟",
    ]
    for sample in samples:
        assert tokenize(sample) == normalize_text(sample)


def test_tokenize_delegates_to_normalize_text(monkeypatch):
    real = bm25mod.normalize_text
    calls = []

    def spy(text):
        calls.append(text)
        return real(text)

    monkeypatch.setattr(bm25mod, "normalize_text", spy)
    assert bm25mod.tokenize("Hello ١٢٣") == ["hello", "123"]
    assert calls == ["Hello ١٢٣"]


def test_constants_frozen():
    assert bm25mod.K1 == 1.2
    assert bm25mod.B == 0.75


# ---------------------------------------------------------------------------
# 2. Exact BM25 formula per design section 5
# ---------------------------------------------------------------------------
def test_bm25_formula_matches_design():
    index, _ = make_index([
        ("CHK-000001", "alpha alpha beta"),
        ("CHK-000002", "alpha gamma"),
    ])
    n = 2
    avgdl = (3 + 2) / 2
    idf_alpha = math.log(1.0 + (n - 2 + 0.5) / (2 + 0.5))
    idf_beta = math.log(1.0 + (n - 1 + 0.5) / (1 + 0.5))
    norm_d1 = 1.2 * (1.0 - 0.75 + 0.75 * (3 / avgdl))
    expected_d1 = (idf_alpha * 2 * 2.2 / (2 + norm_d1)
                   + idf_beta * 1 * 2.2 / (1 + norm_d1))
    scores = index.score("alpha beta")
    assert set(scores) == {"CHK-000001", "CHK-000002"}
    assert math.isclose(scores["CHK-000001"], expected_d1, rel_tol=1e-9)
    # Precomputed idf values are pinned on the index as well.
    assert math.isclose(index._idf["alpha"], idf_alpha, rel_tol=1e-12)
    assert math.isclose(index._idf["beta"], idf_beta, rel_tol=1e-12)


def test_idf_rarity_ordering():
    index, _ = make_index([
        ("CHK-000001", "zebrafish alpha"),
        ("CHK-000002", "common alpha"),
        ("CHK-000003", "common beta"),
        ("CHK-000004", "common gamma"),
    ])
    assert index._idf["zebrafish"] > index._idf["common"]
    results = index.retrieve(
        "zebrafish common", top_k=4,
        chunks_by_id={f"CHK-00000{i}": {} for i in (1, 2, 3, 4)})
    assert results[0]["chunk_id"] == "CHK-000001"


# ---------------------------------------------------------------------------
# 3/4. Identifier exactness (synthetic exactly-one + real corpus)
# ---------------------------------------------------------------------------
def test_identifier_exactness_synthetic():
    index, by_id = make_index([
        ("CHK-000001", "invoice INV-2023-0106 total 500"),
        ("CHK-000002", "invoice INV-2023-0107 total 2023"),
        ("CHK-000003", "annual report 2023 summary"),
    ])
    # The rare token "0106" occurs in exactly one chunk.
    assert index._df.get("0106") == 1
    results = index.retrieve("INV-2023-0106", top_k=3, chunks_by_id=by_id)
    assert results[0]["chunk_id"] == "CHK-000001"
    assert results[0]["rank"] == 1


def test_real_corpus_invoice_id_resolves_to_dev003(real_corpus):
    index, by_id = real_corpus
    holders = [cid for cid, rec in by_id.items()
               if "INV-2023-0106" in rec["text"]]
    assert holders, "expected INV-2023-0106 somewhere in frozen corpus"
    results = index.retrieve("INV-2023-0106", top_k=5, chunks_by_id=by_id)
    assert results
    top = by_id[results[0]["chunk_id"]]
    assert "INV-2023-0106" in top["text"]


# ---------------------------------------------------------------------------
# 5. Arabic-Indic digit parity (both directions)
# ---------------------------------------------------------------------------
def test_arabic_query_matches_western_chunk():
    index, by_id = make_index([
        ("CHK-000001", "رقم الفاتورة 1234 الإجمالي"),
        ("CHK-000002", "تقرير سنوي عن المبيعات"),
    ])
    assert tokenize("١٢٣٤") == ["1234"]
    results = index.retrieve("رقم الفاتورة ١٢٣٤", top_k=2, chunks_by_id=by_id)
    assert results and results[0]["chunk_id"] == "CHK-000001"


def test_western_query_matches_arabic_indic_chunk():
    index, by_id = make_index([
        ("CHK-000001", "رقم الفاتورة ١٢٣٤ الإجمالي"),
        ("CHK-000002", "تقرير سنوي عن المبيعات"),
    ])
    assert tokenize("1234") == tokenize("١٢٣٤") == ["1234"]
    results = index.retrieve("1234", top_k=2, chunks_by_id=by_id)
    assert results and results[0]["chunk_id"] == "CHK-000001"


# ---------------------------------------------------------------------------
# Determinism + tie-break
# ---------------------------------------------------------------------------
def test_determinism_build_and_score_twice():
    pairs = [
        ("CHK-000002", "identical words here"),
        ("CHK-000001", "identical words here"),
        ("CHK-000003", "something else entirely"),
    ]
    first = BM25Index().build([make_record(cid, text) for cid, text in pairs])
    second = BM25Index().build([make_record(cid, text) for cid, text in pairs])
    assert first.score("identical words") == second.score("identical words")
    assert (first.retrieve("identical words", top_k=3)
            == second.retrieve("identical words", top_k=3)
            == first.retrieve("identical words", top_k=3))


def test_tie_break_score_desc_chunk_id_asc():
    index, by_id = make_index([
        ("CHK-000002", "identical words here"),
        ("CHK-000001", "identical words here"),
    ])
    results = index.retrieve("identical words", top_k=2, chunks_by_id=by_id)
    assert [r["chunk_id"] for r in results] == ["CHK-000001", "CHK-000002"]
    assert results[0]["score"] == results[1]["score"]
    assert [r["rank"] for r in results] == [1, 2]


# ---------------------------------------------------------------------------
# Empty / unknown / malformed guards + top_k bounds + result shape
# ---------------------------------------------------------------------------
def test_empty_blank_unknown_queries():
    index, by_id = make_index([
        ("CHK-000001", "hello world"),
        ("CHK-000002", "goodbye world"),
    ])
    assert index.retrieve("", top_k=5, chunks_by_id=by_id) == []
    assert index.retrieve("   ", top_k=5, chunks_by_id=by_id) == []
    assert index.retrieve("،؛؟", top_k=5, chunks_by_id=by_id) == []
    assert index.retrieve("qqqzzzxxxy", top_k=5, chunks_by_id=by_id) == []
    assert index.score("qqqzzzxxxy") == {}
    assert index.score("   ") == {}


def test_malformed_inputs_raise_type_error():
    index, by_id = make_index([("CHK-000001", "hello world")])
    with pytest.raises(TypeError):
        index.score(123)
    with pytest.raises(TypeError):
        index.score(None)
    with pytest.raises(TypeError):
        index.retrieve(123, top_k=1, chunks_by_id=by_id)
    with pytest.raises(TypeError):
        BM25Index().build("not a list")
    with pytest.raises(TypeError):
        BM25Index().build({"chunk_id": "CHK-1", "text": "x"})
    with pytest.raises(TypeError):
        BM25Index().build([{"chunk_id": "CHK-1"}])
    with pytest.raises(TypeError):
        BM25Index().build([{"text": "x"}])
    with pytest.raises(TypeError):
        BM25Index().build([make_record("CHK-1", "a"),
                           make_record("CHK-1", "b")])
    with pytest.raises(TypeError):
        index.retrieve("hello", top_k=1, chunks_by_id=["not", "a", "dict"])


@pytest.mark.parametrize("bad_top_k", [0, -1, 101, 1000, "5", 2.5, None, True])
def test_top_k_bounds_enforced(bad_top_k):
    index, by_id = make_index([("CHK-000001", "hello world")])
    with pytest.raises(TypeError):
        index.retrieve("hello", top_k=bad_top_k, chunks_by_id=by_id)


def test_top_k_valid_bounds_and_truncation():
    index, by_id = make_index([
        ("CHK-000001", "alpha one"),
        ("CHK-000002", "alpha two"),
        ("CHK-000003", "alpha three"),
    ])
    one = index.retrieve("alpha", top_k=1, chunks_by_id=by_id)
    assert len(one) == 1 and one[0]["rank"] == 1
    capped = index.retrieve("alpha", top_k=100, chunks_by_id=by_id)
    assert len(capped) == 3
    assert [r["rank"] for r in capped] == [1, 2, 3]
    two = index.retrieve("alpha", top_k=2, chunks_by_id=by_id)
    assert [r["rank"] for r in two] == [1, 2]


def test_result_shape_and_source():
    index, by_id = make_index([
        ("CHK-000001", "alpha one"),
        ("CHK-000002", "alpha two"),
    ])
    results = index.retrieve("alpha", top_k=2, chunks_by_id=by_id)
    assert len(results) == 2
    for result in results:
        assert set(result) == {"chunk_id", "score", "rank", "source"}
        assert result["source"] == "bm25"
        assert isinstance(result["score"], float)
        assert isinstance(result["rank"], int)


def test_stdlib_only():
    tree = ast.parse((ROOT / "src" / "retrieval" / "bm25.py")
                     .read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(part.name.split(".")[0] for part in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.split(".")[0])
    banned = {"torch", "numpy", "scipy", "sklearn", "requests", "networkx",
              "rank_bm25", "transformers", "sentence_transformers", "qdrant_client"}
    assert not (imported & banned), f"banned imports: {imported & banned}"
    assert imported <= {"math", "retrieval"}, f"unexpected imports: {imported}"


# ---------------------------------------------------------------------------
# R2 recovery: exact identifier gating (review findings 1 + 2)
# ---------------------------------------------------------------------------
def _id_holders_synthetic():
    return make_index([
        ("CHK-000001", "invoice INV-2023-0106 total 500"),
        ("CHK-000002", "invoice INV-2023-0107 total 700"),
        ("CHK-000003", "annual report 2023 summary"),
    ])


def test_nonexistent_identifier_returns_no_hits():
    index, by_id = _id_holders_synthetic()
    assert index.score("INV-2023-9999") == {}
    assert index.retrieve("INV-2023-9999", top_k=3, chunks_by_id=by_id) == []
    # Deterministic: repeated calls agree.
    assert index.retrieve("INV-2023-9999", top_k=3, chunks_by_id=by_id) == []


def test_near_match_identifier_returns_only_exact_holder():
    index, by_id = _id_holders_synthetic()
    first = index.retrieve("INV-2023-0106", top_k=3, chunks_by_id=by_id)
    assert [hit["chunk_id"] for hit in first] == ["CHK-000001"]
    second = index.retrieve("INV-2023-0107", top_k=3, chunks_by_id=by_id)
    assert [hit["chunk_id"] for hit in second] == ["CHK-000002"]


def test_identifier_query_with_extra_terms_scores_normally():
    index, by_id = _id_holders_synthetic()
    mixed = index.retrieve("INV-2023-0106 total", top_k=3, chunks_by_id=by_id)
    assert [hit["chunk_id"] for hit in mixed] == ["CHK-000001"]
    # Unknown extra terms never error and never hide the holder.
    unknown = index.retrieve("INV-2023-0106 qqqzzzxxxy", top_k=3,
                             chunks_by_id=by_id)
    assert [hit["chunk_id"] for hit in unknown] == ["CHK-000001"]
    # Plain keyword queries (no identifier-like part) keep OR semantics.
    assert sorted(index.score("2023")) == ["CHK-000001", "CHK-000002",
                                           "CHK-000003"]


def _run_holders(by_id, run):
    holders = set()
    for chunk_id, record in by_id.items():
        tokens = tokenize(record["text"])
        width = len(run)
        if any(tokens[i:i + width] == run
               for i in range(len(tokens) - width + 1)):
            holders.add(chunk_id)
    return holders


def test_real_corpus_exact_id_returns_only_holders(real_corpus):
    index, by_id = real_corpus
    run = ["inv", "2023", "0106"]
    holders = _run_holders(by_id, run)
    assert holders, "expected at least one holder in frozen corpus"
    results = index.retrieve("INV-2023-0106", top_k=10, chunks_by_id=by_id)
    assert {hit["chunk_id"] for hit in results} == holders
    assert results[0]["chunk_id"] in holders


def test_real_corpus_nonexistent_id_returns_no_hits(real_corpus):
    index, by_id = real_corpus
    assert index.retrieve("INV-2023-9999", top_k=10, chunks_by_id=by_id) == []
    assert index.score("INV-2023-9999") == {}


def test_real_corpus_full_question_resolves_dev003(real_corpus):
    index, by_id = real_corpus
    results = index.retrieve("invoice INV-2023-0106 total", top_k=5,
                             chunks_by_id=by_id)
    assert results
    assert by_id[results[0]["chunk_id"]]["document_id"] == "DEV-003"
