"""R6 scorer tests: G2 retrieval-measurement math + gating (Phase 4 R6).

Per docs/RETRIEVAL_DESIGN.md section 10 and docs/EVALUATION_PLAN.md
sections 4/7/11. Synthetic inputs only; stdlib plus the pure scorer
module (no retrieval machinery anywhere in this file, per the
retrieval-blindness rule asserted by test_gold_integrity.py).
"""

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from evaluation import g2_scoring as G2  # noqa: E402


class FakeRetriever:
    """Recording stub of the R4 unified interface (no retrieval import)."""

    def __init__(self, hits_by_mode):
        self.calls = []
        self.hits_by_mode = hits_by_mode

    def retrieve(self, query, top_k=10, mode="hybrid"):
        self.calls.append({"query": query, "top_k": top_k, "mode": mode})
        return [dict(hit) for hit in self.hits_by_mode[mode]]


def make_question(qid="Q-DEV-001", text="What was net income?",
                  targets=None, lang="en", category="EN->EN", doc="DEV-001"):
    return {"qid": qid, "text": text, "lang": lang, "category": category,
            "doc": doc, "split": "dev",
            "targets": targets if targets is not None else
            [{"chunk_id": "CHK-a", "grade": "primary"}]}


# ---------------------------------------------------------------------------
# recall@K (primary)
# ---------------------------------------------------------------------------
def test_recall_at_k_hand_computed():
    retrieved = ["CHK-x", "CHK-a", "CHK-y", "CHK-b"]
    primary = ["CHK-a", "CHK-b", "CHK-c"]
    assert G2.recall_at_k(retrieved, primary, 5) == pytest.approx(2 / 3)
    assert G2.recall_at_k(retrieved, primary, 2) == pytest.approx(1 / 3)
    assert G2.recall_at_k(retrieved, primary, 1) == 0.0
    assert G2.recall_at_k([], primary, 10) == 0.0
    assert G2.recall_at_k(retrieved, [], 10) == 0.0


@pytest.mark.parametrize("bad_k", [0, -1, "5", 2.5, None, True])
def test_recall_at_k_guards(bad_k):
    with pytest.raises(TypeError):
        G2.recall_at_k(["CHK-a"], ["CHK-a"], bad_k)


# ---------------------------------------------------------------------------
# NDCG@K (graded)
# ---------------------------------------------------------------------------
def test_ndcg_at_k_hand_computed():
    gains = {"CHK-a": 2, "CHK-b": 1}
    dcg = 1 / math.log2(2) + 2 / math.log2(3)
    idcg = 2 / math.log2(2) + 1 / math.log2(3)
    assert G2.ndcg_at_k(["CHK-b", "CHK-a", "CHK-x"], gains, 10) == \
        pytest.approx(dcg / idcg)
    assert G2.ndcg_at_k(["CHK-a", "CHK-b"], gains, 10) == pytest.approx(1.0)
    assert G2.ndcg_at_k(["CHK-x", "CHK-y"], gains, 10) == 0.0
    assert G2.ndcg_at_k(["CHK-a"], {}, 10) == 0.0
    # Truncation: K=1 keeps only the first rank.
    assert G2.ndcg_at_k(["CHK-b", "CHK-a"], gains, 1) == \
        pytest.approx((1 / math.log2(2)) / (2 / math.log2(2)))


@pytest.mark.parametrize("bad_k", [0, -1, "10", None, False])
def test_ndcg_at_k_guards(bad_k):
    with pytest.raises(TypeError):
        G2.ndcg_at_k(["CHK-a"], {"CHK-a": 2}, bad_k)
    with pytest.raises(TypeError):
        G2.ndcg_at_k(["CHK-a"], ["not-a-dict"], 10)


# ---------------------------------------------------------------------------
# score_question
# ---------------------------------------------------------------------------
def test_score_question_mixed_grades():
    retrieved = ["CHK-p1", "CHK-x", "CHK-s1", "CHK-p2"]
    targets = [{"chunk_id": "CHK-p1", "grade": "primary"},
               {"chunk_id": "CHK-p2", "grade": "primary"},
               {"chunk_id": "CHK-s1", "grade": "acceptable"}]
    scored = G2.score_question(retrieved, targets)
    assert scored["recall@5"] == pytest.approx(1.0)
    assert scored["recall@10"] == pytest.approx(1.0)
    assert scored["n_primary"] == 2 and scored["n_targets"] == 3
    dcg = 2 / math.log2(2) + 1 / math.log2(4) + 2 / math.log2(5)
    idcg = 2 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
    assert scored["ndcg@10"] == pytest.approx(dcg / idcg)


def test_score_question_unknown_grade_rejected():
    with pytest.raises(TypeError):
        G2.score_question(["CHK-a"],
                          [{"chunk_id": "CHK-a", "grade": "gold"}])
    with pytest.raises(TypeError):
        G2.score_question("not-a-list", [])
    with pytest.raises(TypeError):
        G2.score_question([], "not-a-list")


# ---------------------------------------------------------------------------
# latency percentiles
# ---------------------------------------------------------------------------
def test_percentile_nearest_rank():
    values = list(range(1, 101))
    assert G2.percentile_nearest_rank(values, 50) == 50.0
    assert G2.percentile_nearest_rank(values, 95) == 95.0
    assert G2.percentile_nearest_rank(values, 0) == 1.0
    assert G2.percentile_nearest_rank(values, 100) == 100.0
    assert G2.percentile_nearest_rank([], 50) is None
    with pytest.raises(TypeError):
        G2.percentile_nearest_rank(values, 101)
    with pytest.raises(TypeError):
        G2.percentile_nearest_rank(values, "50")


def test_latency_summary():
    assert G2.latency_summary([]) == {"n": 0, "p50": None, "p95": None,
                                      "min": None, "max": None}
    summary = G2.latency_summary([10.0, 20.0, 30.0, 40.0])
    assert summary["n"] == 4
    assert summary["p50"] == 20.0  # nearest-rank ceil(.5*4)=2nd value
    assert summary["p95"] == 40.0
    assert summary["min"] == 10.0 and summary["max"] == 40.0
    with pytest.raises(TypeError):
        G2.latency_summary([10.0, "slow"])


# ---------------------------------------------------------------------------
# verification gating + slices
# ---------------------------------------------------------------------------
def _rows():
    return [
        {"qid": "Q-1", "lang": "en", "category": "c",
         "verified": True, "verified_by": "human",
         "verified_date": "2026-10-06",
         "recall@5": 1.0, "recall@10": 1.0, "ndcg@10": 1.0},
        {"qid": "Q-2", "lang": "ar", "category": "c",
         "verified": False, "verified_by": None, "verified_date": None,
         "recall@5": 0.0, "recall@10": 0.0,
         "ndcg@10": 0.0},
    ]


def test_aggregate_excludes_unverified_by_default():
    agg = G2.aggregate_metric(_rows(), "recall@10")
    assert agg == {"mean": 1.0, "n": 1, "n_excluded": 1, "reason": None}
    empty = G2.aggregate_metric([_rows()[1]], "recall@10")
    assert empty["mean"] is None and empty["n"] == 0
    assert "G1" in empty["reason"]
    provisional = G2.aggregate_metric(_rows(), "recall@10",
                                      verified_only=False)
    assert provisional["mean"] == pytest.approx(0.5)
    assert provisional["n"] == 2 and provisional["reason"] is None


def test_slice_report_overall_lang_category_minimum():
    report = G2.slice_report(_rows())
    assert report["overall"]["recall@10"] == {
        "mean": 1.0, "n": 1, "n_excluded": 1, "reason": None}
    assert report["by_lang"]["en"]["recall@10"]["mean"] == 1.0
    assert report["by_lang"]["ar"]["recall@10"]["mean"] is None
    assert report["by_category"]["c"]["recall@10"]["mean"] == 1.0
    # R6 recovery (MAJOR 2): a missing required slice must NOT be dropped.
    # The minimum stays unavailable with an explicit incomplete-coverage
    # reason while any required language lacks verified evidence.
    assert report["min_lang_slice"]["recall@10"] is None
    assert report["min_lang_slice"]["ndcg@10"] is None
    assert "incomplete" in report["min_lang_slice_reason"]["recall@10"].lower()
    assert report["min_lang_slice_missing"]["recall@10"] == ["ar"]


def test_slice_report_all_unverified_minimum_none():
    report = G2.slice_report([_rows()[1]])
    assert report["overall"]["recall@10"]["mean"] is None
    assert report["min_lang_slice"]["recall@10"] is None


# ---------------------------------------------------------------------------
# measure_split wiring (recording fake; gold IDs must never reach retrieval)
# ---------------------------------------------------------------------------
def test_measure_split_passes_only_text_to_retrieval():
    targets = [{"chunk_id": "CHK-secret-1", "grade": "primary"},
               {"chunk_id": "CHK-secret-2", "grade": "acceptable"}]
    question = make_question(targets=targets)
    hits = {"bm25": [{"chunk_id": "CHK-secret-1", "score": 1.0, "rank": 1,
                      "source": "bm25"}],
            "semantic": [{"chunk_id": "CHK-secret-2", "score": 0.9, "rank": 1,
                          "source": "semantic"}],
            "hybrid": [{"chunk_id": "CHK-secret-1", "score": 0.03, "rank": 1,
                        "source": "hybrid"}]}
    fake = FakeRetriever(hits)
    evidence = G2.measure_split(
        fake, [question], {"Q-DEV-001": {"verified": True,
                                         "verified_by": "human",
                                         "verified_date": "2026-10-06"}})
    assert len(fake.calls) == 3
    for call in fake.calls:
        assert call["query"] == question["text"]
        assert call["top_k"] == G2.K_TOP == 10
        assert call["mode"] in ("bm25", "semantic", "hybrid")
        # Leakage guard: no gold chunk ID may enter a retrieval call.
        assert "CHK-secret-1" not in call["query"]
        assert "CHK-secret-2" not in call["query"]
    (row,) = evidence["questions"]
    assert row["qid"] == "Q-DEV-001" and row["verified"] is True
    assert row["targets"] == targets
    assert row["per_mode"]["bm25"]["retrieved_ids"] == ["CHK-secret-1"]
    assert row["per_mode"]["bm25"]["scores_provisional"]["recall@10"] == 1.0


def test_measure_split_deterministic_modulo_latency():
    question = make_question()
    hits = {mode: [{"chunk_id": "CHK-a", "score": 1.0, "rank": 1,
                    "source": mode}] for mode in G2.MODES}
    first = G2.measure_split(FakeRetriever(hits), [question], {})

    def strip(evidence):
        scrubbed = []
        for row in evidence["questions"]:
            entry = {k: v for k, v in row.items() if k != "per_mode"}
            modes = {}
            for mode, detail in row["per_mode"].items():
                modes[mode] = {k: v for k, v in detail.items()
                               if k != "latency_ms"}
            entry["per_mode"] = modes
            scrubbed.append(entry)
        return scrubbed

    second = G2.measure_split(FakeRetriever(hits), [question], {})
    assert strip(first) == strip(second)


def test_measure_split_malformed_inputs():
    fake = FakeRetriever({mode: [] for mode in G2.MODES})
    with pytest.raises(TypeError):
        G2.measure_split(fake, "not-a-list", {})
    with pytest.raises(TypeError):
        G2.measure_split(fake, [{"qid": "Q-1"}], {})
    with pytest.raises(TypeError):
        G2.measure_split(fake, [make_question()], ["not-a-dict"])


def test_build_mode_reports_from_evidence():
    question = make_question()
    hits = {mode: [{"chunk_id": "CHK-a", "score": 1.0, "rank": 1,
                    "source": mode}] for mode in G2.MODES}
    evidence = G2.measure_split(
        FakeRetriever(hits), [question], {"Q-DEV-001": {"verified": True,
                                                       "verified_by": "human",
                                                       "verified_date": "2026-10-06"}})
    reports = G2.build_mode_reports(evidence)
    assert set(reports) == {"bm25", "semantic", "hybrid"}
    for mode in G2.MODES:
        assert reports[mode]["slices"]["overall"]["recall@10"] == {
            "mean": 1.0, "n": 1, "n_excluded": 0, "reason": None}
        assert reports[mode]["latency"]["n"] == 1


def test_build_mode_reports_unverified_excluded():
    question = make_question()
    hits = {mode: [{"chunk_id": "CHK-a", "score": 1.0, "rank": 1,
                    "source": mode}] for mode in G2.MODES}
    evidence = G2.measure_split(FakeRetriever(hits), [question], {})
    reports = G2.build_mode_reports(evidence)
    for mode in G2.MODES:
        agg = reports[mode]["slices"]["overall"]["recall@10"]
        assert agg["mean"] is None and agg["n"] == 0
