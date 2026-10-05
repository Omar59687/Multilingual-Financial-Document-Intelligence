"""R6 bounded-recovery regression: sign-off gating + language minimum.

Phase 4 R6 recovery only. Synthetic inputs; stdlib plus the pure scorer
module (no retrieval machinery in this file, per the retrieval-blindness
rule asserted by test_gold_integrity.py).

MAJOR 1: verified=true alone must not enter official aggregates; valid
verified_by + verified_date sign-off is required (gold verification
schema). Missing/null/blank sign-off is excluded; verified=false stays
excluded; provisional question-level evidence is preserved.

MAJOR 2: the minimum across required language slices must stay
unavailable/null with an explicit incomplete-coverage reason while any
required language lacks verified evidence; only when every required
slice is populated is the minimum computed normally. Uses the
repository-defined REQUIRED_LANGS; no invented languages.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from evaluation import g2_scoring as G2  # noqa: E402


def _row(qid, lang, score, verified=True, by="human", date="2026-10-06"):
    row = {"qid": qid, "lang": lang, "category": "c",
           "verified": verified, "verified_by": by, "verified_date": date,
           "recall@5": score, "recall@10": score, "ndcg@10": score}
    return row


def _valid_record():
    return {"verified": True, "verified_by": "human",
            "verified_date": "2026-10-06"}


# ---------------------------------------------------------------------------
# MAJOR 1 — verification sign-off enforcement
# ---------------------------------------------------------------------------

def test_major1_valid_signoff_included():
    rows = [_row("Q-1", "en", 1.0, verified=True,
                 by="human", date="2026-10-06")]
    agg = G2.aggregate_metric(rows, "recall@10")
    assert agg["mean"] == 1.0 and agg["n"] == 1 and agg["reason"] is None
    assert G2.is_verified_record(_valid_record()) is True


def test_major1_missing_verified_by_excluded():
    rec = {"verified": True, "verified_date": "2026-10-06"}
    assert G2.is_verified_record(rec) is False
    rows = [_row("Q-1", "en", 1.0, verified=True,
                 by="human", date="2026-10-06")]
    rows[0].pop("verified_by")
    agg = G2.aggregate_metric(rows, "recall@10")
    assert agg["mean"] is None and agg["n"] == 0


def test_major1_null_verified_by_excluded():
    rec = {"verified": True, "verified_by": None,
           "verified_date": "2026-10-06"}
    assert G2.is_verified_record(rec) is False
    rows = [_row("Q-1", "en", 1.0, by=None)]
    agg = G2.aggregate_metric(rows, "recall@10")
    assert agg["mean"] is None and agg["n"] == 0


def test_major1_blank_verified_by_excluded():
    for blank in ("", "   ", "\t\n "):
        rec = {"verified": True, "verified_by": blank,
               "verified_date": "2026-10-06"}
        assert G2.is_verified_record(rec) is False, repr(blank)
        rows = [_row("Q-1", "en", 1.0, by=blank)]
        agg = G2.aggregate_metric(rows, "recall@10")
        assert agg["mean"] is None and agg["n"] == 0, repr(blank)


def test_major1_missing_verified_date_excluded():
    rec = {"verified": True, "verified_by": "human"}
    assert G2.is_verified_record(rec) is False
    rows = [_row("Q-1", "en", 1.0, by="human", date="2026-10-06")]
    rows[0].pop("verified_date")
    agg = G2.aggregate_metric(rows, "recall@10")
    assert agg["mean"] is None and agg["n"] == 0


def test_major1_null_verified_date_excluded():
    rec = {"verified": True, "verified_by": "human",
           "verified_date": None}
    assert G2.is_verified_record(rec) is False
    rows = [_row("Q-1", "en", 1.0, date=None)]
    agg = G2.aggregate_metric(rows, "recall@10")
    assert agg["mean"] is None and agg["n"] == 0


def test_major1_blank_verified_date_excluded():
    for blank in ("", "   ", " \t\n"):
        rec = {"verified": True, "verified_by": "human",
               "verified_date": blank}
        assert G2.is_verified_record(rec) is False, repr(blank)
        rows = [_row("Q-1", "en", 1.0, date=blank)]
        agg = G2.aggregate_metric(rows, "recall@10")
        assert agg["mean"] is None and agg["n"] == 0, repr(blank)


def test_major1_verified_false_excluded():
    rec = {"verified": False, "verified_by": None,
           "verified_date": None}
    assert G2.is_verified_record(rec) is False
    rows = [_row("Q-1", "en", 1.0, verified=False,
                 by=None, date=None)]
    agg = G2.aggregate_metric(rows, "recall@10")
    assert agg["mean"] is None and agg["n"] == 0
    assert "G1" in agg["reason"]
    # Provisional view still includes it when explicitly requested.
    prov = G2.aggregate_metric(rows, "recall@10", verified_only=False)
    assert prov["n"] == 1 and prov["reason"] is None


def test_major1_measure_split_gating_and_provisional_preserved():
    class _Fake:
        def __init__(self, hits):
            self.hits = hits

        def retrieve(self, query, top_k=10, mode="hybrid"):
            return [dict(h) for h in self.hits]

    def _q(qid="Q-DEV-001", lang="en"):
        return {"qid": qid, "text": "What was net income?",
                "lang": lang, "category": "c", "doc": "DEV-001",
                "split": "dev",
                "targets": [{"chunk_id": "CHK-a", "grade": "primary"}]}

    hits = [{"chunk_id": "CHK-a", "score": 1.0, "rank": 1,
             "source": "hybrid"}]
    # Valid sign-off -> verified.
    ev = G2.measure_split(_Fake(hits), [_q()],
                          {"Q-DEV-001": _valid_record()})
    assert ev["questions"][0]["verified"] is True
    assert ev["questions"][0]["per_mode"]["hybrid"][
        "scores_provisional"]["recall@10"] == 1.0
    # Each malformed variant -> unverified but provisional preserved.
    bad_variants = [
        {"verified": True},
        {"verified": True, "verified_by": None,
         "verified_date": "2026-10-06"},
        {"verified": True, "verified_by": "   ",
         "verified_date": "2026-10-06"},
        {"verified": True, "verified_by": "human",
         "verified_date": None},
        {"verified": True, "verified_by": "human",
         "verified_date": "  "},
        {"verified": True, "verified_by": "human"},
        {"verified": False, "verified_by": None,
         "verified_date": None},
    ]
    for bad in bad_variants:
        ev = G2.measure_split(_Fake(hits), [_q()], {"Q-DEV-001": bad})
        row = ev["questions"][0]
        assert row["verified"] is False, bad
        # Provisional question-level evidence preserved.
        assert row["per_mode"]["hybrid"]["scores_provisional"][
            "recall@10"] == 1.0
        reports = G2.build_mode_reports(ev)
        agg = reports["hybrid"]["slices"]["overall"]["recall@10"]
        assert agg["mean"] is None and agg["n"] == 0, bad


# ---------------------------------------------------------------------------
# MAJOR 2 — language minimum requires complete coverage
# ---------------------------------------------------------------------------

def test_major2_uses_required_language_set():
    assert tuple(G2.REQUIRED_LANGS) == ("ar", "en")


def test_major2_all_slices_populated_minimum_normal():
    rows = [_row("Q-EN", "en", 0.4), _row("Q-AR", "ar", 0.9)]
    report = G2.slice_report(rows)
    assert report["by_lang"]["en"]["recall@10"]["mean"] == 0.4
    assert report["by_lang"]["ar"]["recall@10"]["mean"] == 0.9
    assert report["min_lang_slice"]["recall@10"] == 0.4
    assert report["min_lang_slice_reason"]["recall@10"] is None
    assert report["min_lang_slice_missing"]["recall@10"] == []


def test_major2_unequal_populated_minimum_computed():
    rows = [_row("Q-EN", "en", 1.0), _row("Q-AR", "ar", 0.25)]
    report = G2.slice_report(rows)
    assert report["min_lang_slice"]["recall@10"] == 0.25
    assert report["min_lang_slice"]["recall@5"] == 0.25
    assert report["min_lang_slice"]["ndcg@10"] == 0.25
    assert report["min_lang_slice_reason"]["recall@10"] is None


def test_major2_arabic_missing_minimum_unavailable():
    # Independently reproduced shape: EN verified mean 1.0, AR count 0.
    rows = [_row("Q-EN", "en", 1.0)]
    report = G2.slice_report(rows)
    assert report["by_lang"]["en"]["recall@10"]["mean"] == 1.0
    assert report["min_lang_slice"]["recall@10"] is None
    reason = report["min_lang_slice_reason"]["recall@10"]
    assert isinstance(reason, str) and "incomplete" in reason.lower()
    assert "ar" in reason
    assert report["min_lang_slice_missing"]["recall@10"] == ["ar"]


def test_major2_english_missing_minimum_unavailable():
    rows = [_row("Q-AR", "ar", 0.75)]
    report = G2.slice_report(rows)
    assert report["min_lang_slice"]["recall@10"] is None
    reason = report["min_lang_slice_reason"]["recall@10"]
    assert isinstance(reason, str) and "incomplete" in reason.lower()
    assert "en" in reason
    assert report["min_lang_slice_missing"]["recall@10"] == ["en"]


def test_major2_all_missing_minimum_unavailable():
    rows = [_row("Q-X", "en", 0.0, verified=False, by=None, date=None)]
    report = G2.slice_report(rows)
    for metric in ("recall@5", "recall@10", "ndcg@10"):
        assert report["min_lang_slice"][metric] is None
        reason = report["min_lang_slice_reason"][metric]
        assert isinstance(reason, str) and "incomplete" in reason.lower()
        assert set(report["min_lang_slice_missing"][metric]) == {"ar", "en"}
    empty = G2.slice_report([])
    for metric in ("recall@5", "recall@10", "ndcg@10"):
        assert empty["min_lang_slice"][metric] is None
        assert "incomplete" in empty["min_lang_slice_reason"][metric].lower()
