"""R6 runner wiring tests: R4 interface + split discipline (Phase 4 R6).

Exercises evaluation.g2_scoring.measure_split against the accepted R4
UnifiedRetriever with a deterministic stub embedding model (no weight
download), plus split-separation and verification-gating behavior.
Synthetic inventory/gold only; never ground_truth/ or canonical/.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from evaluation import g2_scoring as G2  # noqa: E402
from retrieval import embeddings as E  # noqa: E402
from retrieval.hybrid import UnifiedRetriever  # noqa: E402


class StubModel:
    """Deterministic stub embedding model (no download, no torch)."""

    def __init__(self, dim=8):
        self.dim = dim

    def eval(self):
        return None

    def encode(self, texts, **kwargs):
        import hashlib as _hashlib

        rows = []
        for text in texts:
            digest = _hashlib.sha256(text.encode("utf-8")).digest()
            row = np.array([b / 255.0 for b in digest[:self.dim]],
                           dtype=float)
            norm = float(np.linalg.norm(row))
            rows.append(row / norm if norm else row)
        return np.stack(rows)


def make_retriever(pairs):
    records = [{"chunk_id": cid, "text": text} for cid, text in pairs]
    stub = StubModel()
    vectors = E.build_vectors(records, model=stub)
    return UnifiedRetriever(records, vectors=vectors, model=stub), records


def make_gold(dev_text="alpha one", eval_text="alpha two"):
    dev = [{"qid": "Q-DEV-001", "text": dev_text, "lang": "en",
            "category": "t", "doc": "D", "split": "dev",
            "targets": [{"chunk_id": "CHK-000001", "grade": "primary"}]}]
    ev = [{"qid": "Q-EVAL-001", "text": eval_text, "lang": "en",
           "category": "t", "doc": "D", "split": "eval",
           "targets": [{"chunk_id": "CHK-000002", "grade": "primary"}]}]
    verification = {"Q-DEV-001": {"verified": False,
                                  "verified_by": None,
                                  "verified_date": None},
                    "Q-EVAL-001": {"verified": False,
                                   "verified_by": None,
                                   "verified_date": None}}
    return dev, ev, verification


def strip_latency(evidence):
    scrubbed = []
    for row in evidence["questions"]:
        entry = {k: v for k, v in row.items() if k != "per_mode"}
        entry["per_mode"] = {
            mode: {k: v for k, v in detail.items() if k != "latency_ms"}
            for mode, detail in row["per_mode"].items()}
        scrubbed.append(entry)
    return scrubbed


def test_runner_evidence_shape_and_resolvable_hits():
    retriever, records = make_retriever([("CHK-000001", "alpha one"),
                                         ("CHK-000002", "alpha two"),
                                         ("CHK-000003", "unrelated")])
    by_id = {r["chunk_id"]: r for r in records}
    dev, _, verification = make_gold()
    evidence = G2.measure_split(retriever, dev, verification)
    assert evidence["modes"] == ["bm25", "semantic", "hybrid"]
    assert evidence["top_k"] == 10
    (row,) = evidence["questions"]
    assert row["qid"] == "Q-DEV-001" and row["verified"] is False
    assert set(row["per_mode"]) == {"bm25", "semantic", "hybrid"}
    for mode in G2.MODES:
        entry = row["per_mode"][mode]
        assert len(entry["retrieved_ids"]) <= 10
        assert len(entry["retrieved_ids"]) == len(entry["ranks"])
        assert entry["sources"] == [mode] * len(entry["retrieved_ids"])
        assert entry["latency_ms"] >= 0.0
        for cid in entry["retrieved_ids"]:
            assert cid in by_id  # every hit resolves via get_chunk
        for metric in ("recall@5", "recall@10", "ndcg@10"):
            assert metric in entry["scores_provisional"]


def test_runner_deterministic_across_instances():
    pairs = [("CHK-000001", "alpha one"),
             ("CHK-000002", "alpha two")]
    first, _ = make_retriever(pairs)
    second, _ = make_retriever(pairs)
    dev, _, verification = make_gold()
    assert (strip_latency(G2.measure_split(first, dev, verification))
            == strip_latency(G2.measure_split(second, dev, verification)))


def test_splits_measured_separately_never_pooled():
    retriever, _ = make_retriever([("CHK-000001", "alpha one"),
                                   ("CHK-000002", "alpha two")])
    dev, ev, verification = make_gold()
    dev_evidence = G2.measure_split(retriever, dev, verification)
    eval_evidence = G2.measure_split(retriever, ev, verification)
    assert [q["qid"] for q in dev_evidence["questions"]] == ["Q-DEV-001"]
    assert [q["qid"] for q in eval_evidence["questions"]] == ["Q-EVAL-001"]
    dev_reports = G2.build_mode_reports(dev_evidence)
    eval_reports = G2.build_mode_reports(eval_evidence)
    # Each report covers exactly its own split (n=1 question per mode).
    for reports in (dev_reports, eval_reports):
        for mode in G2.MODES:
            assert reports[mode]["latency"]["n"] == 1


def test_unverified_splits_report_null_with_reason():
    retriever, _ = make_retriever([("CHK-000001", "alpha one"),
                                   ("CHK-000002", "alpha two")])
    dev, _, verification = make_gold()
    evidence = G2.measure_split(retriever, dev, verification)
    reports = G2.build_mode_reports(evidence)
    for mode in G2.MODES:
        for metric in ("recall@5", "recall@10", "ndcg@10"):
            agg = reports[mode]["slices"]["overall"][metric]
            assert agg["mean"] is None and agg["n"] == 0
            assert "G1" in agg["reason"]
    # Flipping one item to verified (the G1 action) includes it.
    verification["Q-DEV-001"] = {"verified": True,
                                 "verified_by": "human",
                                 "verified_date": "2026-10-06"}
    evidence = G2.measure_split(retriever, dev, verification)
    reports = G2.build_mode_reports(evidence)
    for mode in G2.MODES:
        agg = reports[mode]["slices"]["overall"]["recall@10"]
        assert agg["n"] == 1 and agg["reason"] is None


def test_empty_split_evidence():
    retriever, _ = make_retriever([("CHK-000001", "alpha one")])
    evidence = G2.measure_split(retriever, [], {})
    assert evidence["questions"] == []
    reports = G2.build_mode_reports(evidence)
    for mode in G2.MODES:
        assert reports[mode]["latency"] == {"n": 0, "p50": None,
                                            "p95": None, "min": None,
                                            "max": None}
        assert (reports[mode]["slices"]["overall"]["recall@10"]["mean"]
                is None)
