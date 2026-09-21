"""Score Kaggle experiment results with the Phase-2 benchmark tooling.

Usage (from repo root):
    python scripts/score_ocr_results.py data/benchmark/results/<run>.json
    python scripts/score_ocr_results.py <run>.json --out <summary>.json

Reads machine-readable results in the standard contract
(src/ocrbench/schema.py) and the derived benchmark truth. Prints
per-document metrics plus careful aggregate summaries — one aggregate per
metric, NEVER a single combined "overall AI score". The ±0.5% numeric
figure is a secondary diagnostic; the primary financial score is exact.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocrbench import metrics, schema, truth  # noqa: E402

TOL_DIAGNOSTIC = 0.005  # secondary only; never replaces exact scoring


def score_result(result: dict, all_truth: dict) -> dict:
    """Score one experiment result against derived benchmark truth."""
    scored = {
        "model": result.get("model"),
        "model_version": result.get("model_version"),
        "device": result.get("device"),
        "document_id": result.get("document_id"),
        "latency_ms": result.get("latency_ms"),
        "schema_errors": schema.validate_result(result),
    }
    doc_id = result.get("document_id")
    doc_truth = all_truth.get(doc_id)
    if doc_truth is None:
        scored["truth_error"] = f"unknown document_id: {doc_id!r}"
        return scored
    fields = result.get("fields") or {}
    text = result.get("text") or ""
    visual = result.get("visual_description") or ""
    tasks = doc_truth.get("tasks") or []

    if doc_truth.get("identifiers"):
        scored["identifiers"] = metrics.score_identifiers(
            fields, doc_truth["identifiers"])
    exact = metrics.score_numerics(
        fields, truth.scorable_numerics(doc_truth))
    scored["numerics_exact"] = exact
    tol = metrics.score_numerics(
        fields, truth.scorable_numerics(doc_truth), rel_tol=TOL_DIAGNOSTIC)
    scored["numerics_tol_diagnostic"] = {
        "rel_tol": TOL_DIAGNOSTIC,
        "accuracy": tol["accuracy"],
        "note": "secondary diagnostic only; exact is the primary score",
    }
    scored["anchors"] = metrics.score_anchors(
        text, doc_truth.get("text_anchors") or [])
    anchor_ref = " ".join(doc_truth.get("text_anchors") or [])
    scored["cer_vs_anchors_diagnostic"] = metrics.cer(anchor_ref, text)
    scored["wer_vs_anchors_diagnostic"] = metrics.wer(anchor_ref, text)
    if "table" in tasks:
        try:
            expected_pairs = truth.expected_table_pairs(doc_truth)
        except ValueError as exc:
            scored["table"] = {"status": "skipped", "reason": str(exc)}
        else:
            pred_pairs = metrics.pairs_from_table_grid(
                result.get("tables") or [],
                [label for label, _ in expected_pairs])
            scored["table"] = metrics.score_table(pred_pairs,
                                                  expected_pairs)
    if "chart" in tasks and doc_truth.get("chart"):
        scored["chart"] = metrics.score_chart_understanding(
            fields, visual, doc_truth["chart"])
    if "kpi" in tasks:
        scored["kpi"] = metrics.score_kpi_understanding(
            fields, visual,
            {"numeric_values": doc_truth.get("numeric_values") or {},
             "budget_verdict": doc_truth.get("budget_verdict") or {}})
    return scored


def summarize(scored_results: list) -> dict:
    """Per-metric aggregates over scored documents (no combined score)."""

    def mean(key):
        nums = []
        for s in scored_results:
            v = s.get(key)
            if isinstance(v, dict) and "accuracy" in v:
                nums.append(v["accuracy"])
        return (sum(nums) / len(nums)) if nums else None

    def recall_mean():
        vals = [s["anchors"]["recall"] for s in scored_results
                if isinstance(s.get("anchors"), dict)]
        return (sum(vals) / len(vals)) if vals else None

    assoc = [s["table"]["association_accuracy"] for s in scored_results
             if isinstance(s.get("table"), dict)
             and "association_accuracy" in s["table"]]
    lat = [s["latency_ms"] for s in scored_results
           if isinstance(s.get("latency_ms"), (int, float))]
    return {
        "n_documents": len(scored_results),
        "mean_identifier_accuracy": mean("identifiers"),
        "mean_numeric_exact_accuracy": mean("numerics_exact"),
        "mean_anchor_recall": recall_mean(),
        "mean_table_association_accuracy":
            (sum(assoc) / len(assoc)) if assoc else None,
        "latency_ms": lat,
        "n_schema_violations": sum(len(s.get("schema_errors") or [])
                                   for s in scored_results),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", help="kaggle_results.json path")
    parser.add_argument("--truth", default=str(
        ROOT / "data" / "benchmark" / "dataset_v0.1"
        / "ocr_benchmark_truth.json"))
    parser.add_argument("--out", default=None,
                        help="write full scored JSON here (default: stdout)")
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 on any schema violation")
    args = parser.parse_args()

    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    if isinstance(results, dict):
        results = [results]
    all_truth = json.loads(Path(args.truth).read_text(encoding="utf-8"))
    scored = [score_result(r, all_truth) for r in results]
    report = {
        "scored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_results": len(scored),
        "documents": scored,
        "summary": summarize(scored),
    }
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).write_text(payload, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(payload)
    print(f"{'document':10} {'ident':>6} {'num_exact':>9} "
          f"{'anchor':>6} {'assoc':>6} {'ms':>9}")
    for s in scored:
        def fmt(d, k):
            v = (d or {}).get(k)
            return f"{v:.2f}" if isinstance(v, float) else "   -  "

        print(f"{str(s.get('document_id')):10} "
              f"{fmt(s.get('identifiers'), 'accuracy'):>6} "
              f"{fmt(s.get('numerics_exact'), 'accuracy'):>9} "
              f"{fmt(s.get('anchors'), 'recall'):>6} "
              f"{fmt(s.get('table'), 'association_accuracy'):>6} "
              f"{s.get('latency_ms') if s.get('latency_ms') is not None else '-':>9}")
    violations = sum(len(s.get("schema_errors") or []) for s in scored)
    if violations:
        print(f"WARNING: {violations} schema violation(s) — see JSON report")
        return 1 if args.strict else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
