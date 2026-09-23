"""Score Kaggle experiment results with the Phase-2 benchmark tooling.

Usage (from repo root):
    python scripts/score_ocr_results.py data/benchmark/results/<run>.json
    python scripts/score_ocr_results.py <run>.json --out <summary>.json

Reads machine-readable results in the standard contract
(src/ocrbench/schema.py) and the derived benchmark truth. Prints
per-document metrics plus careful aggregate summaries — one aggregate per
metric, NEVER a single combined "overall AI score". The ±0.5% numeric
figure is a secondary diagnostic; the primary financial score is exact.

Two independent layers (never merged or averaged):

  LAYER A — RECOGNITION ("did the model see it in raw text?"):
    identifiers_text (identifier_text_accuracy),
    numerics_text_exact (numeric_text_exact_accuracy),
    table_label_text (table_label_text_recall).
    Raw OCR engines with empty fields/tables CAN score here.

  LAYER B — STRUCTURING ("did the model attach it to the right
  field/row?"):
    identifiers_structured (identifier_structured_accuracy),
    numerics_structured_exact (numeric_structured_exact_accuracy),
    table (table_association_accuracy).
    Requires populated fields/tables; a correct number in prose alone
    earns ZERO structured credit.

Legacy aliases identifiers (= structured) and numerics_exact (=
structured exact) are preserved so historical values never disappear.
CER/WER vs anchor concatenations are diagnostics only — not full-document
OCR accuracy, since no complete reference transcription exists.
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
        structured_ids = metrics.score_identifiers(
            fields, doc_truth["identifiers"])
        scored["identifiers"] = structured_ids  # legacy alias (structured)
        scored["identifiers_structured"] = structured_ids
        scored["identifiers_text"] = metrics.score_identifier_text(
            text, doc_truth["identifiers"])
    exact = metrics.score_numerics(
        fields, truth.scorable_numerics(doc_truth))
    scored["numerics_exact"] = exact  # legacy alias (structured exact)
    scored["numerics_structured_exact"] = exact
    scored["numerics_text_exact"] = metrics.score_numeric_text(
        text, truth.scorable_numerics(doc_truth))
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
    scored["cer_wer_note"] = (
        "diagnostic only: CER/WER vs anchor concatenation, NOT "
        "full-document OCR accuracy (no complete reference "
        "transcription exists)")
    if "table" in tasks:
        try:
            expected_pairs = truth.expected_table_pairs(doc_truth)
        except ValueError as exc:
            scored["table"] = {"status": "skipped", "reason": str(exc)}
            scored["table_label_text"] = {"status": "skipped",
                                          "reason": str(exc)}
        else:
            pred_pairs = metrics.pairs_from_table_grid(
                result.get("tables") or [],
                [label for label, _ in expected_pairs])
            scored["table"] = metrics.score_table(pred_pairs,
                                                  expected_pairs)
            scored["table_label_text"] = metrics.score_table_label_text(
                text, [label for label, _ in expected_pairs])
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
    """Per-metric aggregates over scored documents (no combined score).

    Recognition and structuring are aggregated INDEPENDENTLY — never
    averaged together. Legacy mean_* aliases are preserved alongside the
    explicit Layer-A / Layer-B names.
    """

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

    def label_text_mean():
        vals = [s["table_label_text"]["label_text_recall"]
                for s in scored_results
                if isinstance(s.get("table_label_text"), dict)
                and "label_text_recall" in s["table_label_text"]]
        return (sum(vals) / len(vals)) if vals else None

    assoc = [s["table"]["association_accuracy"] for s in scored_results
             if isinstance(s.get("table"), dict)
             and "association_accuracy" in s["table"]]
    lat = [s["latency_ms"] for s in scored_results
           if isinstance(s.get("latency_ms"), (int, float))]
    mean_assoc = (sum(assoc) / len(assoc)) if assoc else None
    return {
        "n_documents": len(scored_results),
        # --- legacy aliases (kept; historical values never removed) ---
        "mean_identifier_accuracy": mean("identifiers_structured"),
        "mean_numeric_exact_accuracy": mean("numerics_structured_exact"),
        "mean_anchor_recall": recall_mean(),
        "mean_table_association_accuracy": mean_assoc,
        # --- explicit Layer A (recognition) ---
        "mean_identifier_text_accuracy": mean("identifiers_text"),
        "mean_numeric_text_exact_accuracy": mean("numerics_text_exact"),
        "mean_table_label_text_recall": label_text_mean(),
        # --- explicit Layer B (structuring) ---
        "mean_identifier_structured_accuracy":
            mean("identifiers_structured"),
        "mean_numeric_structured_exact_accuracy":
            mean("numerics_structured_exact"),
        "mean_table_association_accuracy_explicit": mean_assoc,
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
    print(f"{'document':10} {'model':>12} {'id_text':>8} {'id_struct':>9} "
          f"{'num_text':>9} {'num_struct':>10} "
          f"{'anchor':>6} {'label_txt':>9} {'assoc':>6} {'ms':>9}")
    for s in scored:
        def fmt(d, k):
            v = (d or {}).get(k)
            return f"{v:.2f}" if isinstance(v, float) else "   -    "

        print(f"{str(s.get('document_id')):10} "
              f"{str(s.get('model') or '-')[:12]:>12} "
              f"{fmt(s.get('identifiers_text'), 'accuracy'):>8} "
              f"{fmt(s.get('identifiers_structured'), 'accuracy'):>9} "
              f"{fmt(s.get('numerics_text_exact'), 'accuracy'):>9} "
              f"{fmt(s.get('numerics_structured_exact'), 'accuracy'):>10} "
              f"{fmt(s.get('anchors'), 'recall'):>6} "
              f"{fmt(s.get('table_label_text'), 'label_text_recall'):>9} "
              f"{fmt(s.get('table'), 'association_accuracy'):>6} "
              f"{s.get('latency_ms') if s.get('latency_ms') is not None else '-':>9}")
    violations = sum(len(s.get("schema_errors") or []) for s in scored)
    if violations:
        print(f"WARNING: {violations} schema violation(s) — see JSON report")
        return 1 if args.strict else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
