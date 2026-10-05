"""G1 DEV-set extraction measurement (evaluation harness, thin CLI).

Runs the frozen per-document route table (see DOC_ROUTES below —
harness-side configuration from GT metadata + DEV spec; adapters stay
generic and GT-blind), derives truth-recomputed gold via
evaluation.g1, scores per Evaluation Plan §3, and writes
docs/g1_baseline.json + a console summary.

Gold provenance: canonical CSVs re-read here (origin-1); no GT values
copied. Thresholds are NOT decided here (TBD → human gate).

Usage (repo root):  python scripts/measure_g1.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation import g1 as G
from extraction.runner_native import run_native
from extraction.runner_paddle import run_paddle
from extraction.runner_qwen import run_qwen
from extraction.runner_statements import run_statements
from ingestion.service import ingest_document

DEV_DOCS = ROOT / "data" / "dev" / "dataset_v0.1" / "documents"
DEV_GT = ROOT / "data" / "dev" / "dataset_v0.1" / "ground_truth"
CANON = ROOT / "data" / "canonical" / "dataset_v0.1"
PADDLE_RESULTS = ROOT / "data" / "benchmark" / "results" / "paddle_results.json"
QWEN_DIR = ROOT / "data" / "benchmark" / "results"
OCR_NOISE_DOCS = {"DEV-004", "DEV-010"}  # restricted ±0.5% band (EVAL §3)


def _paddle_result(doc_id):
    with open(PADDLE_RESULTS, encoding="utf-8") as f:
        for entry in json.load(f):
            if entry.get("document_id") == doc_id:
                return entry
    raise KeyError(f"no paddle result for {doc_id}")


def _qwen_result(doc_id):
    for sub in sorted(QWEN_DIR.iterdir()):
        candidate = sub / "qwen_visual_results.json"
        if not candidate.is_file():
            continue
        with open(candidate, encoding="utf-8") as f:
            for entry in json.load(f):
                if entry.get("document_id") == doc_id:
                    return entry
    raise KeyError(f"no qwen result for {doc_id}")


def _gt(doc_id):
    with open(DEV_GT / f"{doc_id}.json", encoding="utf-8") as f:
        return json.load(f)


def extract_for_doc(doc_id):
    """Run the frozen route table for one DEV document (harness config)."""
    if doc_id == "DEV-001":
        doc = ingest_document(DEV_DOCS / "DEV-001_income_statement_2023_EN.pdf").document
        return run_statements(doc, column_periods={1: "2023", 2: "2022"})
    if doc_id == "DEV-002":
        # Arabic statement labels: no frozen mapping (deferred) — fail closed.
        doc = ingest_document(DEV_DOCS / "DEV-002_branch_expenses_2022_AR.pdf").document
        return run_statements(doc, column_periods={})
    if doc_id == "DEV-003":
        doc = ingest_document(DEV_DOCS / "DEV-003_supplier_invoice_2023_MIX.pdf").document
        return run_statements(doc, column_periods={1: "2023-06-07"}, branch_id="BR-RUH")
    if doc_id == "DEV-004":
        return run_paddle(_paddle_result("DEV-004"), document_period="2019-06-09",
                          branch_id="BR-JED", filename="DEV-004_scanned_receipt_2019_AR.png",
                          file_type="png", language="ar")
    if doc_id == "DEV-005":
        doc = ingest_document(DEV_DOCS / "DEV-005_management_commentary_2023_MIX.docx").document
        return run_statements(doc, column_periods={})
    if doc_id == "DEV-006":
        doc = ingest_document(DEV_DOCS / "DEV-006_monthly_transactions_2023_EN.xlsx").document
        return run_native(doc)
    if doc_id == "DEV-007":
        doc = ingest_document(DEV_DOCS / "DEV-007_transactions_extract_2024_MIX.csv").document
        return run_native(doc)
    if doc_id == "DEV-008":
        return run_qwen(_qwen_result("DEV-008"), document_period="2024")
    if doc_id == "DEV-009":
        return run_qwen(_qwen_result("DEV-009"), document_period="2023-Q4")
    if doc_id == "DEV-010":
        return run_paddle(_paddle_result("DEV-010"), document_period="2020", page=1,
                          filename="DEV-010_balance_sheet_2020_EN_scanned.pdf",
                          file_type="pdf", language="en")
    raise ValueError(f"unknown DEV document: {doc_id}")


DOCS = [f"DEV-{i:03d}" for i in range(1, 11)]
MISS_CAP = 50


def main():
    canon = G.load_canonical(CANON)
    per_doc = {}
    oos_registry = []
    for doc_id in DOCS:
        gt = _gt(doc_id)
        expected, oos = G.expected_for_doc(doc_id, gt.get("source_record_ids") or [], canon)
        env = extract_for_doc(doc_id)
        scored = G.score_doc(expected, list(env.records), ocr_noise=(doc_id in OCR_NOISE_DOCS))
        scored["n_misses_total"] = len(scored["misses"])
        scored["misses"] = scored["misses"][:MISS_CAP]
        per_doc[doc_id] = {
            "language": gt.get("language"), "expected": len(expected),
            "extracted": len(env.records), "ok": env.ok,
            "warnings": len(env.warnings), **{k: v for k, v in scored.items()},
        }
        oos_registry.extend([{"doc": d, "class": c, "detail": t} for d, c, t in oos])
    by_lang: dict[str, list[str]] = {}
    for doc_id in DOCS:
        by_lang.setdefault(per_doc[doc_id]["language"], []).append(doc_id)
    slices = {lang: G.aggregate({d: per_doc[d] for d in docs}) for lang, docs in by_lang.items()}
    overall = G.aggregate({d: per_doc[d] for d in DOCS})
    in_scope_docs = [d for d in DOCS if per_doc[d]["expected"] > 0]
    core = G.aggregate({d: per_doc[d] for d in in_scope_docs})
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(),
               "per_doc": per_doc, "slices": slices, "overall": overall,
               "in_scope_core": core, "out_of_scope": oos_registry}
    out_path = ROOT / "docs" / "g1_baseline.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"{'doc':8} {'lang':6} {'exp':>6} {'ext':>6} {'TP':>6} {'P':>6} {'R':>6} {'F1':>6}")
    for doc_id in DOCS:
        d = per_doc[doc_id]
        print(f"{doc_id:8} {d['language']:6} {d['expected']:6d} {d['extracted']:6d} "
              f"{d['tp']:6d} {d['precision']:6.3f} {d['recall']:6.3f} {d['f1']:6.3f}")
    print(f"OVERALL P={overall['precision']:.3f} R={overall['recall']:.3f} F1={overall['f1']:.3f}")
    print(f"IN-SCOPE-CORE P={core['precision']:.3f} R={core['recall']:.3f} F1={core['f1']:.3f}")
    print(f"out-of-scope entries: {len(oos_registry)}; JSON: {out_path}")


if __name__ == "__main__":
    main()
