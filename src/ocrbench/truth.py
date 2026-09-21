"""Derived Phase-2 benchmark truth (evaluation-side ONLY).

Builds the small scoring representation from frozen dataset_v0.1
ground-truth JSONs — programmatically, never by manual duplication:

{ document_id, language, role, tasks,
  identifiers, text_anchors, numeric_values, fields,
  tables, chart, visual_elements, question_targets, source_location,
  table_sections, budget_verdict, unrendered_context, labels_verified }

Production code (src/ingestion and any future OCR service) must NEVER
import this module — guarded by test. GT paths are explicit parameters;
nothing is read at import time.
"""

import json
from pathlib import Path

# Benchmark roles/tasks per frozen DEV document. Primary = visual path is
# the only path; secondary = native text exists, vision output is scored
# against the same truth as a reference comparison.
BENCHMARK_DOCS = {
    "DEV-004": ("primary", ["ocr"]),
    "DEV-008": ("primary", ["ocr", "chart"]),
    "DEV-009": ("primary", ["ocr", "kpi"]),
    "DEV-010": ("primary", ["ocr", "table"]),
    "DEV-002": ("secondary", ["ocr", "table"]),
    "DEV-003": ("secondary", ["ocr", "table"]),
}

# GT numeric keys that exist for comparative questions but are NOT rendered
# in the fixture — OCR must never be penalized for their absence.
# (DEV_DOCUMENT_SPEC §DEV-010: 2019 equity is truth context, not rendered.)
UNRENDERED_CONTEXT = {
    "DEV-010": ["equity_2019_context"],
}

# DEV-010 scanned sections: GT carries titles + row counts and flat numeric
# keys, not rendered row-label strings. Grouping below follows the balance-
# sheet section order in the frozen fixture; rendered label strings still
# need origin-3 human verification (labels_verified=False) before
# (label, value) association scoring runs — values score immediately.
DEV010_SECTIONS = [
    ("Assets", ["cash", "accounts_receivable", "inventory",
                "other_current_assets", "property_and_equipment"],
     "total_assets"),
    ("Liabilities", ["accounts_payable", "debt", "other_liabilities"],
     "total_liabilities"),
    ("Equity", [], "equity"),
]


def load_gt(gt_path) -> dict:
    """Read one frozen ground-truth JSON (explicit path, no scanning)."""
    return json.loads(Path(gt_path).read_text(encoding="utf-8"))


def derive_truth(gt: dict, role: str, tasks: list) -> dict:
    """Project one frozen GT record onto the scoring representation."""
    dev_id = gt["dev_id"]
    sectioned = None
    if dev_id == "DEV-010":
        numerics = gt.get("expected_numeric_values") or {}
        sectioned = [
            {"title": title,
             "field_keys": [f for f in keys if f in numerics],
             "total_field": total if total in numerics else None,
             "expected_rows": next(
                 (t.get("rows") for t in (gt.get("expected_tables") or [])
                  if t.get("title") == title), None)}
            for title, keys, total in DEV010_SECTIONS
        ]
    budget_verdict = None
    if dev_id == "DEV-009":
        numerics = gt.get("expected_numeric_values") or {}
        budget_verdict = {
            # Semantic rule frozen in DEV_DOCUMENT_SPEC §DEV-009:
            # revenue actual >= budget -> met.
            "verdict_text": "Budget met",
            "variance": numerics.get("variance"),
            "actual_field": "revenue",
            "budget_field": "budget_total",
        }
    return {
        "document_id": dev_id,
        "language": gt.get("language"),
        "role": role,
        "tasks": list(tasks),
        "identifiers": gt.get("expected_identifiers") or {},
        "text_anchors": list(gt.get("expected_text") or []),
        "numeric_values": dict(gt.get("expected_numeric_values") or {}),
        "fields": list(gt.get("expected_fields") or []),
        "tables": gt.get("expected_tables"),
        "table_sections": sectioned,
        "labels_verified": False,
        "chart": gt.get("expected_chart_data"),
        "visual_elements": gt.get("expected_visual_elements"),
        "question_targets": list(gt.get("expected_question_targets") or []),
        "source_location": list(gt.get("expected_page_locations") or []),
        "budget_verdict": budget_verdict,
        "unrendered_context": list(UNRENDERED_CONTEXT.get(dev_id, [])),
    }


def derive_all(gt_dir, docs: dict = None) -> dict:
    """Derive benchmark truth for the benchmark set from a GT directory."""
    gt_dir = Path(gt_dir)
    docs = docs or BENCHMARK_DOCS
    out = {}
    for dev_id, (role, tasks) in docs.items():
        out[dev_id] = derive_truth(load_gt(gt_dir / f"{dev_id}.json"),
                                   role, tasks)
    return out


def scorable_numerics(truth: dict) -> dict:
    """Numeric values actually rendered in the fixture (excludes context)."""
    skip = set(truth.get("unrendered_context") or [])
    return {k: v for k, v in (truth.get("numeric_values") or {}).items()
            if k not in skip}
