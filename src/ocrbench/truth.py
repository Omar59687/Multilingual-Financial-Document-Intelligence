"""Derived Phase-2 benchmark truth (evaluation-side ONLY).

Builds the small scoring representation from frozen dataset_v0.1
ground-truth JSONs — programmatically, never by manual duplication:

{ document_id, language, role, tasks,
  identifiers, text_anchors, numeric_values, fields,
  tables, chart, visual_elements, question_targets, source_location,
  table_sections, verified_pairs, budget_verdict, unrendered_context,
  labels_verified }

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

# DEV-010 rendered row labels, VERIFIED against the actual frozen visual
# (data/benchmark/dataset_v0.1/inputs/DEV-010_p1.png, rendered from the
# frozen DEV-010 PDF): (visible label, GT field key) in visual order.
# The "Total liabilities + Equity = ..." line is a printed check equation,
# not a scored row. Equation holds: 29,014,808.77 + 25,945,183.96 =
# 54,959,992.73. Row counts (5/3/1) match GT expected_tables.
DEV010_VERIFIED_LABELS = [
    ("Cash", "cash"),
    ("Accounts receivable", "accounts_receivable"),
    ("Inventory", "inventory"),
    ("Other current assets", "other_current_assets"),
    ("Property and equipment (net)", "property_and_equipment"),
    ("TOTAL ASSETS", "total_assets"),
    ("Accounts payable", "accounts_payable"),
    ("Debt", "debt"),
    ("Other liabilities", "other_liabilities"),
    ("TOTAL LIABILITIES", "total_liabilities"),
    ("EQUITY", "equity"),
]


def dev010_expected_pairs(numerics: dict) -> list:
    """(label, value) associations for DEV-010 table scoring, in visual
    order. Values come from frozen GT (the render was verified to show
    exactly these); labels are the verified visible strings above."""
    return [(label, numerics[field])
            for label, field in DEV010_VERIFIED_LABELS]


DEV010_SECTIONS = [
    ("Assets", ["cash", "accounts_receivable", "inventory",
                "other_current_assets", "property_and_equipment"],
     "total_assets"),
    ("Liabilities", ["accounts_payable", "debt", "other_liabilities"],
     "total_liabilities"),
    ("Equity", [], "equity"),
]

# DEV-009 rendered dashboard labels, VERIFIED against the frozen generator
# (src/document_generation/documents.py build_dev009: 4 KPI cards, 3 branch
# bars, budget-vs-actual panel) and DEV_DOCUMENT_SPEC §DEV-009 resolved
# KPIs. (visible label, GT numeric field key) in visual order. The budget
# verdict line is a visible status statement, not a numeric row, so its
# field key is None and it is scored via the visible-quote rule
# (metrics.score_budget_status), never via recomputation.
DEV009_VERIFIED_LABELS = [
    ("Revenue", "revenue"),
    ("Gross profit", "gross_profit"),
    ("Operating expenses", "operating_expenses"),
    ("Net income", "net_income"),
    ("Riyadh", "branch_revenue_BR-RUH"),
    ("Jeddah", "branch_revenue_BR-JED"),
    ("Dammam", "branch_revenue_BR-DMM"),
    ("Budget", "budget_total"),
    ("Actual", "revenue"),
    ("Variance", "variance"),
]

# Visible budget-status statement for DEV-009 (variance +52,904.62 >= 0,
# revenue-type target -> met). Quoted verbatim on the panel alongside the
# exact variance figure.
DEV009_STATUS_TEXT = "Budget met"


def load_gt(gt_path) -> dict:
    """Read one frozen ground-truth JSON (explicit path, no scanning)."""
    return json.loads(Path(gt_path).read_text(encoding="utf-8"))


def derive_truth(gt: dict, role: str, tasks: list) -> dict:
    """Project one frozen GT record onto the scoring representation."""
    dev_id = gt["dev_id"]
    verified_pairs = None
    labels_verified = False
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
        verified_pairs = dev010_expected_pairs(numerics)
        labels_verified = True  # visible strings checked against the
        # frozen render (see DEV010_VERIFIED_LABELS); association scoring
        # is valid for DEV-010 from here on.
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
        "labels_verified": labels_verified,
        "verified_pairs": verified_pairs,
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


def expected_table_pairs(truth: dict) -> list:
    """Verified (label, value) pairs for association scoring.

    Raises unless labels_verified is True — association scoring without
    verified visible labels is invalid (architect decision 4).
    """
    if not truth.get("labels_verified") or not truth.get("verified_pairs"):
        raise ValueError(
            f"table association truth not verified for "
            f"{truth.get('document_id')}")
    return [(label, value) for label, value in truth["verified_pairs"]]
