"""Offline semantic-normalization tests (no Kaggle, no GPU).

Covers the canonical DEV-009 mapping layer: explicit visible pairs ->
canonical keys with rounding provenance. Never reads ground truth to
populate fields; never alters visual association scoring.
"""

import json
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ocrbench import canonical, metrics  # noqa: E402

RUN_FIELDS = {
    "Revenue": "23.90M SAR",
    "Gross profit": "8.48M SAR",
    "Operating expenses": "5.35M SAR",
    "Net income": "2.80M SAR",
    "Riyadh": "13.83M",
    "Jeddah": "6.86M",
    "Dammam": "3.21M",
    "Budget:": "23.85M SAR",
    "Actual:": "23.90M SAR",
    "Variance:": "+52,904.62 SAR",
    "(+0.22%) Budget met": "(+0.22%) Budget met",
}


# 1. alias mapping -------------------------------------------------------------------
def test_1_label_alias_mapping():
    assert canonical.alias_key("Revenue") == "revenue"
    assert canonical.alias_key("Gross profit") == "gross_profit"
    assert canonical.alias_key("Operating expenses") == "operating_expenses"
    assert canonical.alias_key("Net income") == "net_income"
    assert canonical.alias_key("Budget") == "budget_total"
    assert canonical.alias_key("Budget:") == "budget_total"  # colon artifact
    assert canonical.alias_key("Variance:") == "variance"
    assert canonical.alias_key("  REVENUE  ") == "revenue"  # case/space


# 2. branch mapping ---------------------------------------------------------------------
def test_2_branch_label_mapping():
    assert canonical.alias_key("Riyadh") == "branch_revenue_BR-RUH"
    assert canonical.alias_key("Jeddah") == "branch_revenue_BR-JED"
    assert canonical.alias_key("Dammam") == "branch_revenue_BR-DMM"


# 3. suffixes --------------------------------------------------------------------------------
def test_3_magnitude_suffix_normalization():
    assert canonical.normalize_display_value("23.90M SAR")[
        "normalized_value"] == Decimal("23900000.00")
    assert canonical.normalize_display_value("1.5B")[
        "normalized_value"] == Decimal("1500000000.00")
    assert canonical.normalize_display_value("250K")[
        "normalized_value"] == Decimal("250000.00")


# 4. currency -----------------------------------------------------------------------------------
def test_4_currency_stripping():
    assert canonical.normalize_display_value("23.90M SAR")[
        "normalized_value"] == Decimal("23900000.00")
    assert canonical.normalize_display_value("$1,234.56")[
        "normalized_value"] == Decimal("1234.56")
    assert canonical.normalize_display_value("8.48M")[
        "normalized_value"] == Decimal("8480000.00")


# 5. signed variance --------------------------------------------------------------------------------
def test_5_signed_variance():
    out = canonical.normalize_display_value("+52,904.62 SAR")
    assert out["normalized_value"] == Decimal("52904.62")
    assert out["precision"] == "exact-visible"
    out = canonical.normalize_display_value("-1.5M")
    assert out["normalized_value"] == Decimal("-1500000.00")
    assert out["precision"] == "display-rounded"


# 6. provenance ------------------------------------------------------------------------------------------
def test_6_display_rounded_provenance():
    assert canonical.normalize_display_value("23.90M SAR")[
        "precision"] == "display-rounded"
    assert canonical.normalize_display_value("+52,904.62 SAR")[
        "precision"] == "exact-visible"


# 7. no hidden precision --------------------------------------------------------------------------------------
def test_7_no_hidden_precision_fabrication():
    norm = canonical.normalize_display_value("23.90M SAR")[
        "normalized_value"]
    assert norm != Decimal("23902434.34")  # full-precision truth differs
    bad = canonical.normalize_display_value("not a number")
    assert bad["normalized_value"] is None  # never silent zero
    assert canonical.normalize_display_value("")[
        "normalized_value"] is None


# 8. unknown labels ----------------------------------------------------------------------------------------------------------------
def test_8_unknown_labels_stay_unmapped():
    assert canonical.alias_key("Actual:") is None
    assert canonical.alias_key("(+0.22%) Budget met") is None
    assert canonical.alias_key("XYZ") is None
    assert canonical.alias_key("") is None
    out = canonical.canonicalize_fields(RUN_FIELDS)
    assert "Actual:" in out["unmapped"]
    assert "(+0.22%) Budget met" in out["unmapped"]
    assert len(out["canonical_fields"]) == 9


# 9. provenance --------------------------------------------------------------------------------------------------------------------------
def test_9_explicit_provenance_preserved():
    out = canonical.canonicalize_fields(RUN_FIELDS)
    revenue = out["canonical_fields"]["revenue"]
    assert revenue["display_value"] == "23.90M SAR"
    assert revenue["normalized_value"] == "23900000.00"
    assert revenue["source_label"] == "Revenue"
    assert revenue["precision"] == "display-rounded"
    variance = out["canonical_fields"]["variance"]
    assert variance["source_label"] == "Variance:"
    assert variance["precision"] == "exact-visible"


# 10. no truth dependency ----------------------------------------------------------------------------------------------------------------------
def test_10_no_ground_truth_dependency():
    src = (ROOT / "src" / "ocrbench" / "canonical.py").read_text(
        encoding="utf-8")
    assert "import truth" not in src and "from . import truth" not in src \
        and "from ocrbench import truth" not in src
    assert "23902434" not in src  # no truth values embedded
    out = canonical.canonicalize_fields({"Revenue": "23.90M SAR"})
    assert out["canonical_fields"]["revenue"]["normalized_value"] == \
        "23900000.00"


# 11. association unaffected --------------------------------------------------------------------------------------------------------------------------
def test_11_old_visual_association_metrics_unaffected():
    before = json.dumps(RUN_FIELDS, sort_keys=True)
    out = canonical.canonicalize_fields(RUN_FIELDS)
    assert json.dumps(RUN_FIELDS, sort_keys=True) == before  # no mutation
    assert out["canonical_fields"]["revenue"]["display_value"] == \
        RUN_FIELDS["Revenue"]  # display layer intact for scorer input
    pred = metrics.pairs_from_fields_map(RUN_FIELDS)
    assert ("Revenue", "23.90M SAR") in pred  # scorer input unchanged
