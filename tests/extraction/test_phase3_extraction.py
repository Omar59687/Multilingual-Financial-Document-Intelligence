"""Phase 3 extraction validation tests (I5, deterministic).

Plain-dict tests only: they import ``extraction.validation`` and never
import schemas / provenance / store, so they pass before integration.
No network, no duckdb, no randomness, no wall-clock.
"""

import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# Package import (tests/extraction has no __init__.py by design so that
# `import extraction` resolves to src/extraction, never shadowing).
import extraction.validation as V


def _base_record(**overrides):
    rec = {
        "record_id": "R-001",
        "metric": "revenue",
        "value": "125000.00",
        "revenue": "125000.00",
        "period": "2021-03-01",
        "branch_id": "BR-RUH",
        "provenance": {
            "document_id": "DEV-004",
            "source_label": "Revenue",
            "display_value": "125,000.00",
            "precision": "exact-visible",
        },
        "claims_exact": True,
    }
    rec.update(overrides)
    return rec


# 1. valid exact-visible record passes -------------------------------------
def test_01_valid_exact_visible_record_passes():
    errors = V.validate_record(_base_record())
    assert errors == []


# 2. negative revenue fails ------------------------------------------------
def test_02_negative_revenue_fails_sign():
    rec = _base_record(value="-10.00", revenue="-10.00")
    rec["provenance"] = dict(rec["provenance"],
                             display_value="-10.00")
    errors = V.validate_record(rec)
    assert any(e.startswith("E_SIGN:") for e in errors), errors


# 3. negative net_income passes --------------------------------------------
def test_03_negative_net_income_passes():
    rec = _base_record(metric="net_income", value="-5000.00",
                       claims_exact=True)
    rec.pop("revenue", None)
    rec["net_income"] = "-5000.00"
    rec["provenance"] = {"document_id": "DEV-009",
                         "source_label": "Net income",
                         "display_value": "-5,000.00",
                         "precision": "exact-visible"}
    errors = V.validate_record(rec)
    assert not any(e.startswith("E_SIGN:") for e in errors), errors
    assert errors == []


# 4. non-2dp fails ----------------------------------------------------------
def test_04_non_2dp_fails_money():
    for bad in ("100.1", "100.123", "100", "100.000"):
        rec = _base_record(value=bad, revenue=bad)
        rec["provenance"] = dict(rec["provenance"], display_value=bad)
        # Display "100.1" is coarse -> would also trip precision when
        # claiming exact; drop the exact claim to isolate the E_MONEY
        # signal for this money-format test.
        rec["claims_exact"] = False
        rec["provenance"] = dict(rec["provenance"],
                                 precision="display-rounded")
        errors = V.validate_record(rec)
        assert any(e.startswith("E_MONEY:") for e in errors), (bad, errors)


# 5. display-rounded claiming exact fails -----------------------------------
def test_05_display_rounded_claiming_exact_fails():
    rec = _base_record()
    rec["provenance"] = {"document_id": "DEV-009",
                         "source_label": "Revenue",
                         "display_value": "23.90M SAR",
                         "precision": "exact-visible"}
    rec["claims_exact"] = True
    errors = V.validate_record(rec)
    assert any("exact_claim_on_rounded" in e for e in errors), errors
    assert any(e.startswith("E_PRECISION:") for e in errors)


# 6. Dammam 2018-12 rejected -------------------------------------------------
def test_06_dammam_pre_2019_rejected():
    rec = _base_record(branch_id="BR-DMM", period="2018-12-01")
    errors = V.validate_record(rec)
    assert any(e.startswith("E_BRANCH:") and "dammam_pre_opening" in e
               for e in errors), errors


# 7. bad period rejected ------------------------------------------------------
def test_07_bad_period_rejected():
    rec = _base_record(period="not-a-period")
    errors = V.validate_record(rec)
    assert any(e.startswith("E_PERIOD:") for e in errors), errors
    rec2 = _base_record(period="2021-13-01")
    assert any(e.startswith("E_PERIOD:")
               for e in V.validate_record(rec2))


# 8. missing provenance rejected ------------------------------------------------
def test_08_missing_provenance_rejected():
    rec = _base_record()
    rec.pop("provenance", None)
    rec.pop("claims_exact", None)
    errors = V.validate_record(rec)
    assert any(e.startswith("E_PROVENANCE:") for e in errors), errors


# 9. R1/R2/R3 helper valid -------------------------------------------------------
def test_09_check_r1_r3_valid():
    errors = V.check_r1_r3(revenue="1000.00", cogs="400.00",
                           gross_profit="600.00",
                           operating_expenses="100.00",
                           operating_profit="500.00",
                           other_income="50.00", finance_costs="20.00",
                           tax_expense="30.00", net_income="500.00")
    assert errors == []


# 10. R1/R2/R3 helpers flag violations ---------------------------------------------
def test_10_check_r1_r3_invalid():
    assert any("R1" in e for e in V.check_r1_r3(
        revenue="1000.00", cogs="400.00", gross_profit="999.99",
        operating_expenses="100.00", operating_profit="899.99",
        other_income="0.00", finance_costs="0.00", tax_expense="0.00",
        net_income="899.99"))
    assert any("R2" in e for e in V.check_r1_r3(
        revenue="1000.00", cogs="400.00", gross_profit="600.00",
        operating_expenses="100.00", operating_profit="1.00",
        other_income="0.00", finance_costs="0.00", tax_expense="0.00",
        net_income="1.00"))
    assert any("R3" in e for e in V.check_r1_r3(
        revenue="1000.00", cogs="400.00", gross_profit="600.00",
        operating_expenses="100.00", operating_profit="500.00",
        other_income="50.00", finance_costs="20.00", tax_expense="30.00",
        net_income="1.00"))


# 11. invoice total -------------------------------------------------------------------
def test_11_check_invoice_total():
    assert V.check_invoice_total(subtotal="100.00", vat="15.00",
                                 total="115.00") == []
    bad = V.check_invoice_total(subtotal="100.00", vat="15.00",
                                total="999.99")
    assert any(e.startswith("E_RECONCILIATION:") for e in bad), bad


# 12. budget variance + zero-budget None -------------------------------------------------
def test_12_budget_variance_and_zero_budget():
    out = V.budget_variance("120.00", "100.00")
    assert out["variance"] == Decimal("20.00")
    assert out["variance_pct"] == Decimal("20.00")
    assert out["budget_zero"] is False
    zero = V.budget_variance("120.00", "0.00")
    assert zero["variance"] == Decimal("120.00")
    assert zero["variance_pct"] is None
    assert zero["budget_zero"] is True
    # negative variance is representable (may be negative)
    neg = V.budget_variance("80.00", "100.00")
    assert neg["variance"] == Decimal("-20.00")


# 13. determinism ----------------------------------------------------------------------------
def test_13_determinism_same_input_same_order():
    rec = _base_record(period="bogus", branch_id="NOPE")
    rec.pop("provenance", None)
    first = V.validate_record(rec)
    second = V.validate_record(rec)
    assert first == second
    assert len(first) >= 2
    # fixed stage order: provenance before branch before period
    kinds = [e.split(":")[0] for e in first]
    assert kinds.index("E_PROVENANCE") < kinds.index("E_BRANCH")
    assert kinds.index("E_BRANCH") < kinds.index("E_PERIOD")
    batch1 = V.validate_batch([_base_record(), rec])
    batch2 = V.validate_batch([_base_record(), rec])
    assert batch1 == batch2
    assert list(batch1.keys()) == list(batch2.keys())


# 14. unknown labels passthrough vs strict ----------------------------------------------------
def test_14_unknown_label_passthrough_and_strict():
    rec = _base_record(metric="mystery_metric")
    assert V.validate_record(rec, strict=False) == []
    strict_errors = V.validate_record(rec, strict=True)
    assert any(e.startswith("E_LABEL:") for e in strict_errors), \
        strict_errors
    assert V.warnings_for_record(rec) and any(
        w.startswith("W_LABEL:") for w in V.warnings_for_record(rec))


# 15. unparseable display claiming exact fails --------------------------------------------------
def test_15_unparseable_display_claiming_exact_fails():
    rec = _base_record()
    rec["provenance"] = {"document_id": "DEV-009",
                         "source_label": "Revenue",
                         "display_value": "n/a",
                         "precision": "exact-visible"}
    rec["claims_exact"] = True
    errors = V.validate_record(rec)
    assert any(e.startswith("E_PRECISION:") for e in errors), errors
    assert any("unparseable" in e for e in errors), errors


# 16. batch keys + is_valid helpers ------------------------------------------------------------------
def test_16_batch_keys_and_is_valid_helpers():
    good = _base_record(record_id="KEEP")
    bad = _base_record()
    bad.pop("provenance", None)
    bad.pop("claims_exact", None)
    out = V.validate_batch([good, bad, {"record_id": "KEEP"}])
    assert out["KEEP"] == []
    assert any(e.startswith("E_PROVENANCE:") for e in out["R-001"]), out
    assert "KEEP#2" in out  # colliding record_id disambiguated
    assert V.is_valid_money("10.00") is True
    assert V.is_valid_money("10.1") is False
    assert V.is_valid_period("2021-03-01") is True
    assert V.is_valid_period("bogus") is False
    assert V.is_valid_branch("BR-RUH") is True
    assert V.is_valid_branch("XX") is False
    assert V.is_valid_record(good) is True
    assert V.is_valid_record(bad) is False


# 17. no live-DB / stdlib-only --------------------------------------------------------------------------
def test_17_no_live_db_required():
    # Validation must work with plain dicts whether or not duckdb exists.
    assert V.validate_record(_base_record()) == []
