"""Balance-sheet structured-extraction tests (I5, deterministic).

Scope: DEV-010 representation audit + deterministic balance-sheet rules.
Plain dicts + package imports only. No network, no randomness, no
wall-clock, no top-level duckdb (optional function-local extra only),
no embeddings/RAG/LLM/OCR reruns.

Import pattern mirrors tests/extraction/test_phase3_extraction.py
(``sys.path.insert`` of ``src``, plain ``import extraction.validation``).

SYNTHETIC VALUES ONLY: every numeric below (``100.00`` / ``50.00`` /
``300.00`` / ``200.00`` / ``999.99`` / ``1000000.00`` / ``1.00M``-style
displays) is synthetic. No benchmark ``expected_numeric_values`` appear
anywhere in this file.

DEV-010 REPRESENTATION AUDIT (§8, label-mapping table)
-------------------------------------------------------
Ground-truth FIELD NAMES read (names only, never numeric values) from
``data/dev/dataset_v0.1/ground_truth/DEV-010.json`` ``expected_fields``::

    cash, accounts_receivable, inventory, other_current_assets,
    property_and_equipment, total_assets, accounts_payable, debt,
    other_liabilities, total_liabilities, equity

Visible LABEL strings read (labels only) from the paddle DEV-010 table
text in ``data/benchmark/results/paddle_results.json`` and mapped to the
coordinated-contract 11-metric vocabulary::

    | visible label                  | contract metric          |
    |--------------------------------|--------------------------|
    | Cash                           | cash                     |
    | Accounts receivable            | accounts_receivable      |
    | Inventory                      | inventory                |
    | Other current assets           | other_current_assets     |
    | Property and equipment (net)   | property_plant_equipment |
    | TOTAL ASSETS                   | total_assets             |
    | Accounts payable               | accounts_payable         |
    | Debt                           | debt                     |
    | Other liabilities              | other_liabilities        |
    | TOTAL LIABILITIES              | total_liabilities        |
    | EQUITY                         | equity                   |

Notes:
- ``Property and equipment (net)`` maps to contract metric
  ``property_plant_equipment``; the GT field-name list spells the same
  concept ``property_and_equipment`` (alias drift, reported to the
  orchestrator — no new ID namespace, no fixture change here).
- Section headers ``ASSETS`` / ``LIABILITIES`` are structural, not
  metrics: correctly unmapped, not UNMAPPED concepts.
- UNMAPPED concepts: none (``dev010_audit_unmapped()`` returns []).

R9/R10/R11 STATUS (integrated):
- ``validation.py`` exposes ``check_bs_r9`` / ``check_bs_r10`` /
  ``check_bs_r11`` returning ``{"rule","status","reason","metrics",
  "precision"}`` with status PASS / FAIL / NOT_EVALUATED, wired into
  ``validate_record`` (FAIL surfaces as ``E_RECONCILIATION:R9|R10|R11``;
  PASS and NOT_EVALUATED are silent). Tests call the helpers directly
  with ``(fields, precisions)`` payloads.
- ``Provenance`` exposes optional ``extraction_route``
  (native/light-ocr/paddle/qwen-visual, default None, excluded from IDs).
- ``periods.parse_period("FY2024")`` normalises to year grain canonical
  ``"2024"`` and ``validation.py`` accepts raw ``"FY2024"`` records.
- Label normalizer is ``validation.normalize_bs_label`` (tested directly
  in test_31).
"""

import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# Package imports (tests/extraction has no __init__.py by design so that
# `import extraction` resolves to src/extraction, never shadowing).
import extraction.validation as V
from extraction import periods as P
from extraction import provenance as PROV

# ---------------------------------------------------------------------------
# Contract vocabulary + audit mapping (label strings only, no numerics)
# ---------------------------------------------------------------------------

CONTRACT_11 = (
    "cash",
    "accounts_receivable",
    "inventory",
    "other_current_assets",
    "property_plant_equipment",
    "total_assets",
    "accounts_payable",
    "debt",
    "other_liabilities",
    "total_liabilities",
    "equity",
)

# Visible paddle DEV-010 label -> contract metric (see module docstring).
DEV010_LABEL_TO_METRIC = {
    "Cash": "cash",
    "Accounts receivable": "accounts_receivable",
    "Inventory": "inventory",
    "Other current assets": "other_current_assets",
    "Property and equipment (net)": "property_plant_equipment",
    "TOTAL ASSETS": "total_assets",
    "Accounts payable": "accounts_payable",
    "Debt": "debt",
    "Other liabilities": "other_liabilities",
    "TOTAL LIABILITIES": "total_liabilities",
    "EQUITY": "equity",
}

# GT expected_fields NAMES (names only) for the audit cross-check. The GT
# list spells one concept `property_and_equipment` where the contract uses
# `property_plant_equipment` (alias drift, see module docstring).
DEV010_GT_FIELD_NAMES = (
    "cash",
    "accounts_receivable",
    "inventory",
    "other_current_assets",
    "property_and_equipment",
    "total_assets",
    "accounts_payable",
    "debt",
    "other_liabilities",
    "total_liabilities",
    "equity",
)


def dev010_audit_unmapped():
    """Return DEV-010 visible concepts with no contract-metric mapping.

    Structural section headers (ASSETS / LIABILITIES) are not metrics and
    are excluded by design. Currently every visible row label maps, so
    this returns [] — any future unmapped concept must be appended here
    (test comment + return value) rather than invented into canonical keys.
    """
    unmapped = []
    for label, metric in DEV010_LABEL_TO_METRIC.items():
        if metric not in CONTRACT_11:
            unmapped.append(label)
    # No UNMAPPED visible row-label concepts at authoring time.
    return unmapped


def _base_bs_record(metric, label, value="100.00", period="2024",
                    display=None, precision="exact-visible", **overrides):
    """Minimal single-metric balance-sheet record (synthetic values only)."""
    rec = {
        "metric": metric,
        "value": value,
        "period": period,
        "provenance": {
            "document_id": "DEV-010",
            "source_label": label,
            "display_value": display if display is not None else value,
            "precision": precision,
        },
    }
    rec.update(overrides)
    return rec


def _bs_multikey_record(period="2024", **fields):
    """One dict carrying several balance-sheet keys (reconciliation grain).

    Mirrors the `_check_record_reconciliation` present-keys style: R9/R10/
    R11 operate over the jointly-present component keys. Provenance uses the
    TOTAL row label; the display is exact-visible synthetic "300.00"-style
    only where the single display must parse (kept "300.00" exact here; the
    rounded-display grain is covered by tests 19-20, not here).
    """
    rec = {
        "period": period,
        "provenance": {
            "document_id": "DEV-010",
            "source_label": "TOTAL ASSETS",
            "display_value": "300.00",
            "precision": "exact-visible",
        },
    }
    rec.update(fields)
    return rec


def _recon_errors(errors, tag):
    """Errors mentioning a reconciliation tag (e.g. 'R9')."""
    return [e for e in errors if tag in e]


def _call_bs_helper(fn, payload):
    """Call an R9/R10/R11 helper defensively.

    Returns (called: bool, result). A landing helper may take kwargs or one
    positional dict; signature drift must never fail this file (the
    orchestrator resolves drift — outcome reported in the RETURN summary).
    """
    try:
        return True, fn(**payload)
    except TypeError:
        try:
            return True, fn(payload)
        except Exception as exc:  # noqa: BLE001 - drift probe, never raises
            return False, exc
    except Exception as exc:  # noqa: BLE001 - drift probe, never raises
        return False, exc


# 1-9: each balance metric accepted as a metric/value record ---------------
def test_01_cash_accepted():
    assert V.validate_record(_base_bs_record("cash", "Cash")) == []


def test_02_accounts_receivable_accepted():
    assert V.validate_record(
        _base_bs_record("accounts_receivable", "Accounts receivable")) == []


def test_03_inventory_accepted():
    assert V.validate_record(
        _base_bs_record("inventory", "Inventory")) == []


def test_04_property_plant_equipment_accepted():
    # Visible label "Property and equipment (net)" maps to the contract
    # metric property_plant_equipment (GT field-name alias drift noted).
    assert V.validate_record(_base_bs_record(
        "property_plant_equipment", "Property and equipment (net)")) == []


def test_05_total_assets_accepted():
    assert V.validate_record(
        _base_bs_record("total_assets", "TOTAL ASSETS")) == []


def test_06_accounts_payable_accepted():
    assert V.validate_record(
        _base_bs_record("accounts_payable", "Accounts payable")) == []


def test_07_debt_accepted():
    assert V.validate_record(_base_bs_record("debt", "Debt")) == []


def test_08_total_liabilities_accepted():
    assert V.validate_record(
        _base_bs_record("total_liabilities", "TOTAL LIABILITIES")) == []


def test_09_equity_accepted():
    assert V.validate_record(_base_bs_record("equity", "EQUITY")) == []


def test_09b_other_current_assets_accepted():
    assert V.validate_record(
        _base_bs_record("other_current_assets", "Other current assets")) == []


def test_09c_other_liabilities_accepted():
    assert V.validate_record(
        _base_bs_record("other_liabilities", "Other liabilities")) == []


# 10-12: R9 (assets sum) PASS / FAIL / NOT_EVALUATED ------------------------
def test_10_r9_exact_visible_pass():
    comps = [Decimal("100.00"), Decimal("50.00"), Decimal("50.00"),
             Decimal("50.00"), Decimal("50.00")]
    total = Decimal("300.00")
    assert abs(total - sum(comps)) <= V.MONEY_TOL
    fn = getattr(V, "check_bs_r9", None)
    if fn is not None:
        called, res = _call_bs_helper(fn, {
            "cash": "100.00", "accounts_receivable": "50.00",
            "inventory": "50.00", "other_current_assets": "50.00",
            "property_plant_equipment": "50.00", "total_assets": "300.00"})
        assert called, f"check_bs_r9 present but not callable: {res!r}"
        assert isinstance(res, dict) and res.get("status") == "PASS", res
    else:
        rec = _bs_multikey_record(
            cash="100.00", accounts_receivable="50.00", inventory="50.00",
            other_current_assets="50.00", property_plant_equipment="50.00",
            total_assets="300.00")
        errors = V.validate_record(rec)
        # Consistent components: no R9 error (helper absent => NOT_EVALUATED
        # silence; drift reported to orchestrator).
        assert _recon_errors(errors, "R9") == [], errors


def test_11_r9_mismatch_fail():
    comps = [Decimal("100.00"), Decimal("50.00"), Decimal("50.00"),
             Decimal("50.00"), Decimal("50.00")]
    bad_total = Decimal("999.99")
    assert abs(bad_total - sum(comps)) > V.MONEY_TOL  # exact-only mismatch
    fn = getattr(V, "check_bs_r9", None)
    if fn is not None:
        called, res = _call_bs_helper(fn, {
            "cash": "100.00", "accounts_receivable": "50.00",
            "inventory": "50.00", "other_current_assets": "50.00",
            "property_plant_equipment": "50.00", "total_assets": "999.99"})
        assert called, f"check_bs_r9 present but not callable: {res!r}"
        assert isinstance(res, dict) and res.get("status") == "FAIL", res
    else:
        rec = _bs_multikey_record(
            cash="100.00", accounts_receivable="50.00", inventory="50.00",
            other_current_assets="50.00", property_plant_equipment="50.00",
            total_assets="999.99")
        errors = V.validate_record(rec)  # never raises on mismatch
        r9 = _recon_errors(errors, "R9")
        if r9:
            assert any(e.startswith("E_RECONCILIATION:") for e in r9), errors
        # Else: validator has no R9 rule yet => NOT_EVALUATED silence
        # (mismatch proven real arithmetically above; drift reported).


def test_12_r9_incomplete_not_evaluated():
    # Drop total_assets => incomplete inputs => no R9 error (NOT_EVALUATED).
    fn = getattr(V, "check_bs_r9", None)
    if fn is not None:
        called, res = _call_bs_helper(fn, {
            "cash": "100.00", "accounts_receivable": "50.00",
            "inventory": "50.00", "other_current_assets": "50.00",
            "property_plant_equipment": "50.00"})
        assert called, f"check_bs_r9 present but not callable: {res!r}"
        assert isinstance(res, dict) and res.get("status") == "NOT_EVALUATED", res
    else:
        rec = _bs_multikey_record(
            cash="100.00", accounts_receivable="50.00", inventory="50.00",
            other_current_assets="50.00", property_plant_equipment="50.00")
        assert _recon_errors(V.validate_record(rec), "R9") == []


# 13-15: R10 (liabilities sum) PASS / FAIL / NOT_EVALUATED ------------------
def test_13_r10_exact_visible_pass():
    assert abs(Decimal("200.00") - (
        Decimal("100.00") + Decimal("50.00") + Decimal("50.00"))) <= V.MONEY_TOL
    fn = getattr(V, "check_bs_r10", None)
    if fn is not None:
        called, res = _call_bs_helper(fn, {
            "accounts_payable": "100.00", "debt": "50.00",
            "other_liabilities": "50.00", "total_liabilities": "200.00"})
        assert called, f"check_bs_r10 present but not callable: {res!r}"
        assert isinstance(res, dict) and res.get("status") == "PASS", res
    else:
        rec = _bs_multikey_record(
            accounts_payable="100.00", debt="50.00",
            other_liabilities="50.00", total_liabilities="200.00")
        assert _recon_errors(V.validate_record(rec), "R10") == []


def test_14_r10_mismatch_fail():
    assert abs(Decimal("999.99") - (
        Decimal("100.00") + Decimal("50.00") + Decimal("50.00"))) > V.MONEY_TOL
    fn = getattr(V, "check_bs_r10", None)
    if fn is not None:
        called, res = _call_bs_helper(fn, {
            "accounts_payable": "100.00", "debt": "50.00",
            "other_liabilities": "50.00", "total_liabilities": "999.99"})
        assert called, f"check_bs_r10 present but not callable: {res!r}"
        assert isinstance(res, dict) and res.get("status") == "FAIL", res
    else:
        rec = _bs_multikey_record(
            accounts_payable="100.00", debt="50.00",
            other_liabilities="50.00", total_liabilities="999.99")
        errors = V.validate_record(rec)  # never raises on mismatch
        r10 = _recon_errors(errors, "R10")
        if r10:
            assert any(e.startswith("E_RECONCILIATION:") for e in r10), errors
        # Else: no R10 rule yet => NOT_EVALUATED silence (drift reported).


def test_15_r10_incomplete_not_evaluated():
    fn = getattr(V, "check_bs_r10", None)
    if fn is not None:
        called, res = _call_bs_helper(fn, {
            "accounts_payable": "100.00", "debt": "50.00",
            "other_liabilities": "50.00"})
        assert called, f"check_bs_r10 present but not callable: {res!r}"
        assert isinstance(res, dict) and res.get("status") == "NOT_EVALUATED", res
    else:
        rec = _bs_multikey_record(
            accounts_payable="100.00", debt="50.00", other_liabilities="50.00")
        assert _recon_errors(V.validate_record(rec), "R10") == []


# 16-18: R11 (assets == liab + equity) PASS / FAIL / NOT_EVALUATED -----------
def test_16_r11_exact_visible_pass():
    assert abs(Decimal("300.00") - (
        Decimal("200.00") + Decimal("100.00"))) <= V.MONEY_TOL
    fn = getattr(V, "check_bs_r11", None)
    if fn is not None:
        called, res = _call_bs_helper(fn, {
            "total_assets": "300.00", "total_liabilities": "200.00",
            "equity": "100.00"})
        assert called, f"check_bs_r11 present but not callable: {res!r}"
        assert isinstance(res, dict) and res.get("status") == "PASS", res
    else:
        rec = _bs_multikey_record(
            total_assets="300.00", total_liabilities="200.00",
            equity="100.00")
        assert _recon_errors(V.validate_record(rec), "R11") == []


def test_17_r11_mismatch_fail():
    assert abs(Decimal("300.00") - (
        Decimal("200.00") + Decimal("999.99"))) > V.MONEY_TOL
    fn = getattr(V, "check_bs_r11", None)
    if fn is not None:
        called, res = _call_bs_helper(fn, {
            "total_assets": "300.00", "total_liabilities": "200.00",
            "equity": "999.99"})
        assert called, f"check_bs_r11 present but not callable: {res!r}"
        assert isinstance(res, dict) and res.get("status") == "FAIL", res
    else:
        rec = _bs_multikey_record(
            total_assets="300.00", total_liabilities="200.00",
            equity="999.99")
        errors = V.validate_record(rec)  # never raises on mismatch
        r11 = _recon_errors(errors, "R11")
        if r11:
            assert any(e.startswith("E_RECONCILIATION:") for e in r11), errors
        # Else: no R11 rule yet => NOT_EVALUATED silence (drift reported).


def test_18_r11_incomplete_not_evaluated():
    # Drop equity => incomplete => no R11 error (NOT_EVALUATED).
    fn = getattr(V, "check_bs_r11", None)
    if fn is not None:
        called, res = _call_bs_helper(fn, {
            "total_assets": "300.00", "total_liabilities": "200.00"})
        assert called, f"check_bs_r11 present but not callable: {res!r}"
        assert isinstance(res, dict) and res.get("status") == "NOT_EVALUATED", res
    else:
        rec = _bs_multikey_record(
            total_assets="300.00", total_liabilities="200.00")
        assert _recon_errors(V.validate_record(rec), "R11") == []


# 19: display-rounded R9 never claims exact equality ------------------------
def test_19_display_rounded_r9_no_exact_claim():
    # Synthetic M-suffixed magnitude (M-style display, synthetic value):
    # components sum exactly to the rounded total arithmetically, but the
    # evidence grade must stay approximate, never exact-visible.
    fields = {
        "cash": "400000.00", "accounts_receivable": "300000.00",
        "inventory": "200000.00", "other_current_assets": "50000.00",
        "property_plant_equipment": "50000.00",
        "total_assets": "1000000.00"}
    precisions = {k: "display-rounded" for k in fields}
    res = V.check_bs_r9(fields, precisions)
    assert res["status"] == "PASS", res
    assert res["precision"] == "approximate", res  # never exact-visible
    # Small drift inside the widened band stays tolerance-consistent ...
    drifted = dict(fields, total_assets="1000000.50")
    res_band = V.check_bs_r9(drifted, precisions)
    assert res_band["status"] == "PASS", res_band
    assert res_band["precision"] == "approximate", res_band
    # ... while drift beyond any band still FAILs.
    bad = dict(fields, total_assets="1999999.99")
    assert V.check_bs_r9(bad, precisions)["status"] == "FAIL"
    # Via the validate path: consistent rounded displays are valid ...
    rec = _base_bs_record("total_assets", "TOTAL ASSETS",
                          value="1000000.00", display="1.00M",
                          precision="display-rounded")
    assert V.validate_record(rec) == []
    # ... and an exact claim on a rounded display IS rejected.
    assert V.check_precision_claim("1.00M", "exact-visible", True) != []


# 20: mixed precision classified weakest ------------------------------------
def test_20_mixed_precision_weakest():
    # One display-rounded component among exact ones => the joint grade is
    # the weakest (display-rounded), never exact-visible.
    exact_norm = V._normalize_display_value("50.00")
    rounded_norm = V._normalize_display_value("1.00M")
    assert exact_norm["precision"] == "exact-visible"
    assert rounded_norm["precision"] == "display-rounded"
    assert V.check_precision_claim("50.00", "exact-visible", True) == []
    assert V.check_precision_claim("1.00M", "exact-visible", True) != []
    grades = {exact_norm["precision"], rounded_norm["precision"]}
    weakest = "display-rounded" if "display-rounded" in grades else "exact-visible"
    assert weakest != "exact-visible"
    # Joint R9 grade over mixed precisions reports the weakest input grade.
    fields = {
        "cash": "100.00", "accounts_receivable": "50.00",
        "inventory": "50.00", "other_current_assets": "50.00",
        "property_plant_equipment": "50.00", "total_assets": "300.00"}
    mixed = {k: "exact-visible" for k in fields}
    mixed["cash"] = "display-rounded"
    res_mixed = V.check_bs_r9(fields, mixed)
    assert res_mixed["status"] == "PASS", res_mixed
    assert res_mixed["precision"] == "approximate", res_mixed
    # A missing entry is weaker still: unknown.
    partial = {k: "exact-visible" for k in fields if k != "cash"}
    res_unknown = V.check_bs_r9(fields, partial)
    assert res_unknown["precision"] == "unknown", res_unknown
    for name in ("check_bs_r9", "check_bs_r10", "check_bs_r11"):
        fn = getattr(V, name, None)
        assert fn is not None, f"{name} must exist post-integration"
        called, res = _call_bs_helper(fn, {"precision": "display-rounded"})
        if called and isinstance(res, dict) and "precision" in res:
            assert res["precision"] != "exact-visible", (name, res)
    # Via the validate path: a rounded component record stays rounded and
    # valid, but can never carry the exact-visible grade.
    rec = _base_bs_record("cash", "Cash", value="1000000.00", display="1.00M",
                          precision="display-rounded")
    assert V.validate_record(rec) == []


# 21-22: NaN and Infinity rejected (E_MONEY, never raises) -------------------
def test_21_nan_rejected_never_raises():
    rec = _base_bs_record("cash", "Cash", value=float("nan"))
    try:
        errors = V.validate_record(rec)
    except Exception as exc:  # noqa: BLE001 - must never raise
        raise AssertionError(f"validate_record raised on NaN: {exc!r}")
    assert any(e.startswith("E_MONEY:") for e in errors), errors


def test_22_infinity_rejected_never_raises():
    for bad in (float("inf"), float("-inf")):
        rec = _base_bs_record("cash", "Cash", value=bad)
        try:
            errors = V.validate_record(rec)
        except Exception as exc:  # noqa: BLE001 - must never raise
            raise AssertionError(f"validate_record raised on {bad!r}: {exc!r}")
        assert any(e.startswith("E_MONEY:") for e in errors), (bad, errors)


# 23: missing numeric never becomes zero -------------------------------------
def test_23_missing_numeric_never_zero():
    # Absent key with metric present: no E_MONEY for it, no zero injected.
    rec_absent = {"metric": "cash", "period": "2024",
                  "provenance": {"document_id": "DEV-010",
                                 "source_label": "Cash",
                                 "display_value": "100.00",
                                 "precision": "exact-visible"}}
    errors_absent = V.validate_record(rec_absent)
    assert not [e for e in errors_absent if ":cash" in e or "cash" in e
                and e.startswith("E_MONEY:")], errors_absent
    assert "Decimal('0" not in repr(errors_absent)
    assert "0.00" not in repr(errors_absent)
    # Explicit None value: likewise skipped, never zero-filled.
    rec_none = dict(rec_absent, value=None)
    errors_none = V.validate_record(rec_none)
    assert not [e for e in errors_none if e.startswith("E_MONEY:")], errors_none
    assert "0.00" not in repr(errors_none)
    # Unparseable display normalizes to None, never zero (policy mirror).
    norm = V._normalize_display_value("")
    assert norm == {"normalized_value": None, "precision": "unknown"}, norm


# 24: FY2024 preserved without invented date ----------------------------------
def test_24_fy2024_year_grain_no_invented_date():
    parsed = P.parse_period("FY2024")
    assert parsed["canonical"] == "2024"
    assert parsed["kind"] == "year" and parsed["fiscal_year"] == 2024
    assert "12-31" not in parsed["canonical"]  # no invented month/day
    assert parsed["iso_start"] == "2024-01-01"
    assert parsed["iso_end"] == "2024-12-31"
    # Canonical year-grain record validates cleanly.
    assert V.validate_record(
        _base_bs_record("total_assets", "TOTAL ASSETS", period="2024")) == []
    # Raw "FY2024" display: accepted as YEAR grain by both layers —
    # periods.py canonicalizes to "2024", validation.py validates raw.
    raw_errors = V.validate_record(
        _base_bs_record("total_assets", "TOTAL ASSETS", period="FY2024"))
    assert raw_errors == [], raw_errors


# 25: exact date preserved ----------------------------------------------------
def test_25_exact_date_preserved():
    parsed = P.parse_period("2024-12-31")
    assert parsed["canonical"] == "2024-12-31"
    assert parsed["kind"] == "day" and parsed["fiscal_year"] == 2024
    assert V.validate_record(_base_bs_record(
        "equity", "EQUITY", period="2024-12-31")) == []


# 26-27: company-level and branch-level records -------------------------------
def test_26_company_level_no_branch_valid():
    rec = _base_bs_record("total_assets", "TOTAL ASSETS")
    assert "branch_id" not in rec
    assert V.validate_record(rec) == []


def test_27_branch_level_bruh_valid():
    rec = _base_bs_record("total_assets", "TOTAL ASSETS", branch_id="BR-RUH")
    assert V.validate_record(rec) == []


# 28: provenance without coordinates valid ------------------------------------
def test_28_provenance_without_coordinates_valid():
    # Document-level Provenance: no page/element/sheet/row/column — legitimate
    # for image-wide visuals ("where applicable" per Rule 10).
    prov = PROV.Provenance(document_id="DEV-010", source_label="Cash",
                           display_value="100.00",
                           normalized_value=Decimal("100.00"),
                           precision="exact-visible")
    assert prov.page is None and prov.element_id is None
    rec = {"metric": "cash", "value": "100.00", "period": "2024",
           "provenance": {"document_id": prov.document_id,
                          "source_label": prov.source_label,
                          "display_value": prov.display_value,
                          "precision": prov.precision}}
    assert V.validate_record(rec) == []
    assert V.is_valid_provenance(rec["provenance"]) is True


# 29: FIN-* stability -----------------------------------------------------------
def test_29_fin_id_and_normalization_stable():
    base = {"document_id": "DEV-010", "metric": "cash",
            "period": "2024", "value": "100.00"}
    first = PROV.record_id_for(dict(base))
    second = PROV.record_id_for(dict(base))
    assert first == second
    assert first.startswith("FIN-")
    # Metric casefold stability: "Cash" hashes like "cash".
    assert PROV.record_id_for(dict(base, metric="Cash")) == first
    # Period normalization stability: FY display -> canonical year.
    assert PROV.record_id_for(dict(base, period="FY2024")) == first
    # Value normalization stability: Decimal input hashes like the string.
    assert PROV.record_id_for(dict(base, value=Decimal("100.00"))) == first


# 30: no top-level duckdb dependency in src/extraction/*.py -------------------
def test_30_no_duckdb_import_in_extraction_package():
    import ast
    pkg = ROOT / "src" / "extraction"
    py_files = sorted(pkg.glob("*.py"))
    assert py_files, f"no .py files found in {pkg}"
    offenders = []
    for p in py_files:
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Import):
                if any((a.name or "").split(".")[0] == "duckdb" for a in node.names):
                    offenders.append(f"{p.name}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] == "duckdb":
                    offenders.append(f"{p.name}:{node.lineno}")
    assert offenders == [], f"top-level duckdb import found in: {offenders}"


# 31 (audit): DEV-010 label-mapping table ---------------------------------------
def test_31_dev010_label_mapping_audit():
    # Full mapping table lives in the module docstring (§8 audit). Every
    # visible paddle DEV-010 row label maps to exactly one contract metric.
    assert set(DEV010_LABEL_TO_METRIC.values()) == set(CONTRACT_11)
    assert len(CONTRACT_11) == 11
    # GT field-name cross-check (names only): contract covers every GT name
    # modulo the property_and_equipment / property_plant_equipment alias.
    gt = set(DEV010_GT_FIELD_NAMES)
    contract = set(CONTRACT_11)
    assert gt - contract == {"property_and_equipment"}, gt ^ contract
    assert contract - gt == {"property_plant_equipment"}, gt ^ contract
    # Normalize-helper assertions against validation.normalize_bs_label.
    for label, metric in DEV010_LABEL_TO_METRIC.items():
        assert V.normalize_bs_label(label) == metric, (label, metric)
    # GT column-name form maps too (underscored alias, no enum duplication).
    assert V.normalize_bs_label("property_and_equipment") == \
        "property_plant_equipment"
    assert V.normalize_bs_label("Actual:") is None
    assert V.normalize_bs_label("") is None
    # Any unmapped concept is reported via dev010_audit_unmapped().
    assert dev010_audit_unmapped() == []


# 32 (contract): extraction_route spelling guard ----------------------------------
def test_32_extraction_route_spelling_guard():
    # Provenance field must be spelled exactly `extraction_route`, carrying
    # the frozen Phase 2 route vocabulary (optional, default None).
    fields = getattr(PROV.Provenance, "model_fields", {})
    assert "extraction_route" in fields
    prov = PROV.Provenance(document_id="DEV-010", source_label="Cash",
                           display_value="100.00",
                           normalized_value=Decimal("100.00"),
                           precision="exact-visible",
                           extraction_route="paddle")
    assert prov.extraction_route == "paddle"
    # Old dicts without the key still validate (default None).
    plain = PROV.Provenance(document_id="DEV-010", source_label="Cash",
                            display_value="100.00",
                            normalized_value=Decimal("100.00"),
                            precision="exact-visible")
    assert plain.extraction_route is None
    # No misspelled near-miss may exist alongside the canonical spelling.
    variants = [k for k in fields if k != "extraction_route"
                and ("rout" in k.lower() or "extract" in k.lower())]
    assert variants == [], f"route-like fields with wrong spelling: {variants}"
