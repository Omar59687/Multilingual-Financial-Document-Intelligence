"""Evaluation scoring tests (bounded slice, deterministic).

Synthetic gold/extracted sets only: no corpus, no GT, no network.
Covers score_doc matching classes, F1 math, ocr-noise band, aggregate,
norm_2dp, and calc.run_case cell comparison.
"""

import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from evaluation import calc as C
from evaluation import g1 as G


def _rec(doc, metric, period, branch, value):
    return SimpleNamespace(document_id=doc, metric=metric, period=period,
                           branch_id=branch, value=Decimal(value))


def test_01_exact_match_tp_and_f1():
    exp = {("D", "revenue", "2023", None, "10.00"),
           ("D", "cogs", "2023", None, "6.00")}
    out = G.score_doc(exp, [_rec("D", "revenue", "2023", None, "10.00"),
                            _rec("D", "cogs", "2023", None, "6.00")])
    assert (out["tp"], out["fp"], out["fn"]) == (2, 0, 0)
    assert (out["precision"], out["recall"], out["f1"]) == (1.0, 1.0, 1.0)
    assert out["numeric_accuracy"] == 1.0


def test_02_mismatch_invented_missed_duplicate():
    exp = {("D", "revenue", "2023", None, "10.00"),
           ("D", "cogs", "2023", None, "6.00")}
    out = G.score_doc(exp, [_rec("D", "revenue", "2023", None, "11.00"),
                            _rec("D", "vat", "2023", None, "1.00"),
                            _rec("D", "vat", "2023", None, "1.00")])
    assert out["tp"] == 0
    kinds = sorted(m["kind"] for m in out["misses"])
    assert kinds == ["duplicate", "invented", "missed", "missed", "numeric-mismatch"]
    assert (out["fp"], out["fn"]) == (3, 2)


def test_03_branch_distinguishes_and_ocr_band():
    exp = {("D", "revenue", "2023", None, "10.00")}
    out = G.score_doc(exp, [_rec("D", "revenue", "2023", "BR-JED", "10.00")])
    assert out["tp"] == 0 and out["fp"] == 1 and out["fn"] == 1
    noisy = G.score_doc({("D", "revenue", "2023", None, "100.00")},
                        [_rec("D", "revenue", "2023", None, "100.40")],
                        ocr_noise=True)
    assert noisy["tp"] == 1 and noisy["band_matches"] == 1
    strict = G.score_doc({("D", "revenue", "2023", None, "100.00")},
                         [_rec("D", "revenue", "2023", None, "100.40")])
    assert strict["tp"] == 0
    # Adversarial multiplicity: two near-values, one expected occurrence —
    # exactly one band match; no double rescue, no negative counts.
    multi = G.score_doc({("D", "revenue", "2023", None, "100.00")},
                        [_rec("D", "revenue", "2023", None, "100.40"),
                         _rec("D", "revenue", "2023", None, "100.30")],
                        ocr_noise=True)
    assert (multi["tp"], multi["fp"], multi["fn"]) == (1, 1, 0)
    assert multi["fn"] >= 0
    # Duplicated expectation matched twice inside the band.
    dbl = G.score_doc([("D", "revenue", "2023", None, "100.00"),
                       ("D", "revenue", "2023", None, "100.00")],
                      [_rec("D", "revenue", "2023", None, "100.40"),
                       _rec("D", "revenue", "2023", None, "100.30")],
                      ocr_noise=True)
    assert (dbl["tp"], dbl["fp"], dbl["fn"]) == (2, 0, 0)
    # Crossing neighborhoods: scorer TP must equal the brute-force
    # maximum-cardinality optimum computed independently here.
    import itertools as _it
    exp_vals = ["100.00", "100.80", "101.60"]
    ext_vals = ["100.40", "100.75", "101.00"]
    exp = [("D", "revenue", "2023", None, v) for v in exp_vals]
    ext = [_rec("D", "revenue", "2023", None, v) for v in ext_vals]
    got = G.score_doc(exp, ext, ocr_noise=True)
    best = 0
    for perm in _it.permutations(range(len(ext_vals))):
        used, n = set(), 0
        for i, v in enumerate(exp_vals):
            j = perm[i] if i < len(ext_vals) else None
            if j is None or j in used:
                continue
            rel = abs(Decimal(ext_vals[j]) - Decimal(v)) / abs(Decimal(v))
            if rel <= Decimal("0.005"):
                used.add(j)
                n += 1
        best = max(best, n)
    assert got["tp"] == best and got["fn"] >= 0
    # True greedy-stranding regression: global-nearest-first would match
    # X->B (0.00397) before Y->B (0.00446), stranding Y and A (TP 1);
    # maximum matching yields X->A + Y->B (TP 2).
    exp2 = [("D", "revenue", "2023", None, "100.00"),
            ("D", "revenue", "2023", None, "100.90")]
    ext2 = [_rec("D", "revenue", "2023", None, "100.50"),
            _rec("D", "revenue", "2023", None, "101.35")]
    strand = G.score_doc(exp2, ext2, ocr_noise=True)
    assert (strand["tp"], strand["fp"], strand["fn"]) == (2, 0, 0)


def test_04_aggregate_and_norm():
    per = {"a": {"tp": 2, "fp": 0, "fn": 1}, "b": {"tp": 2, "fp": 2, "fn": 0}}
    agg = G.aggregate(per)
    assert (agg["tp"], agg["fp"], agg["fn"]) == (4, 2, 1)
    assert abs(agg["f1"] - 2 * (4 / 6) * (4 / 5) / ((4 / 6) + (4 / 5))) < 1e-9
    assert G.norm_2dp("1,000.5") == "1000.50"
    try:
        G.norm_2dp("nope")
        raise AssertionError("must raise")
    except ValueError:
        pass


def _fake_con(rows):
    return SimpleNamespace(
        execute=lambda sql: SimpleNamespace(fetchall=lambda: rows))


def test_05_calc_run_case_cells():
    good = C.run_case(_fake_con([[Decimal("10.00")]]), "x", "SELECT 1", [[Decimal("10.00")]])
    assert good["status"] == "PASS"
    bad = C.run_case(_fake_con([[Decimal("10.00")]]), "x", "SELECT 1", [[Decimal("11.00")]])
    assert bad["status"] == "FAIL"
    floats = C.run_case(_fake_con([[25.0]]), "x", "SELECT 1", [[25.0000001]])
    assert floats["status"] == "PASS"
    nulls = C.run_case(_fake_con([[None]]), "x", "SELECT 1", [[None]])
    assert nulls["status"] == "PASS"
    null_mismatch = C.run_case(_fake_con([[None]]), "x", "SELECT 1", [[Decimal("1.00")]])
    assert null_mismatch["status"] == "FAIL"


def test_06_loaders_on_synthetic_canonical(tmp_path):
    monthly = tmp_path / "monthly_financials.csv"
    monthly.write_text(
        "period,branch_id,revenue,cogs,gross_profit,operating_expenses,"
        "operating_profit,other_income,finance_costs,tax_expense,net_income\n"
        "2023-01-01,BR-RUH,10.00,6.00,4.00,2.00,2.00,0.00,0.00,0.00,2.00\n",
        encoding="utf-8")
    (tmp_path / "invoices.csv").write_text(
        "invoice_id,invoice_date,vendor_customer,branch_id,subtotal,vat,total,"
        "status,related_transaction_id\n", encoding="utf-8")
    (tmp_path / "transactions.csv").write_text(
        "transaction_id,date,branch_id,department_id,transaction_type,category,"
        "vendor_customer,description_en,description_ar,amount,tax_amount,"
        "total_amount,payment_method,reference_number\n", encoding="utf-8")
    (tmp_path / "annual_balance_sheet.csv").write_text(
        "year,cash,accounts_receivable,inventory,other_current_assets,"
        "property_and_equipment,total_assets,accounts_payable,debt,"
        "other_liabilities,total_liabilities,equity\n", encoding="utf-8")
    (tmp_path / "budgets.csv").write_text(
        "period,branch_id,department_id,metric,budget_amount\n", encoding="utf-8")
    recs = C.mint_monthly_records(monthly)
    assert len(recs) == 9
    assert recs[0].provenance.document_id == C.EVAL_DOC
    assert C.month_sums(monthly, "revenue", year=2023) == Decimal("10.00")
    assert G.load_canonical(tmp_path)["monthly"][("2023-01-01", "BR-RUH")]["revenue"] == "10.00"
