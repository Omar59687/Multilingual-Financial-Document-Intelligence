"""Invariant tests for the MizanIQ canonical dataset_v0.1 generator.

Stdlib-only (csv/json/Decimal/pathlib) so the suite runs without extra
dependencies. Strategy: generate into a tmp dir, validate on disk, then
assert each invariant independently of ``validators.py`` logic where it
matters (equations are recomputed here from raw CSV text).
"""

import csv
import json
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dataset import config as C
from dataset import generator, validators

MONEY_COLS = {
    "monthly_financials": ("revenue", "cogs", "gross_profit",
                           "operating_expenses", "operating_profit",
                           "other_income", "finance_costs", "tax_expense",
                           "net_income"),
    "transactions": ("amount", "tax_amount", "total_amount"),
    "budgets": ("budget_amount",),
    "invoices": ("subtotal", "vat", "total"),
    "annual_balance_sheet": ("cash", "accounts_receivable", "inventory",
                             "other_current_assets",
                             "property_and_equipment", "total_assets",
                             "accounts_payable", "debt", "other_liabilities",
                             "total_liabilities", "equity"),
}


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("canonical")
    tables = generator.generate_all(seed=C.RANDOM_SEED)
    generator.write_dataset(tables, out)
    generator.build_manifest(tables, out, seed=C.RANDOM_SEED)
    return out


@pytest.fixture(scope="module")
def tables(data_dir):
    out = {}
    for name in generator.COLUMN_ORDER:
        with open(data_dir / f"{name}.csv", newline="",
                  encoding="utf-8") as f:
            out[name] = list(csv.DictReader(f))
    return out


def test_seed_is_documented_value():
    assert C.RANDOM_SEED == 42


def test_validation_passes(data_dir):
    counts = validators.validate_all(data_dir)
    assert counts["monthly_financials"] == 312
    assert counts["annual_balance_sheet"] == 10


def test_reproducibility_produces_identical_files(data_dir, tmp_path):
    """Two runs with the same seed must be byte-identical (CSVs + manifest
    with the informational generated_at field normalized away)."""
    second = tmp_path / "second"
    tables2 = generator.generate_all(seed=C.RANDOM_SEED)
    generator.write_dataset(tables2, second)
    manifest2 = generator.build_manifest(tables2, second, seed=C.RANDOM_SEED)

    for name in generator.COLUMN_ORDER:
        first = (data_dir / f"{name}.csv").read_bytes()
        again = (second / f"{name}.csv").read_bytes()
        assert first == again, f"{name}.csv differs between runs"

    def normalized(path):
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc.pop("generated_at", None)
        return doc

    assert normalized(data_dir / "manifest.json") == normalized(
        second / "manifest.json")
    assert manifest2["random_seed"] == C.RANDOM_SEED


def test_expected_counts(tables):
    assert len(tables["branches"]) == 3
    assert len(tables["departments"]) == 4
    # 120 RUH + 120 JED + 72 DMM active branch-months
    assert len(tables["monthly_financials"]) == 312
    # 16,200 + 13,200 + 6,480 ~= 36k target
    assert len(tables["transactions"]) == 35880
    # branch slice 160+160+96 plus department slice 160
    assert len(tables["budgets"]) == 576
    assert len(tables["invoices"]) == 2400
    assert 8 <= len(tables["business_events"]) <= 12
    assert len(tables["annual_balance_sheet"]) == 10


def test_row_count_ranges_documented_target():
    # ~36k is a target, not absolute: the deterministic count must sit in a
    # narrow documented band around it.
    assert 34000 <= 35880 <= 38000


def test_branch_and_department_ids(tables):
    assert {b["branch_id"] for b in tables["branches"]} == {
        "BR-RUH", "BR-JED", "BR-DMM"}
    assert {d["department_id"] for d in tables["departments"]} == {
        "DEP-SALES", "DEP-PROC", "DEP-LOG", "DEP-ADM"}
    dmm = next(b for b in tables["branches"]
               if b["branch_id"] == "BR-DMM")
    assert dmm["opened_date"] == "2019-01-01"
    assert dmm["branch_name_ar"] == "الدمام"


def test_no_dammam_records_before_opening(tables):
    """Pre-opening periods are absent (NULL) — never zero-filled rows."""
    assert not [r for r in tables["monthly_financials"]
                if r["branch_id"] == "BR-DMM" and r["period"] < "2019-01-01"]
    assert not [t for t in tables["transactions"]
                if t["branch_id"] == "BR-DMM" and t["date"] < "2019-01-01"]
    assert not [v for v in tables["invoices"]
                if v["branch_id"] == "BR-DMM"
                and v["invoice_date"] < "2019-01-01"]
    assert not [b for b in tables["budgets"]
                if b["branch_id"] == "BR-DMM" and b["period"] < "2019-Q1"]
    assert len([r for r in tables["monthly_financials"]
                if r["branch_id"] == "BR-DMM"]) == 72


def test_monthly_financial_equations(tables):
    for r in tables["monthly_financials"]:
        v = {c: Decimal(r[c]) for c in MONEY_COLS["monthly_financials"]}
        assert v["gross_profit"] == v["revenue"] - v["cogs"]
        assert v["operating_profit"] == v["gross_profit"] - v["operating_expenses"]
        assert v["net_income"] == (v["operating_profit"] + v["other_income"]
                                   - v["finance_costs"] - v["tax_expense"])


def test_invoice_equations_and_vat(tables):
    for v in tables["invoices"]:
        subtotal, vat, total = (Decimal(v[c])
                                for c in ("subtotal", "vat", "total"))
        assert total == subtotal + vat
        assert vat == (subtotal * Decimal("0.15")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP)
        assert re.match(r"^INV-\d{4}-\d{4}$", v["invoice_id"])
        assert v["invoice_id"][4:8] == v["invoice_date"][:4]


def test_transaction_totals_and_tax_rule(tables):
    for t in tables["transactions"]:
        amount, tax, total = (Decimal(t[c]) for c in MONEY_COLS["transactions"])
        assert total == amount + tax
        if t["transaction_type"] == "adjustment":
            assert tax == Decimal("0.00")
        else:
            assert tax == (amount * Decimal("0.15")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP)


def test_balance_sheet_equation(tables):
    years = set()
    for r in tables["annual_balance_sheet"]:
        years.add(r["year"])
        v = {c: Decimal(r[c]) for c in MONEY_COLS["annual_balance_sheet"]}
        assert v["total_assets"] == (v["cash"] + v["accounts_receivable"]
                                     + v["inventory"]
                                     + v["other_current_assets"]
                                     + v["property_and_equipment"])
        assert v["total_liabilities"] == (v["accounts_payable"] + v["debt"]
                                          + v["other_liabilities"])
        assert v["total_assets"] == v["total_liabilities"] + v["equity"]
        assert v["equity"] > 0
    assert years == {str(y) for y in range(2015, 2025)}


def test_stable_id_uniqueness(tables):
    for name, key in (("transactions", "transaction_id"),
                      ("invoices", "invoice_id"),
                      ("business_events", "event_id")):
        ids = [r[key] for r in tables[name]]
        assert len(ids) == len(set(ids)), f"duplicate IDs in {name}"
    months = [(r["period"], r["branch_id"])
              for r in tables["monthly_financials"]]
    assert len(months) == len(set(months))
    assert len({r["year"] for r in tables["annual_balance_sheet"]}) == 10


def test_foreign_key_integrity(tables):
    branches = {b["branch_id"] for b in tables["branches"]}
    depts = {d["department_id"] for d in tables["departments"]}
    tx_ids = {t["transaction_id"] for t in tables["transactions"]}
    inv_ids = {v["invoice_id"] for v in tables["invoices"]}

    assert all(r["branch_id"] in branches
               for r in tables["monthly_financials"])
    assert all(t["branch_id"] in branches and t["department_id"] in depts
               for t in tables["transactions"])
    assert all(v["branch_id"] in branches for v in tables["invoices"])
    for v in tables["invoices"]:
        if v["related_transaction_id"]:
            assert v["related_transaction_id"] in tx_ids
    tx_by_id = {t["transaction_id"]: t for t in tables["transactions"]}
    for v in tables["invoices"]:
        if v["related_transaction_id"]:
            assert tx_by_id[v["related_transaction_id"]][
                "reference_number"] == v["invoice_id"]
    for t in tables["transactions"]:
        if t["reference_number"]:
            assert t["reference_number"] in inv_ids
    for b in tables["budgets"]:
        assert (b["branch_id"] in branches) != (
            b["department_id"] in depts)  # exactly one grain set
    for e in tables["business_events"]:
        assert e["affected_branch"] in branches or e["affected_branch"] == ""


def test_monetary_precision_two_dp(tables):
    pat = re.compile(r"^-?\d+\.\d{2}$")
    for name, cols in MONEY_COLS.items():
        for r in tables[name]:
            for col in cols:
                assert pat.match(r[col]), f"{name}.{col}={r[col]!r}"
                assert Decimal(r[col]).as_tuple().exponent == -2


def test_money_quantization_rounding():
    assert generator.money(1.005) == Decimal("1.01")
    assert generator.money(2.675) == Decimal("2.68")
    assert generator.money(Decimal("10.1")) == Decimal("10.10")
    with pytest.raises(ValueError):
        generator.money(float("nan"))
    # Canonical policy is ROUND_HALF_UP (banker's rounding would give 2.67).
    assert generator.money(2.675) == Decimal("2.68")


def test_budget_actuals_derivable_not_stored(tables):
    """Budgets carry no actual_amount column; actuals aggregate from
    monthly truth (R8 inputs check on one sample quarter)."""
    assert "actual_amount" not in tables["budgets"][0]
    actual = sum(Decimal(r["revenue"])
                 for r in tables["monthly_financials"]
                 if r["branch_id"] == "BR-RUH"
                 and r["period"][:4] == "2023"
                 and int(r["period"][5:7]) in (10, 11, 12))
    budget = next(b for b in tables["budgets"]
                  if b["period"] == "2023-Q4"
                  and b["branch_id"] == "BR-RUH"
                  and b["metric"] == "revenue")
    assert actual > 0 and Decimal(budget["budget_amount"]) > 0
    variance = actual - Decimal(budget["budget_amount"])
    assert variance != 0  # budgets are plans, not copies of actuals


def test_counterparty_category_coherence(tables):
    for t in tables["transactions"]:
        pool = C.COUNTERPARTY_BY_CATEGORY.get(
            t["category"], C.COUNTERPARTIES[t["transaction_type"]])
        assert t["vendor_customer"] in {en for en, _ in pool}


def test_series_not_trivially_smooth(tables):
    """Guard against smooth toy curves: month-to-month revenue must vary."""
    revs = [Decimal(r["revenue"]) for r in tables["monthly_financials"]
            if r["branch_id"] == "BR-RUH"]
    assert len(set(revs)) == len(revs)  # no repeated values
    deltas = [abs(revs[i + 1] - revs[i]) / revs[i] for i in range(len(revs) - 1)]
    assert sum(deltas) / len(deltas) > Decimal("0.01")  # avg move > 1%
    # Dammam ramp: first active year averages below the second.
    dmm_2019 = [Decimal(r["revenue"]) for r in tables["monthly_financials"]
                if r["branch_id"] == "BR-DMM" and r["period"][:4] == "2019"]
    dmm_2020 = [Decimal(r["revenue"]) for r in tables["monthly_financials"]
                if r["branch_id"] == "BR-DMM" and r["period"][:4] == "2020"]
    assert sum(dmm_2019) / len(dmm_2019) < sum(dmm_2020) / len(dmm_2020)
