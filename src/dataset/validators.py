"""Explicit validators for the MizanIQ canonical dataset (Phase 0).

Validation is fail-closed: the first violation raises ``ValidationError``
and generation must stop. Nothing is silently repaired.

``validate_all(data_dir)`` checks the written CSVs + manifest on disk, so
it verifies exactly what downstream phases will consume.
"""

import csv
import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path

from . import config as C
from .generator import COLUMN_ORDER, MONEY_COLUMNS

MONEY_RE = re.compile(r"^-?\d+\.\d{2}$")
PERIOD_RE = re.compile(r"^(\d{4})-(\d{2})-01$")
QUARTER_RE = re.compile(r"^(\d{4})-Q([1-4])$")
DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")

EVENT_TYPES = {
    "branch_opening", "warehouse_expansion", "supply_cost_increase",
    "temporary_closure", "promotional_campaign", "logistics_disruption",
    "demand_shock",
}


class ValidationError(Exception):
    """Raised on the first canonical-truth invariant violation."""


def _fail(msg: str) -> None:
    raise ValidationError(msg)


def _read(data_dir: Path, name: str) -> list:
    path = Path(data_dir) / f"{name}.csv"
    if not path.exists():
        _fail(f"missing output file: {path.name}")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    expected = list(COLUMN_ORDER[name])
    if (rows and list(rows[0].keys()) != expected) or (
            not rows and _header_of(path) != expected):
        _fail(f"{path.name}: header mismatch; expected {expected}")
    return rows


def _header_of(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return next(csv.reader(f))


def _money(table: str, row: dict, col: str, i: int) -> Decimal:
    raw = row[col]
    if not MONEY_RE.match(raw or ""):
        _fail(f"{table}[{i}].{col}={raw!r}: not SAR 2dp money")
    value = Decimal(raw)
    if value.as_tuple().exponent != -2:
        _fail(f"{table}[{i}].{col}={raw!r}: must carry exactly 2dp")
    return value


def _nonneg(table: str, row: dict, col: str, i: int, value: Decimal) -> None:
    if value < 0:
        _fail(f"{table}[{i}].{col}={value}: must never be negative")


def _ym(date: str) -> tuple:
    return int(date[0:4]), int(date[5:7])


def validate_all(data_dir) -> dict:
    """Validate every canonical file in ``data_dir``. Returns row counts."""
    data_dir = Path(data_dir)
    tables = {name: _read(data_dir, name) for name in COLUMN_ORDER}

    branches = tables["branches"]
    departments = tables["departments"]
    monthly = tables["monthly_financials"]
    transactions = tables["transactions"]
    budgets = tables["budgets"]
    invoices = tables["invoices"]
    events = tables["business_events"]
    balance = tables["annual_balance_sheet"]

    branch_ids = {b["branch_id"] for b in branches}
    dept_ids = {d["department_id"] for d in departments}
    tx_ids = {t["transaction_id"] for t in transactions}

    # -- static tables -----------------------------------------------------
    if len(branches) != 3 or branch_ids != {"BR-RUH", "BR-JED", "BR-DMM"}:
        _fail(f"branches: expected BR-RUH/BR-JED/BR-DMM, got {sorted(branch_ids)}")
    dmm = next(b for b in branches if b["branch_id"] == "BR-DMM")
    if dmm["opened_date"] != "2019-01-01":
        _fail(f"BR-DMM opened_date must be 2019-01-01, got {dmm['opened_date']}")
    for i, b in enumerate(branches):
        for col in ("branch_id", "branch_name_en", "branch_name_ar",
                    "city", "opened_date", "status"):
            if not b[col]:
                _fail(f"branches[{i}].{col}: required value missing")
        if b["status"] not in ("open", "closed", "planned"):
            _fail(f"branches[{i}].status={b['status']!r}: bad enum")

    if len(departments) != 4 or dept_ids != {"DEP-SALES", "DEP-PROC",
                                            "DEP-LOG", "DEP-ADM"}:
        _fail(f"departments: expected 4 approved IDs, got {sorted(dept_ids)}")
    for i, d in enumerate(departments):
        for col in ("department_id", "department_name_en",
                    "department_name_ar"):
            if not d[col]:
                _fail(f"departments[{i}].{col}: required value missing")

    if not (8 <= len(events) <= 12):
        _fail(f"business_events: expected 8-12 curated events, got {len(events)}")
    seen_events = set()
    for i, e in enumerate(events):
        if e["event_id"] in seen_events:
            _fail(f"business_events[{i}]: duplicate {e['event_id']}")
        seen_events.add(e["event_id"])
        for col in ("event_id", "start_date", "end_date", "event_type",
                    "affected_metric", "title_en", "title_ar",
                    "explanation_en", "explanation_ar", "expected_effect"):
            if not e[col]:
                _fail(f"business_events[{i}].{col}: required value missing")
        if e["event_type"] not in EVENT_TYPES:
            _fail(f"business_events[{i}].event_type={e['event_type']!r}")
        if e["affected_branch"] and e["affected_branch"] not in branch_ids:
            _fail(f"business_events[{i}].affected_branch: bad FK")
        if not (DATE_RE.match(e["start_date"]) and DATE_RE.match(e["end_date"])):
            _fail(f"business_events[{i}]: bad ISO dates")
        if e["start_date"] > e["end_date"]:
            _fail(f"business_events[{i}]: start_date after end_date")

    # -- monthly financials ------------------------------------------------
    if len(monthly) != 312:  # 120 + 120 + 72 active branch-months
        _fail(f"monthly_financials: expected 312 rows, got {len(monthly)}")
    seen_months = set()
    for i, r in enumerate(monthly):
        if r["branch_id"] not in branch_ids:
            _fail(f"monthly_financials[{i}]: bad branch FK {r['branch_id']!r}")
        m = PERIOD_RE.match(r["period"] or "")
        if not m:
            _fail(f"monthly_financials[{i}].period={r['period']!r}: "
                  "must be YYYY-MM-01")
        if not (C.START_YEAR <= int(m.group(1)) <= C.END_YEAR
                and 1 <= int(m.group(2)) <= 12):
            _fail(f"monthly_financials[{i}].period out of range")
        if r["branch_id"] == "BR-DMM" and r["period"] < "2019-01-01":
            _fail(f"monthly_financials[{i}]: Dammam row before 2019-01 opening")
        if (r["period"], r["branch_id"]) in seen_months:
            _fail(f"monthly_financials[{i}]: duplicate branch-month")
        seen_months.add((r["period"], r["branch_id"]))

        v = {c: _money("monthly_financials", r, c, i)
             for c in MONEY_COLUMNS["monthly_financials"]}
        for c in ("revenue", "cogs", "operating_expenses", "other_income",
                  "finance_costs", "tax_expense"):
            _nonneg("monthly_financials", r, c, i, v[c])
        if v["gross_profit"] != v["revenue"] - v["cogs"]:  # R1
            _fail(f"monthly_financials[{i}]: R1 gross_profit violated")
        if v["operating_profit"] != v["gross_profit"] - v["operating_expenses"]:  # R2
            _fail(f"monthly_financials[{i}]: R2 operating_profit violated")
        if v["net_income"] != (v["operating_profit"] + v["other_income"]
                               - v["finance_costs"] - v["tax_expense"]):  # R3
            _fail(f"monthly_financials[{i}]: R3 net_income violated")

    # -- transactions ------------------------------------------------------
    expected_tx = 0
    for b in C.BRANCHES:
        opened = (int(b["opened_date"][:4]), int(b["opened_date"][5:7]))
        active_months = sum(
            1 for y in range(C.START_YEAR, C.END_YEAR + 1) for m in range(1, 13)
            if (y, m) >= opened
        )
        expected_tx += C.TRANSACTIONS_PER_BRANCH_MONTH[b["branch_id"]] * active_months
    if len(transactions) != expected_tx:
        _fail(f"transactions: expected {expected_tx} rows, got {len(transactions)}")
    seen_tx = set()
    for i, t in enumerate(transactions):
        if t["transaction_id"] in seen_tx:
            _fail(f"transactions[{i}]: duplicate {t['transaction_id']}")
        seen_tx.add(t["transaction_id"])
        if t["branch_id"] not in branch_ids:
            _fail(f"transactions[{i}]: bad branch FK")
        if t["department_id"] not in dept_ids:
            _fail(f"transactions[{i}]: bad department FK")
        dm = DATE_RE.match(t["date"] or "")
        if not dm or not (C.START_YEAR <= int(dm.group(1)) <= C.END_YEAR):
            _fail(f"transactions[{i}].date={t['date']!r}: bad/out-of-range date")
        if t["branch_id"] == "BR-DMM" and t["date"] < "2019-01-01":
            _fail(f"transactions[{i}]: Dammam transaction before opening")
        if t["transaction_type"] not in C.TRANSACTION_TYPES:
            _fail(f"transactions[{i}]: bad transaction_type")
        if t["category"] not in C.CATEGORIES_BY_TYPE[t["transaction_type"]]:
            _fail(f"transactions[{i}]: category {t['category']!r} not valid "
                  f"for type {t['transaction_type']!r}")
        for col in ("vendor_customer", "description_en", "description_ar"):
            if not t[col]:
                _fail(f"transactions[{i}].{col}: required value missing")
        if t["payment_method"] not in C.PAYMENT_METHODS:
            _fail(f"transactions[{i}]: bad payment_method")
        amount = _money("transactions", t, "amount", i)
        tax = _money("transactions", t, "tax_amount", i)
        total = _money("transactions", t, "total_amount", i)
        _nonneg("transactions", t, "amount", i, amount)
        if total != amount + tax:  # R5
            _fail(f"transactions[{i}]: R5 total_amount violated")
        if t["transaction_type"] == "adjustment":
            if tax != Decimal("0.00"):
                _fail(f"transactions[{i}]: adjustment must carry zero tax")
        elif tax != (amount * C.VAT_RATE).quantize(
                Decimal("0.01"), rounding="ROUND_HALF_UP"):
            _fail(f"transactions[{i}]: non-adjustment tax must be 15% of amount")

    # -- invoices ----------------------------------------------------------
    if len(invoices) != C.INVOICES_PER_COMPANY_MONTH * 120:  # 2,400
        _fail(f"invoices: expected 2400 rows, got {len(invoices)}")
    seen_inv = set()
    for i, v in enumerate(invoices):
        if v["invoice_id"] in seen_inv:
            _fail(f"invoices[{i}]: duplicate {v['invoice_id']}")
        seen_inv.add(v["invoice_id"])
        if not re.match(r"^INV-\d{4}-\d{4}$", v["invoice_id"] or ""):
            _fail(f"invoices[{i}].invoice_id={v['invoice_id']!r}: bad pattern")
        if v["invoice_id"][4:8] != v["invoice_date"][:4]:
            _fail(f"invoices[{i}]: invoice_id year must match invoice_date")
        dm = DATE_RE.match(v["invoice_date"] or "")
        if not dm or not (C.START_YEAR <= int(dm.group(1)) <= C.END_YEAR):
            _fail(f"invoices[{i}].invoice_date: bad/out-of-range date")
        if v["branch_id"] not in branch_ids:
            _fail(f"invoices[{i}]: bad branch FK")
        if v["branch_id"] == "BR-DMM" and v["invoice_date"] < "2019-01-01":
            _fail(f"invoices[{i}]: Dammam invoice before opening")
        if not v["vendor_customer"]:
            _fail(f"invoices[{i}].vendor_customer: required value missing")
        if v["status"] not in ("paid", "pending", "overdue"):
            _fail(f"invoices[{i}]: bad status enum")
        subtotal = _money("invoices", v, "subtotal", i)
        vat = _money("invoices", v, "vat", i)
        total = _money("invoices", v, "total", i)
        _nonneg("invoices", v, "subtotal", i, subtotal)
        if vat != (subtotal * C.VAT_RATE).quantize(
                Decimal("0.01"), rounding="ROUND_HALF_UP"):
            _fail(f"invoices[{i}]: vat must be flat 15% of subtotal")
        if total != subtotal + vat:  # R4
            _fail(f"invoices[{i}]: R4 total violated")
        if v["related_transaction_id"] and v["related_transaction_id"] not in tx_ids:
            _fail(f"invoices[{i}]: bad related_transaction_id FK")

    # bidirectional link consistency
    tx_by_id = {t["transaction_id"]: t for t in transactions}
    for v in invoices:
        rel = v["related_transaction_id"]
        if rel and tx_by_id[rel]["reference_number"] != v["invoice_id"]:
            _fail(f"invoices: {v['invoice_id']} -> {rel} not mirrored on tx")
    for t in transactions:
        ref = t["reference_number"]
        if ref and ref not in seen_inv:
            _fail(f"transactions: {t['transaction_id']} references "
                  f"unknown invoice {ref}")

    # -- budgets (truth only: no actual_amount column may exist) -----------
    for i, b in enumerate(budgets):
        qm = QUARTER_RE.match(b["period"] or "")
        if not qm or not (C.START_YEAR <= int(qm.group(1)) <= C.END_YEAR):
            _fail(f"budgets[{i}].period={b['period']!r}: must be YYYY-Qn in range")
        has_branch, has_dept = bool(b["branch_id"]), bool(b["department_id"])
        if has_branch == has_dept:  # exactly one grain per row
            _fail(f"budgets[{i}]: exactly one of branch_id/department_id "
                  "must be set")
        if has_branch and b["branch_id"] not in branch_ids:
            _fail(f"budgets[{i}]: bad branch FK")
        if has_dept and b["department_id"] not in dept_ids:
            _fail(f"budgets[{i}]: bad department FK")
        if b["metric"] not in C.BUDGET_METRICS:
            _fail(f"budgets[{i}]: bad metric {b['metric']!r}")
        if has_dept and b["metric"] != "operating_expenses":
            _fail(f"budgets[{i}]: department slice is opex-only")
        if b["branch_id"] == "BR-DMM" and b["period"] < "2019-Q1":
            _fail(f"budgets[{i}]: Dammam budget before opening")
        amt = _money("budgets", b, "budget_amount", i)
        _nonneg("budgets", b, "budget_amount", i, amt)
    # exact expected count: RUH/JED 40q*4 + DMM 24q*4 + dept 40q*4
    if len(budgets) != (40 * 4 * 2) + (24 * 4) + (40 * 4):
        _fail(f"budgets: expected 576 rows, got {len(budgets)}")

    # -- annual balance sheet (exactly 10 company-level rows) --------------
    if len(balance) != 10:
        _fail(f"annual_balance_sheet: expected 10 rows, got {len(balance)}")
    seen_years = set()
    for i, r in enumerate(balance):
        if r["year"] in seen_years:
            _fail(f"annual_balance_sheet[{i}]: duplicate year")
        seen_years.add(r["year"])
        if not re.match(r"^(201[5-9]|202[0-4])$", r["year"] or ""):
            _fail(f"annual_balance_sheet[{i}].year={r['year']!r}")
        v = {c: _money("annual_balance_sheet", r, c, i)
             for c in MONEY_COLUMNS["annual_balance_sheet"]}
        for c in ("cash", "accounts_receivable", "inventory",
                  "other_current_assets", "property_and_equipment",
                  "total_assets", "accounts_payable", "debt",
                  "other_liabilities", "total_liabilities"):
            _nonneg("annual_balance_sheet", r, c, i, v[c])
        if v["total_assets"] != (v["cash"] + v["accounts_receivable"]
                                 + v["inventory"] + v["other_current_assets"]
                                 + v["property_and_equipment"]):  # R9
            _fail(f"annual_balance_sheet[{i}]: R9 total_assets violated")
        if v["total_liabilities"] != (v["accounts_payable"] + v["debt"]
                                      + v["other_liabilities"]):  # R10
            _fail(f"annual_balance_sheet[{i}]: R10 total_liabilities violated")
        if v["total_assets"] != v["total_liabilities"] + v["equity"]:  # R11
            _fail(f"annual_balance_sheet[{i}]: R11 balance-sheet equation violated")

    # -- deterministic ordering --------------------------------------------
    order_keys = {
        "branches": lambda r: r["branch_id"],
        "departments": lambda r: r["department_id"],
        "monthly_financials": lambda r: (r["period"], r["branch_id"]),
        "transactions": lambda r: r["transaction_id"],
        "budgets": lambda r: (r["period"], r["branch_id"],
                              r["department_id"], r["metric"]),
        "invoices": lambda r: r["invoice_id"],
        "business_events": lambda r: r["event_id"],
        "annual_balance_sheet": lambda r: r["year"],
    }
    for name, key in order_keys.items():
        keys = [key(r) for r in tables[name]]
        if keys != sorted(keys):
            _fail(f"{name}.csv: rows are not in deterministic PK order")

    # -- manifest -----------------------------------------------------------
    manifest_path = data_dir / C.MANIFEST_FILE
    if not manifest_path.exists():
        _fail("missing manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_version") != C.DATASET_VERSION:
        _fail("manifest dataset_version mismatch")
    if manifest.get("random_seed") != C.RANDOM_SEED:
        _fail("manifest random_seed mismatch")
    if manifest.get("currency") != C.CURRENCY:
        _fail("manifest currency must be SAR")
    for name in COLUMN_ORDER:
        fname = f"{name}.csv"
        if manifest.get("row_counts", {}).get(name) != len(tables[name]):
            _fail(f"manifest row_counts[{name}] mismatch")
        digest = hashlib.sha256(
            (data_dir / fname).read_bytes()).hexdigest()
        if manifest.get("files", {}).get(fname, {}).get("sha256") != digest:
            _fail(f"manifest sha256 mismatch for {fname}")

    return {name: len(rows) for name, rows in tables.items()}
