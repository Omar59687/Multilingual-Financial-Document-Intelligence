"""Deterministic canonical synthetic-data generator for MizanIQ dataset_v0.1.

Produces the CLEAN BUSINESS TRUTH only: structured rows for branches,
departments, monthly financials, transactions, budgets, invoices,
business events, and the annual balance sheet. No PDFs, no RAG, no models.

Determinism contract: one ``random.Random(seed)`` stream consumed in a
fixed order (monthly -> transactions -> invoices -> budgets -> balance
sheet). Same seed + same code version => byte-identical CSVs. Stable IDs,
deterministic (sorted) record ordering, no timestamps in content rows.
"""

import calendar
import csv
import hashlib
import random
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from . import config as C

CENT = Decimal("0.01")


def money(value) -> Decimal:
    """Quantize any numeric input to SAR with 2dp, ROUND_HALF_UP.

    Floats go through ``str()`` first so binary representation noise never
    leaks into canonical truth. Raises on NaN/inf.
    """
    if isinstance(value, Decimal):
        dec = value
    elif isinstance(value, bool):
        raise TypeError("money() does not accept bool")
    elif isinstance(value, int):
        dec = Decimal(value)
    elif isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"money() got non-finite float: {value!r}")
        dec = Decimal(str(value))
    elif isinstance(value, str):
        dec = Decimal(value)
    else:
        raise TypeError(f"money() got unsupported type: {type(value)}")
    return dec.quantize(CENT, rounding=ROUND_HALF_UP)


def fmt(d: Decimal) -> str:
    """Render a quantized Decimal with exactly 2dp for CSV output."""
    return format(d, ".2f")


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------
def iter_months():
    """Yield (year, month) from 2015-01 through 2024-12 inclusive."""
    for year in range(C.START_YEAR, C.END_YEAR + 1):
        for month in range(1, 13):
            yield year, month


def period_str(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}-01"


def quarter_str(year: int, month: int) -> str:
    return f"{year:04d}-Q{(month - 1) // 3 + 1}"


def branch_open_ym(branch) -> tuple:
    opened = branch["opened_date"]  # YYYY-MM-DD
    return int(opened[0:4]), int(opened[5:7])


def active_branches(year: int, month: int):
    """Branches open in a given month (Dammam only from 2019-01)."""
    return [
        b for b in C.BRANCHES
        if (year, month) >= branch_open_ym(b)
    ]


def _weighted_choice(rng: random.Random, weights: dict):
    total = sum(weights.values())
    draw = rng.random() * total
    cumulative = 0.0
    for key, weight in weights.items():
        cumulative += weight
        if draw < cumulative:
            return key
    return next(reversed(weights))


# ---------------------------------------------------------------------------
# Static tables
# ---------------------------------------------------------------------------
def generate_branches():
    return sorted((dict(b) for b in C.BRANCHES),
                  key=lambda b: b["branch_id"])


def generate_departments():
    return sorted((dict(d) for d in C.DEPARTMENTS),
                  key=lambda d: d["department_id"])


def generate_business_events():
    rows = []
    for e in C.BUSINESS_EVENTS:
        rows.append({
            "event_id": e["event_id"],
            "start_date": e["start_date"],
            "end_date": e["end_date"],
            "event_type": e["event_type"],
            "affected_branch": e["affected_branch"] or "",
            "affected_metric": e["affected_metric"],
            "title_en": e["title_en"],
            "title_ar": e["title_ar"],
            "explanation_en": e["explanation_en"],
            "explanation_ar": e["explanation_ar"],
            "expected_effect": e["expected_effect"],
        })
    rows.sort(key=lambda r: r["event_id"])
    return rows


# ---------------------------------------------------------------------------
# Monthly financials (branch-month grain)
# ---------------------------------------------------------------------------
def _event_factors(branch_id: str, year: int, month: int):
    """Combine multiplicative/additive effects of all events covering a month."""
    revenue_mult, opex_mult, cogs_pp = 1.0, 1.0, 0.0
    stamp = f"{year:04d}-{month:02d}"
    for e in C.BUSINESS_EVENTS:
        if e["start_date"][:7] <= stamp <= e["end_date"][:7]:
            if e["affected_branch"] in (None, branch_id):
                eff = e["effect"]
                revenue_mult *= eff["revenue_mult"]
                opex_mult *= eff["opex_mult"]
                cogs_pp += eff["cogs_pp"]
    return revenue_mult, opex_mult, cogs_pp


def generate_monthly_financials(rng: random.Random):
    rows = []
    for branch in C.BRANCHES:
        p = C.MONTHLY_PARAMS[branch["branch_id"]]
        open_y, open_m = branch_open_ym(branch)
        # Anchor growth to the branch's first ACTIVE month, so Dammam's
        # 2019-01 base is exact and no pre-opening values are ever implied.
        anchor = open_y * 12 + open_m
        for year, month in iter_months():
            if (year, month) < (open_y, open_m):
                continue  # NOT APPLICABLE / NULL — absent, never zero
            elapsed_months = (year * 12 + month) - anchor
            growth = (1.0 + p["annual_growth"]) ** (elapsed_months / 12.0)

            ramadan_peak = C.RAMADAN_PEAK_MONTH[year]
            seasonal = 1.0
            if month == ramadan_peak:
                seasonal += p["ramadan_uplift"]
            elif month in (ramadan_peak - 1, ramadan_peak + 1):
                seasonal += p["ramadan_uplift"] / 2.0
            if month in C.SUMMER_PEAK_MONTHS:
                seasonal += p["summer_uplift"]

            noise = max(0.5, rng.gauss(1.0, p["revenue_noise_cv"]))

            ramp = 1.0
            if p["opening_ramp_months"] and elapsed_months < p["opening_ramp_months"]:
                start = p["opening_ramp_start"]
                ramp = start + (1.0 - start) * (elapsed_months + 1) / p["opening_ramp_months"]

            rev_mult, opex_mult, cogs_pp = _event_factors(
                branch["branch_id"], year, month)

            revenue = money(p["base_revenue"] * growth * seasonal
                            * noise * ramp * rev_mult)

            cogs_ratio = min(0.95, max(0.30,
                             p["cogs_ratio"] + rng.gauss(0.0, p["cogs_noise_sd"])
                             + cogs_pp))
            cogs = money(revenue * Decimal(str(cogs_ratio)))
            gross_profit = revenue - cogs

            opex_ratio = min(0.60, max(0.05,
                             p["opex_ratio"] + rng.gauss(0.0, p["opex_noise_sd"])))
            operating_expenses = money(
                revenue * Decimal(str(opex_ratio)) * Decimal(str(opex_mult)))
            operating_profit = gross_profit - operating_expenses

            if rng.random() < C.OTHER_INCOME_PROB:
                other_income = money(rng.uniform(*C.OTHER_INCOME_BAND))
            else:
                other_income = Decimal("0.00")
            if rng.random() < C.FINANCE_COSTS_PROB:
                finance_costs = money(rng.uniform(*C.FINANCE_COSTS_BAND))
            else:
                finance_costs = Decimal("0.00")

            pretax = operating_profit + other_income - finance_costs
            tax_expense = (money(pretax * C.TAX_RATE_ON_POSITIVE_PRETAX)
                           if pretax > 0 else Decimal("0.00"))
            net_income = pretax - tax_expense

            rows.append({
                "period": period_str(year, month),
                "branch_id": branch["branch_id"],
                "revenue": revenue,
                "cogs": cogs,
                "gross_profit": gross_profit,
                "operating_expenses": operating_expenses,
                "operating_profit": operating_profit,
                "other_income": other_income,
                "finance_costs": finance_costs,
                "tax_expense": tax_expense,
                "net_income": net_income,
            })
    rows.sort(key=lambda r: (r["period"], r["branch_id"]))
    return rows


# ---------------------------------------------------------------------------
# Transactions (~36k selected operational transactions)
# ---------------------------------------------------------------------------
def _tx_description(tx_type: str, category: str, cp_en: str, cp_ar: str):
    return (
        f"{tx_type} / {category} with {cp_en}",
        f"{tx_type} / {category} مع {cp_ar}",
    )


def generate_transactions(rng: random.Random):
    raws = []
    for year, month in iter_months():
        dim = calendar.monthrange(year, month)[1]
        for branch in active_branches(year, month):
            count = C.TRANSACTIONS_PER_BRANCH_MONTH[branch["branch_id"]]
            for _ in range(count):
                day = rng.randint(1, dim)
                dept = _weighted_choice(rng, C.DEPARTMENT_TX_WEIGHTS)
                if rng.random() < C.ADJUSTMENT_SHARE:
                    tx_type = "adjustment"
                else:
                    tx_type = C.DEPARTMENT_TX_TYPE[dept]
                category = rng.choice(C.CATEGORIES_BY_TYPE[tx_type])
                pool = C.COUNTERPARTY_BY_CATEGORY.get(
                    category, C.COUNTERPARTIES[tx_type])
                cp_en, cp_ar = rng.choice(pool)
                mu, sigma = C.AMOUNT_PARAMS[tx_type]
                amount = money(rng.lognormvariate(mu, sigma))
                if tx_type == "adjustment":
                    tax_amount = Decimal("0.00")
                else:
                    tax_amount = money(amount * C.VAT_RATE)
                total_amount = amount + tax_amount
                desc_en, desc_ar = _tx_description(tx_type, category, cp_en, cp_ar)
                raws.append({
                    "date": f"{year:04d}-{month:02d}-{day:02d}",
                    "branch_id": branch["branch_id"],
                    "department_id": dept,
                    "transaction_type": tx_type,
                    "category": category,
                    "vendor_customer": cp_en,
                    "description_en": desc_en,
                    "description_ar": desc_ar,
                    "amount": amount,
                    "tax_amount": tax_amount,
                    "total_amount": total_amount,
                    "payment_method": _weighted_choice(rng, C.PAYMENT_METHODS),
                    "reference_number": "",
                })
    # Deterministic global chronological order, then stable IDs.
    raws.sort(key=lambda r: (r["date"], r["branch_id"], r["department_id"],
                             r["category"], r["vendor_customer"],
                             r["amount"]))
    rows = []
    for seq, r in enumerate(raws, start=1):
        rows.append({"transaction_id": f"TX-{r['date'][:4]}-{seq:06d}", **r})
    return rows


# ---------------------------------------------------------------------------
# Invoices (own canonical entity; INV-YYYY-####; flat 15% VAT)
# ---------------------------------------------------------------------------
def generate_invoices(rng: random.Random, transactions):
    raws = []
    for year, month in iter_months():
        actives = active_branches(year, month)
        weights = {b["branch_id"]: C.INVOICE_BRANCH_WEIGHTS[b["branch_id"]]
                   for b in actives}
        dim = calendar.monthrange(year, month)[1]
        for _ in range(C.INVOICES_PER_COMPANY_MONTH):
            branch_id = _weighted_choice(rng, weights)
            day = rng.randint(1, dim)
            mu, sigma = C.INVOICE_SUBTOTAL_PARAMS
            subtotal = money(rng.lognormvariate(mu, sigma))
            vat = money(subtotal * C.VAT_RATE)
            total = subtotal + vat
            vendor = rng.choice(C.COUNTERPARTIES["purchase"])[0]
            date = f"{year:04d}-{month:02d}-{day:02d}"
            if date >= C.INVOICE_PENDING_CUTOFF:
                status = "pending"
            elif rng.random() < C.INVOICE_OVERDUE_SHARE:
                status = "overdue"
            else:
                status = "paid"
            raws.append({
                "invoice_date": date,
                "vendor_customer": vendor,
                "branch_id": branch_id,
                "subtotal": subtotal,
                "vat": vat,
                "total": total,
                "status": status,
                "related_transaction_id": "",
            })

    # Deterministic per-year sequences after sorting.
    raws.sort(key=lambda r: (r["invoice_date"], r["branch_id"], r["subtotal"]))
    by_year_seq = {}
    rows = []
    for r in raws:
        year = r["invoice_date"][:4]
        by_year_seq[year] = by_year_seq.get(year, 0) + 1
        rows.append({"invoice_id": f"INV-{year}-{by_year_seq[year]:04d}", **r})

    # Link each invoice to one purchase transaction of the same branch-month
    # (both directions: invoice.related_transaction_id + tx.reference_number).
    purchase_pool = {}
    for t in transactions:
        if t["transaction_type"] == "purchase" and not t["reference_number"]:
            purchase_pool.setdefault(
                (t["branch_id"], t["date"][:7]), []).append(t)
    for key in purchase_pool:
        purchase_pool[key].sort(key=lambda t: t["transaction_id"])
    for inv in rows:
        pool = purchase_pool.get((inv["branch_id"], inv["invoice_date"][:7]), [])
        if pool:
            tx = pool.pop(0)
            inv["related_transaction_id"] = tx["transaction_id"]
            tx["reference_number"] = inv["invoice_id"]
    return rows


# ---------------------------------------------------------------------------
# Budgets — quarterly, budget truth ONLY (actuals derived, never stored)
# ---------------------------------------------------------------------------
def _quarter_range(year: int, quarter: int):
    months = {1: (1, 3), 2: (4, 6), 3: (7, 9), 4: (10, 12)}[quarter]
    return months


def _aggregate_quarters(monthly_rows):
    """(branch_id, quarter_str, metric) -> Decimal actual sum."""
    agg = {}
    for r in monthly_rows:
        year, month = int(r["period"][:4]), int(r["period"][5:7])
        q = quarter_str(year, month)
        for metric in C.BUDGET_METRICS:
            agg[(r["branch_id"], q, metric)] = (
                agg.get((r["branch_id"], q, metric), Decimal("0.00"))
                + r[metric]
            )
    return agg


def generate_budgets(rng: random.Random, monthly_rows):
    actuals = _aggregate_quarters(monthly_rows)
    rows = []

    def plan(base: Decimal, metric: str) -> Decimal:
        growth = C.BUDGET_PLANNED_GROWTH[metric] + rng.gauss(0.0, C.BUDGET_NOISE_SD)
        budgeted = money(base * Decimal(str(1.0 + growth)))
        return max(budgeted, Decimal("0.00"))  # budgets are non-negative plans

    # Slice (a): branch x metric, active branches only (no pre-opening truth).
    # Dammam opens exactly on the 2019-Q1 boundary, so no quarter straddles
    # the opening: skipping quarters that start before opening is sufficient.
    for branch in C.BRANCHES:
        open_y, open_m = branch_open_ym(branch)
        for year in range(C.START_YEAR, C.END_YEAR + 1):
            for quarter in range(1, 5):
                first_m, _ = _quarter_range(year, quarter)
                if (year, first_m) < (open_y, open_m):
                    continue
                q = f"{year:04d}-Q{quarter}"
                for metric in C.BUDGET_METRICS:
                    prior = actuals.get(
                        (branch["branch_id"], f"{year - 1:04d}-Q{quarter}", metric))
                    base = prior if prior is not None else actuals[
                        (branch["branch_id"], q, metric)]
                    rows.append({
                        "period": q,
                        "branch_id": branch["branch_id"],
                        "department_id": "",
                        "metric": metric,
                        "budget_amount": plan(base, metric),
                    })

    # Slice (b): company-wide department operating-expenses (supports
    # department budget-vs-actual questions without ERP-grain explosion).
    for year in range(C.START_YEAR, C.END_YEAR + 1):
        for quarter in range(1, 5):
            q = f"{year:04d}-Q{quarter}"
            company_opex = sum(
                actuals.get((b["branch_id"], q, "operating_expenses"),
                            Decimal("0.00"))
                for b in C.BRANCHES
            )
            for dept in C.DEPARTMENTS:
                share = C.DEPT_OPEX_SHARES[dept["department_id"]]
                growth = (C.BUDGET_PLANNED_GROWTH["operating_expenses"]
                          + rng.gauss(0.0, C.BUDGET_NOISE_SD))
                budgeted = money(company_opex * Decimal(str(share))
                                 * Decimal(str(1.0 + growth)))
                rows.append({
                    "period": q,
                    "branch_id": "",
                    "department_id": dept["department_id"],
                    "metric": "operating_expenses",
                    "budget_amount": max(budgeted, Decimal("0.00")),
                })

    rows.sort(key=lambda r: (r["period"], r["branch_id"],
                             r["department_id"], r["metric"]))
    return rows


# ---------------------------------------------------------------------------
# Annual balance sheet — 1 company-level row/year (10 rows, stated figures,
# NOT a double-entry ledger). equity is the plug enforcing the invariant.
# ---------------------------------------------------------------------------
def generate_balance_sheet(rng: random.Random, monthly_rows):
    company_revenue = {}
    for r in monthly_rows:
        year = int(r["period"][:4])
        company_revenue[year] = company_revenue.get(year, Decimal("0.00")) + r["revenue"]

    ratios = C.BALANCE_SHEET_RATIOS
    rows = []
    for year in range(C.START_YEAR, C.END_YEAR + 1):
        rev = company_revenue[year]

        def scaled(ratio: float) -> Decimal:
            return money(rev * Decimal(str(ratio))
                         * Decimal(str(max(0.5, rng.gauss(1.0,
                                                         C.BALANCE_SHEET_NOISE_CV)))))

        ppe_steps = sum(step[0] for y, step in C.BALANCE_SHEET_CAPEX_STEPS.items()
                        if y <= year)
        # Capex-linked borrowing amortizes: each step retains 80%/yr.
        debt_steps = sum(step[1] * (0.8 ** (year - y))
                         for y, step in C.BALANCE_SHEET_CAPEX_STEPS.items()
                         if y <= year)

        cash = scaled(ratios["cash"])
        accounts_receivable = scaled(ratios["accounts_receivable"])
        inventory = scaled(ratios["inventory"])
        other_current_assets = scaled(ratios["other_current_assets"])
        property_and_equipment = scaled(ratios["ppe_base"]) + money(ppe_steps)
        total_assets = (cash + accounts_receivable + inventory
                        + other_current_assets + property_and_equipment)

        accounts_payable = scaled(ratios["accounts_payable"])
        debt = scaled(ratios["debt_base"]) + money(debt_steps)
        other_liabilities = scaled(ratios["other_liabilities"])
        total_liabilities = accounts_payable + debt + other_liabilities

        equity = total_assets - total_liabilities  # plug: invariant by construction
        rows.append({
            "year": str(year),
            "cash": cash,
            "accounts_receivable": accounts_receivable,
            "inventory": inventory,
            "other_current_assets": other_current_assets,
            "property_and_equipment": property_and_equipment,
            "total_assets": total_assets,
            "accounts_payable": accounts_payable,
            "debt": debt,
            "other_liabilities": other_liabilities,
            "total_liabilities": total_liabilities,
            "equity": equity,
        })
    rows.sort(key=lambda r: r["year"])
    return rows


# ---------------------------------------------------------------------------
# Orchestration / output
# ---------------------------------------------------------------------------
MONEY_COLUMNS = {
    "monthly_financials": ("revenue", "cogs", "gross_profit",
                           "operating_expenses", "operating_profit",
                           "other_income", "finance_costs", "tax_expense",
                           "net_income"),
    "transactions": ("amount", "tax_amount", "total_amount"),
    "budgets": ("budget_amount",),
    "invoices": ("subtotal", "vat", "total"),
    "annual_balance_sheet": ("cash", "accounts_receivable", "inventory",
                             "other_current_assets", "property_and_equipment",
                             "total_assets", "accounts_payable", "debt",
                             "other_liabilities", "total_liabilities",
                             "equity"),
}

COLUMN_ORDER = {
    "branches": ("branch_id", "branch_name_en", "branch_name_ar", "city",
                 "opened_date", "status"),
    "departments": ("department_id", "department_name_en",
                    "department_name_ar"),
    "monthly_financials": ("period", "branch_id") + MONEY_COLUMNS["monthly_financials"],
    "transactions": ("transaction_id", "date", "branch_id", "department_id",
                     "transaction_type", "category", "vendor_customer",
                     "description_en", "description_ar")
    + MONEY_COLUMNS["transactions"] + ("payment_method", "reference_number"),
    "budgets": ("period", "branch_id", "department_id", "metric",
                "budget_amount"),
    "invoices": ("invoice_id", "invoice_date", "vendor_customer", "branch_id",
                 "subtotal", "vat", "total", "status",
                 "related_transaction_id"),
    "business_events": ("event_id", "start_date", "end_date", "event_type",
                        "affected_branch", "affected_metric", "title_en",
                        "title_ar", "explanation_en", "explanation_ar",
                        "expected_effect"),
    "annual_balance_sheet": ("year",) + MONEY_COLUMNS["annual_balance_sheet"],
}


def generate_all(seed: int = C.RANDOM_SEED) -> dict:
    """Run the full deterministic pipeline; returns table-name -> rows.

    Money values are Decimals (quantized). Use ``write_dataset`` to render
    CSVs. Consumes one RNG stream in fixed order.
    """
    rng = random.Random(seed)
    branches = generate_branches()
    departments = generate_departments()
    monthly = generate_monthly_financials(rng)
    transactions = generate_transactions(rng)
    invoices = generate_invoices(rng, transactions)
    budgets = generate_budgets(rng, monthly)
    balance = generate_balance_sheet(rng, monthly)
    events = generate_business_events()
    return {
        "branches": branches,
        "departments": departments,
        "monthly_financials": monthly,
        "transactions": transactions,
        "budgets": budgets,
        "invoices": invoices,
        "business_events": events,
        "annual_balance_sheet": balance,
    }


def _render_row(table: str, row: dict) -> dict:
    out = {}
    for col in COLUMN_ORDER[table]:
        value = row[col]
        if col in MONEY_COLUMNS.get(table, ()):
            out[col] = fmt(value)
        else:
            out[col] = value
    return out


def write_dataset(tables: dict, out_dir: Path) -> list:
    """Write all tables as CSVs (UTF-8, LF). Returns written file names."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for table in COLUMN_ORDER:
        path = out_dir / f"{table}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(COLUMN_ORDER[table]), lineterminator="\n")
            writer.writeheader()
            for row in tables[table]:
                writer.writerow(_render_row(table, row))
        written.append(path.name)
    return written


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(tables: dict, out_dir: Path, seed: int = C.RANDOM_SEED) -> dict:
    """Manifest with version/seed/counts/hashes/notes.

    ``generated_at`` is informational ONLY and excluded from determinism
    comparisons (see tests + §14 of the task spec).
    """
    out_dir = Path(out_dir)
    linked_invoices = sum(1 for t in tables["transactions"] if t["reference_number"])
    manifest = {
        "dataset_version": C.DATASET_VERSION,
        "schema_version": C.SCHEMA_VERSION,
        "random_seed": seed,
        "company_en": C.COMPANY_NAME_EN,
        "company_ar": C.COMPANY_NAME_AR,
        "historical_range": f"{C.START_YEAR}-01-01..{C.END_YEAR}-12-31",
        "forecast_holdout": "2024-01-01..2024-12-31 (default final holdout)",
        "currency": C.CURRENCY,
        "money_policy": "SAR, 2 decimal places, ROUND_HALF_UP; Decimal internally",
        "vat_policy": ("simplified controlled flat 15% on all invoice/transaction "
                       "tax fields regardless of date (V1, not historical)"),
        "tax_policy": ("generic synthetic tax_expense at 11% of positive pre-tax "
                       "profit; Saudi Zakat rules NOT modeled"),
        "generation_date_policy": ("generated_at is informational only and is "
                                   "excluded from determinism checks"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "row_counts": {name: len(rows) for name, rows in tables.items()},
        "transaction_counts_per_branch_month":
            dict(C.TRANSACTIONS_PER_BRANCH_MONTH),
        "invoice_linkage": {
            "invoices_with_transaction": sum(
                1 for i in tables["invoices"] if i["related_transaction_id"]),
            "purchase_transactions_with_invoice": linked_invoices,
        },
        "simplification_notes": [
            "Dammam (BR-DMM) opened 2019-01-01; pre-opening periods are absent (NULL), never zero.",
            "Budgets store budget truth only; actuals are derived by aggregating monthly_financials.",
            "Annual balance sheet rows are stated-figure fixtures (equity is the plug); no double-entry ledger, no cash-flow reconciliation in V1.",
            "Transactions are selected operational transactions (~36k target), not every POS receipt.",
            "Below-operating lines (other_income, finance_costs, tax_expense) are sparse by design.",
        ],
        "files": {},
    }
    for name in COLUMN_ORDER:
        manifest["files"][f"{name}.csv"] = {
            "rows": len(tables[name]),
            "sha256": sha256_file(out_dir / f"{name}.csv"),
        }
    manifest_path = out_dir / C.MANIFEST_FILE
    manifest_path.write_text(
        __import__("json").dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest
