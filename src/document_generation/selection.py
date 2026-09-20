"""Deterministic canonical source-record selection for the 10 DEV documents.

Every DEV document's inputs are fixed by an explicit rule (see
docs/DEV_DOCUMENT_SPEC.md). Rules re-resolve against canonical CSVs at
generation time; single-row rules carry pinned cross-checks so generation
fails closed if canonical truth ever drifts under a frozen spec.
"""

import csv
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

CANONICAL_VERSION = "dataset_v0.1"
SEED = 42

# Pinned single-row resolutions (spec §DEV-003/DEV-004). The selection RULE is
# normative; the pin is a tripwire validated at generation time.
PINNED_INVOICE_ID = "INV-2023-0106"       # highest-total 2023 invoice
PINNED_RECEIPT_TX_ID = "TX-2019-013526"   # largest BR-JED expense tx, 2019-06

PNL_METRICS = ("revenue", "cogs", "gross_profit", "operating_expenses",
               "operating_profit", "other_income", "finance_costs",
               "tax_expense", "net_income")

AR_MONTHS = ("يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو",
             "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر")


def load_canonical(canonical_dir) -> dict:
    """Read canonical CSVs; money columns become Decimal."""
    from dataset.generator import MONEY_COLUMNS  # local import: path setup
    canonical_dir = Path(canonical_dir)
    tables = {}
    for name in ("branches", "departments", "monthly_financials",
                 "transactions", "budgets", "invoices",
                 "business_events", "annual_balance_sheet"):
        with open(canonical_dir / f"{name}.csv", newline="",
                  encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            for col in MONEY_COLUMNS.get(name, ()):
                row[col] = Decimal(row[col])
        tables[name] = rows
    return tables


def _sum(rows, metric: str) -> Decimal:
    return sum((r[metric] for r in rows), Decimal("0.00"))


def select_dev001(tables: dict) -> dict:
    rows23 = [r for r in tables["monthly_financials"]
              if r["period"][:4] == "2023"]
    rows22 = [r for r in tables["monthly_financials"]
              if r["period"][:4] == "2022"]
    assert len(rows23) == 36 and len(rows22) == 36
    totals23 = {m: _sum(rows23, m) for m in PNL_METRICS}
    totals22 = {m: _sum(rows22, m) for m in PNL_METRICS}
    branch_rev = {b: _sum([r for r in rows23 if r["branch_id"] == b], "revenue")
                  for b in ("BR-RUH", "BR-JED", "BR-DMM")}
    event = next(e for e in tables["business_events"]
                 if e["event_id"] == "EVT-2023-001")
    return {
        "year": "2023", "rows23": rows23, "rows22": rows22,
        "totals23": totals23, "totals22": totals22,
        "branch_revenue": branch_rev, "event": event,
        "source_record_ids": sorted({f"{r['period']}:{r['branch_id']}"
                                     for r in rows23 + rows22}
                                    | {"EVT-2023-001"}),
    }


def select_dev002(tables: dict) -> dict:
    monthly = sorted(
        (r for r in tables["monthly_financials"]
         if r["period"][:4] == "2022" and r["branch_id"] == "BR-JED"),
        key=lambda r: r["period"])
    assert len(monthly) == 12
    expenses = [t for t in tables["transactions"]
                if t["date"][:4] == "2022" and t["branch_id"] == "BR-JED"
                and t["transaction_type"] == "expense"]
    by_cat = defaultdict(Decimal)
    for t in expenses:
        by_cat[t["category"]] += t["total_amount"]
    return {
        "branch_id": "BR-JED", "year": "2022", "monthly": monthly,
        "total_opex": _sum(monthly, "operating_expenses"),
        "extract": expenses, "extract_by_category": dict(sorted(by_cat.items())),
        "extract_total": sum(by_cat.values(), Decimal("0.00")),
        "source_record_ids": sorted({f"{r['period']}:{r['branch_id']}"
                                     for r in monthly}
                                    | {t["transaction_id"] for t in expenses}),
    }


def select_dev003(tables: dict) -> dict:
    cands = [v for v in tables["invoices"] if v["invoice_date"][:4] == "2023"]
    best = sorted(cands, key=lambda v: (v["total"], v["invoice_id"]),
                  reverse=True)[0]
    if best["invoice_id"] != PINNED_INVOICE_ID:
        raise ValueError(
            f"DEV-003 rule resolved to {best['invoice_id']}, spec pins "
            f"{PINNED_INVOICE_ID}: canonical drift or rule change")
    tx = next(t for t in tables["transactions"]
              if t["transaction_id"] == best["related_transaction_id"])
    return {"invoice": best, "transaction": tx,
            "source_record_ids": [best["invoice_id"], tx["transaction_id"]]}


def select_dev004(tables: dict) -> dict:
    cands = [t for t in tables["transactions"]
             if t["date"][:7] == "2019-06" and t["branch_id"] == "BR-JED"
             and t["transaction_type"] == "expense"]
    best = sorted(cands, key=lambda t: (t["total_amount"],
                                        t["transaction_id"]), reverse=True)[0]
    if best["transaction_id"] != PINNED_RECEIPT_TX_ID:
        raise ValueError(
            f"DEV-004 rule resolved to {best['transaction_id']}, spec pins "
            f"{PINNED_RECEIPT_TX_ID}: canonical drift or rule change")
    return {"transaction": best,
            "source_record_ids": [best["transaction_id"]]}


def select_dev005(tables: dict) -> dict:
    by_id = {e["event_id"]: e for e in tables["business_events"]}
    events = [by_id["EVT-2023-001"], by_id["EVT-2022-001"],
              by_id["EVT-2019-001"]]
    jed_dip_23 = _sum(
        [r for r in tables["monthly_financials"]
         if r["branch_id"] == "BR-JED" and r["period"][:7] in ("2023-09", "2023-10")],
        "revenue")
    jed_dip_22 = _sum(
        [r for r in tables["monthly_financials"]
         if r["branch_id"] == "BR-JED" and r["period"][:7] in ("2022-09", "2022-10")],
        "revenue")
    company_rev_23 = _sum(
        [r for r in tables["monthly_financials"] if r["period"][:4] == "2023"],
        "revenue")
    return {"events": events, "jed_sep_oct_2023": jed_dip_23,
            "jed_sep_oct_2022": jed_dip_22,
            "company_revenue_2023": company_rev_23,
            "source_record_ids": [e["event_id"] for e in events]}


def select_dev006(tables: dict) -> dict:
    rows = sorted(
        (t for t in tables["transactions"] if t["date"][:4] == "2023"),
        key=lambda t: (t["date"], t["transaction_id"]))
    assert len(rows) == 4020
    return {"rows": rows, "year": "2023",
            "source_record_ids": [t["transaction_id"] for t in rows]}


def select_dev007(tables: dict) -> dict:
    months = ("2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06")
    rows = sorted(
        (t for t in tables["transactions"]
         if t["department_id"] == "DEP-LOG" and t["date"][:7] in months),
        key=lambda t: (t["date"], t["transaction_id"]))
    assert len(rows) == 346
    return {"rows": rows, "total": _sum(rows, "total_amount"),
            "source_record_ids": [t["transaction_id"] for t in rows]}


def select_dev008(tables: dict) -> dict:
    annual = []
    for year in range(2015, 2025):
        total = _sum(
            [r for r in tables["monthly_financials"]
             if r["period"][:4] == str(year)], "revenue")
        annual.append((str(year), total))
    return {"annual": annual,
            "source_record_ids": sorted(
                {f"{r['period']}:{r['branch_id']}"
                 for r in tables["monthly_financials"]})}


def select_dev009(tables: dict) -> dict:
    months = ("2023-10", "2023-11", "2023-12")
    rows = [r for r in tables["monthly_financials"]
            if r["period"][:7] in months]
    assert len(rows) == 9
    kpis = {m: _sum(rows, m) for m in ("revenue", "gross_profit",
                                       "operating_expenses", "net_income")}
    branch_rev = {b: _sum([r for r in rows if r["branch_id"] == b], "revenue")
                  for b in ("BR-RUH", "BR-JED", "BR-DMM")}
    budgets = {b["branch_id"]: b["budget_amount"]
               for b in tables["budgets"]
               if b["period"] == "2023-Q4" and b["metric"] == "revenue"
               and b["branch_id"]}
    budget_total = sum(budgets.values(), Decimal("0.00"))
    variance = kpis["revenue"] - budget_total
    return {"kpis": kpis, "branch_revenue": branch_rev,
            "budgets": budgets, "budget_total": budget_total,
            "variance": variance,
            "variance_pct": variance / budget_total * 100,
            "source_record_ids": sorted(
                {f"{r['period']}:{r['branch_id']}" for r in rows}
                | {f"2023-Q4:{b}:revenue" for b in budgets})}


def select_dev010(tables: dict) -> dict:
    row20 = next(r for r in tables["annual_balance_sheet"] if r["year"] == "2020")
    row19 = next(r for r in tables["annual_balance_sheet"] if r["year"] == "2019")
    return {"row2020": row20, "equity2019": row19["equity"],
            "source_record_ids": ["2020", "2019"]}


def resolve_all(tables: dict) -> dict:
    """Resolve every DEV selection; fails closed on any violation."""
    return {
        "DEV-001": select_dev001(tables),
        "DEV-002": select_dev002(tables),
        "DEV-003": select_dev003(tables),
        "DEV-004": select_dev004(tables),
        "DEV-005": select_dev005(tables),
        "DEV-006": select_dev006(tables),
        "DEV-007": select_dev007(tables),
        "DEV-008": select_dev008(tables),
        "DEV-009": select_dev009(tables),
        "DEV-010": select_dev010(tables),
    }
