"""§9 calculation-evaluation support (Phase 3, bounded slice).

Evaluation-side only: mints eval ``FinancialRecord``s from canonical
monthly rows (evaluation provenance, never documents), runs the frozen
query builders over a truth-populated live target, and compares
against independently recomputed Decimal expectations (different
aggregation path from the SQL under test — Python sums vs SQL sums —
plus reviewer SQL-logic review).

Covers the Evaluation Plan §9 evaluated operations used by the query
library: single-value growth/difference, period sums/averages, margins,
budget variance (budgets.csv binds client-side, R8), branch comparisons
(rank + gap). Cases stay at or below 2023: the 2024 holdout is never
touched, even in evaluation.

Thresholds: none decided here (G4 near-zero-logical-error reported as
measured case results → human gate with G1). All monthly rows are
loaded and minted (2024 included in the target), but every evaluated
case stays at or below 2023: 2024 is never queried, trained on, or
tuned against.
"""

from __future__ import annotations

import csv
from decimal import Decimal
from pathlib import Path
from typing import Any

__all__ = [
    "EVAL_DOC",
    "mint_monthly_records",
    "load_budgets",
    "month_sums",
    "run_case",
]

#: Synthetic-but-truthful eval document identity (valid pattern, no
#: collision with DEV-001..010). Provenance carries no coordinates
#: (evaluation-minted aggregates, documented as such).
EVAL_DOC = "DOC-000000000000"


def mint_monthly_records(monthly_csv: str | Path) -> list:
    """Mint one FinancialRecord per canonical monthly cell (9 metrics).

    Evaluation provenance: source_label = metric, display = exact 2dp,
    precision exact-visible, no coordinates. Raises on malformed input
    (gold gaps are harness bugs, never silent).
    """
    from extraction.provenance import Provenance, record_id_for
    from extraction.schemas import FinancialRecord

    with open(monthly_csv, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    metrics = ("revenue", "cogs", "gross_profit", "operating_expenses",
               "operating_profit", "other_income", "finance_costs",
               "tax_expense", "net_income")
    records = []
    for row in rows:
        for m in metrics:
            value = Decimal(row[m])
            if value.as_tuple().exponent != -2:
                raise ValueError(f"canonical money not 2dp: {row['period']} {row['branch_id']} {m}")
            # Distinct source rows need distinct element_ids: the frozen
            # FIN content-address excludes branch_id (evidence, not core),
            # so same-valued facts across branches would otherwise share
            # an ID and dedup on persist. (Extraction routes always carry
            # distinct native elements; evaluation minting mirrors that.)
            el = f"eval:{row['period']}:{row['branch_id']}:{m}"
            rid = record_id_for({"document_id": EVAL_DOC, "page": None,
                                 "element_id": el, "metric": m,
                                 "period": row["period"], "value": value})
            prov = Provenance(document_id=EVAL_DOC, element_id=el,
                              source_label=m,
                              display_value=format(value, "f"),
                              normalized_value=value,
                              precision="exact-visible",
                              branch_id=row["branch_id"])
            records.append(FinancialRecord(
                record_id=rid, metric=m, period=row["period"],
                fiscal_year=int(row["period"][:4]), value=value,
                currency="SAR", branch_id=row["branch_id"],
                department_id=None, provenance=prov, document_id=EVAL_DOC,
                page=None, source_label=m, display_value=format(value, "f"),
                precision="exact-visible"))
    return records


def load_budgets(budgets_csv: str | Path) -> list[dict]:
    """Load canonical budget rows (client-side binders, never stored)."""
    with open(budgets_csv, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def month_sums(monthly_csv: str | Path, metric: str,
               year: int | None = None, branch: str | None = None,
               months: list[int] | None = None) -> Decimal:
    """Independently recompute a truth sum (Python path, not SQL)."""
    total = Decimal("0.00")
    with open(monthly_csv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            period = row["period"]
            y, m = int(period[:4]), int(period[5:7])
            if year is not None and y != year:
                continue
            if months is not None and m not in months:
                continue
            if branch is not None and row["branch_id"] != branch:
                continue
            total += Decimal(row[metric])
    return total.quantize(Decimal("0.01"))


def run_case(connection: Any, name: str, sql: str, expected: Any,
             tol: Decimal = Decimal("0.01")) -> dict:
    """Execute one calc case; compare rows cell-wise (Decimal exact, float tol).

    Returns {"name", "status" ("PASS"/"FAIL"), "actual", "expected",
    "detail"}. DECIMAL cells compare exactly; float cells within tol;
    NULL/None must match exactly; row counts must match.
    """
    rows = connection.execute(sql).fetchall()

    def _cell_ok(a: Any, e: Any) -> bool:
        if a is None or e is None:
            return a is None and e is None
        if isinstance(a, Decimal) or isinstance(e, Decimal):
            try:
                return Decimal(str(a)) == Decimal(str(e))
            except Exception:
                return False
        if isinstance(a, float) or isinstance(e, float):
            try:
                return abs(float(a) - float(e)) <= float(tol)
            except Exception:
                return False
        return a == e

    ok = len(rows) == len(expected) and all(
        len(a) == len(e) and all(_cell_ok(x, y) for x, y in zip(a, e))
        for a, e in zip(rows, expected))
    return {"name": name, "status": "PASS" if ok else "FAIL",
            "actual": [[str(c) for c in r] for r in rows],
            "expected": [[str(c) for c in r] for r in expected],
            "detail": "" if ok else "cell/row mismatch (see actual vs expected)"}
