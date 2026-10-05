"""§9 calculation evaluation over a truth-populated target (harness, thin CLI).

Mints eval records from canonical monthly rows (evaluation provenance),
populates a fresh :memory: target via store.init_db/insert_records,
runs the frozen query builders on ≤2023 cases (2024 holdout never
touched), and compares against independently recomputed Decimal
expectations (Python path vs SQL path + reviewer SQL-logic review).

Writes docs/g4_calc_baseline.json + a console summary. Thresholds are
NOT decided here (G4 near-zero-logical-error reported as measured case
results → human gate with G1).

Usage (repo root):  python scripts/eval_calc.py
"""

import json
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

# Known environment caveat (out of scope — no env repair attempted):
# the installed pandas was compiled against NumPy 1.x while NumPy 2.2.6
# is present, so `import pandas` fails with a long printed traceback.
# DuckDB's parameterized path attempts that import per call and falls
# back cleanly, but the noise is crippling for bulk inserts. Blocking
# the import up front turns it into a fast silent ImportError taking
# the same fallback path (Decimal passthrough verified exact).
# See .factory/state/duckdb-second-slice.md.
sys.modules.setdefault("pandas", None)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation import calc as C
from extraction import queries as Q
from extraction.store import connect, init_db, insert_records

CANON = ROOT / "data" / "canonical" / "dataset_v0.1"


def main():
    monthly_csv = CANON / "monthly_financials.csv"
    budgets_csv = CANON / "budgets.csv"
    records = C.mint_monthly_records(monthly_csv)
    con = connect(":memory:")
    try:
        init_db(con)
        inserted = insert_records(con, records)
        assert inserted == len(records), (inserted, len(records))

        rev23m6 = C.month_sums(monthly_csv, "revenue", year=2023, months=[6])
        rev22m6 = C.month_sums(monthly_csv, "revenue", year=2022, months=[6])
        gross23m6 = C.month_sums(monthly_csv, "gross_profit", year=2023, months=[6])
        avg23 = (sum(C.month_sums(monthly_csv, "revenue", year=2023, months=[m])
                     for m in range(1, 13)) / 12).quantize(Decimal("0.01"))
        q4_rev = sum(C.month_sums(monthly_csv, "revenue", year=2023, months=[m])
                     for m in (10, 11, 12))
        budget_q4 = sum(Decimal(r["budget_amount"]) for r in C.load_budgets(budgets_csv)
                        if r["period"] == "2023-Q4" and r["metric"] == "revenue")
        branches_23m6 = sorted(
            (b, C.month_sums(monthly_csv, "revenue", year=2023, months=[6], branch=b))
            for b in ("BR-RUH", "BR-JED", "BR-DMM"))
        ranked = sorted(branches_23m6, key=lambda t: t[1], reverse=True)
        top = ranked[0][1]
        rank_expected = [(b, v, i + 1, top - v) for i, (b, v) in
                         enumerate(sorted(branches_23m6, key=lambda t: (-t[1], t[0])))]

        cases = [
            ("company-total-revenue-2023",
             Q.sql_company_total("revenue", "2023"),
             [("2023-%s-01" % f"{m:02d}", rev) for m, rev in
              [(m, C.month_sums(monthly_csv, "revenue", year=2023, months=[m])) for m in range(1, 13)]]),
            ("branch-comparison-revenue-2023-06",
             Q.sql_branch_comparison("revenue", "2023-06-01"),
             [(b, v) for b, v in branches_23m6]),
            ("yoy-revenue-2023-06-vs-2022-06",
             Q.sql_yoy_growth("revenue", "2023-06-01", "2022-06-01"),
             [[rev23m6, rev22m6, rev23m6 - rev22m6,
               ((rev23m6 - rev22m6) / rev22m6 * 100).quantize(Decimal("0.01"))]]),
            # Budget window over monthly rows: prefix "2023-1" selects
            # exactly Oct/Nov/Dec 2023 month rows (Q4 grain, lexicographic).
            ("margin-gross-2023-06",
             Q.sql_margin("gross_profit", "revenue", "2023-06-01"),
             [[gross23m6, rev23m6,
               (gross23m6 / rev23m6 * 100).quantize(Decimal("0.01"))]]),
            ("budget-variance-revenue-2023-Q4",
             Q.sql_budget_variance("revenue", "2023-1", budget_q4),
             [[q4_rev, budget_q4, (q4_rev - budget_q4).quantize(Decimal("0.01")),
               ((q4_rev - budget_q4) / budget_q4 * 100).quantize(Decimal("0.01"))]]),
            ("branch-rank-gap-revenue-2023-06",
             Q.sql_branch_rank_gap("revenue", "2023-06-01"),
             [[b, v, r, g] for b, v, r, g in rank_expected]),
            ("monthly-average-revenue-2023",
             Q.sql_period_average("revenue", "2023"),
             [[avg23]]),
        ]
        results = [C.run_case(con, name, sql, expected) for name, sql, expected in cases]
        payload = {"generated_at": datetime.now(timezone.utc).isoformat(),
                   "cases": results,
                   "passed": sum(1 for r in results if r["status"] == "PASS"),
                   "total": len(results)}
        out_path = ROOT / "docs" / "g4_calc_baseline.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        for r in results:
            print(f"{r['status']:5} {r['name']}")
            if r["status"] != "PASS":
                print(f"  actual:   {r['actual']}")
                print(f"  expected: {r['expected']}")
        print(f"{payload['passed']}/{payload['total']} calc cases PASS; JSON: {out_path}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
