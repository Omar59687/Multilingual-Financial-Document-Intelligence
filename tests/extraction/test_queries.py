"""Calculation-query library tests (bounded slice, deterministic).

Live DuckDB behind pytest.importorskip (store-test pattern). Synthetic
fixtures only: every numeric below is synthetic. No GT/canonical reads.
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from extraction import queries as Q
from extraction import store as S
from extraction.provenance import Provenance, record_id_for
from extraction.runner_native import run_native
from extraction.schemas import FinancialRecord
from extraction.store import connect, init_db, insert_records


def _rec(metric, period, value, branch=None, doc="DEV-100", el="e1", page=1):
    v = Decimal(value)
    rid = record_id_for({"document_id": doc, "page": page, "element_id": el,
                         "metric": metric, "period": period, "value": v})
    prov = Provenance(document_id=doc, page=page, element_id=el,
                      source_label=metric, display_value=str(v),
                      normalized_value=v, precision="exact-visible",
                      branch_id=branch)
    return FinancialRecord(record_id=rid, metric=metric, period=period,
                           fiscal_year=int(period[:4]), value=v, currency="SAR",
                           branch_id=branch, department_id=None, provenance=prov,
                           document_id=doc, page=page, source_label=metric,
                           display_value=str(v), precision="exact-visible")


@pytest.fixture()
def db():
    con = connect(":memory:")
    try:
        init_db(con)
        rows = [
            _rec("revenue", "2023-01-01", "100.00", "BR-RUH", el="r1"),
            _rec("revenue", "2023-01-01", "50.00", "BR-JED", el="r2"),
            _rec("revenue", "2023-01-01", "999.00", None, el="decoy"),
            _rec("revenue", "2023-02-01", "200.00", "BR-RUH", el="r3"),
            _rec("revenue", "2023-02-01", "100.00", "BR-JED", el="r4"),
            _rec("revenue", "2022-01-01", "80.00", "BR-RUH", el="r5"),
            _rec("revenue", "2022-01-01", "40.00", "BR-JED", el="r6"),
            _rec("revenue", "2023-04-01", "70.00", "BR-RUH", el="r7"),
            _rec("revenue", "2023-04-01", "70.00", "BR-DMM", el="r8"),
            _rec("revenue", "2023-04-01", "30.00", "BR-JED", el="r9"),
            _rec("cogs", "2023-01-01", "60.00", "BR-RUH", el="c1"),
            _rec("cogs", "2023-01-01", "30.00", "BR-JED", el="c2"),
        ]
        assert insert_records(con, rows) == len(rows)
        yield con
    finally:
        con.close()


def _one(con, sql):
    return con.execute(sql).fetchall()


def test_01_company_total_delegates_and_excludes_null_branch(db):
    assert Q.sql_company_total is S.sql_company_total
    assert Q.sql_branch_comparison is S.sql_branch_comparison
    rows = _one(db, Q.sql_company_total("revenue", "2023-01"))
    assert [(p, str(t)) for p, t in rows] == [("2023-01-01", "150.00")]
    # The 999.00 NULL-branch decoy must NOT double-count (R6).


def test_02_period_sum_and_average(db):
    assert _one(db, Q.sql_period_sum("revenue", "2023-01-01", "2023-02-01"))[0][0] == Decimal("450.00")
    assert _one(db, Q.sql_period_sum("revenue", "2023-01-01", "2023-01-01", branch_id="BR-RUH"))[0][0] == Decimal("100.00")
    # Company scope averages MONTHLY TOTALS (Jan 150, Feb 300, Apr 170),
    # never a plain row average (which would vary with branch count).
    assert float(_one(db, Q.sql_period_average("revenue", "2023"))[0][0]) == 206.67
    assert "GROUP BY period" in Q.sql_period_average("revenue", "2023")


def test_03_yoy_growth_with_nullif(db):
    assert "NULLIF" in Q.sql_yoy_growth("revenue", "2023-01-01", "2022-01-01")
    cur, prv, delta, pct = _one(db, Q.sql_yoy_growth("revenue", "2023-01-01", "2022-01-01"))[0]
    assert (str(cur), str(prv), str(delta), float(pct)) == ("150.00", "120.00", "30.00", 25.00)
    zero = _one(db, Q.sql_yoy_growth("revenue", "2023-01-01", "2021-06-01"))[0]
    assert str(zero[2]) == "150.00" and zero[3] is None


def test_04_margin_with_nullif(db):
    assert "NULLIF" in Q.sql_margin("cogs", "revenue", "2023-01-01")
    num, den, pct = _one(db, Q.sql_margin("cogs", "revenue", "2023-01-01"))[0]
    assert (str(num), str(den), float(pct)) == ("90.00", "150.00", 60.00)
    assert _one(db, Q.sql_margin("cogs", "debt", "2023-01-01"))[0][2] is None


def test_05_budget_variance_r8(db):
    assert "NULLIF" in Q.sql_budget_variance("revenue", "2023-01", "150.00")
    actual, budget, var, pct = _one(db, Q.sql_budget_variance("revenue", "2023-01", "150.00"))[0]
    assert (str(actual), str(budget), str(var), float(pct)) == ("150.00", "150.00", "0.00", 0.00)
    over = _one(db, Q.sql_budget_variance("revenue", "2023-01", "100.00"))[0]
    assert (str(over[2]), float(over[3])) == ("50.00", 50.00)
    zero = _one(db, Q.sql_budget_variance("revenue", "2023-01", "0.00"))[0]
    assert zero[3] is None
    with pytest.raises(ValueError):
        Q.sql_budget_variance("revenue", "2023-01", "10.0")
    with pytest.raises(TypeError):
        Q.sql_budget_variance("revenue", "2023-01", True)


def test_06_branch_rank_gap(db):
    rows = _one(db, Q.sql_branch_rank_gap("revenue", "2023-01-01"))
    assert [(b, str(v), r, str(g)) for b, v, r, g in rows] == [
        ("BR-RUH", "100.00", 1, "0.00"), ("BR-JED", "50.00", 2, "50.00")]
    tied = _one(db, Q.sql_branch_rank_gap("revenue", "2023-04-01"))
    assert [(b, str(v), r, str(g)) for b, v, r, g in tied] == [
        ("BR-DMM", "70.00", 1, "0.00"), ("BR-RUH", "70.00", 1, "0.00"),
        ("BR-JED", "30.00", 3, "40.00")]


def test_07_runner_store_query_integration():
    doc = {"document_id": "DEV-101", "filename": "s.csv", "file_type": "csv",
           "language_hint": "en",
           "elements": [
               {"element_id": "e1", "element_type": "csv_row", "row": 1,
                "value": {"date": "2023-03-01", "branch_id": "BR-RUH",
                          "amount": "10.00", "tax_amount": "1.50",
                          "total_amount": "11.50"}},
               {"element_id": "e2", "element_type": "csv_row", "row": 2,
                "value": {"date": "2023-03-01", "branch_id": "BR-JED",
                          "amount": "20.00", "tax_amount": "3.00",
                          "total_amount": "23.00"}},
           ]}
    env = run_native(doc)
    assert env.ok and len(env.records) == 6
    con = connect(":memory:")
    try:
        init_db(con)
        assert insert_records(con, env.records) == 6
        rows = _one(con, Q.sql_company_total("amount", "2023-03"))
        assert [(p, str(t)) for p, t in rows] == [("2023-03-01", "30.00")]
    finally:
        con.close()


def test_08_builder_guards_and_no_duckdb_import():
    import ast as _ast
    with pytest.raises(TypeError):
        Q.sql_period_sum(123, "2023-01-01", "2023-02-01")
    with pytest.raises(TypeError):
        Q.sql_branch_rank_gap("revenue", None)
    with pytest.raises(TypeError):
        Q.sql_margin("a", "b", "2023-01-01", branch_id=123)
    source = (ROOT / "src" / "extraction" / "queries.py").read_text(encoding="utf-8")
    tree = _ast.parse(source)
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] != "duckdb"
        elif isinstance(node, _ast.ImportFrom):
            assert "duckdb" not in (node.module or "")
