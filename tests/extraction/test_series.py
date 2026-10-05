"""Forecasting-handoff tests (bounded slice, deterministic).

Live DuckDB behind pytest.importorskip. Synthetic fixtures only.
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from extraction.provenance import Provenance, record_id_for
from extraction.schemas import FinancialRecord
from extraction.series import DEFAULT_END_PERIOD, fetch_monthly_series
from extraction.store import connect, init_db, insert_records


def _rec(metric, period, value, branch=None, doc="DEV-102", el="e1", page=1):
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
            # 2023-02 intentionally absent (absence, never zero-fill).
            _rec("revenue", "2023-03-01", "200.00", "BR-RUH", el="r3"),
            _rec("revenue", "2024-01-01", "300.00", "BR-RUH", el="hold"),
            _rec("revenue", "2023-Q1", "10.00", "BR-RUH", el="qgrain"),
        ]
        assert insert_records(con, rows) == len(rows)
        yield con
    finally:
        con.close()


def test_01_company_series_r6_and_ordering(db):
    assert [(p, str(v)) for p, v in fetch_monthly_series(db, "revenue")] == [
        ("2023-01-01", "150.00"), ("2023-03-01", "200.00")]
    # Decoy excluded (R6); Feb absent (never zero-filled); Q1 grain excluded;
    # 2024 holdout excluded by default.


def test_02_branch_scope_and_explicit_holdout_opt_in(db):
    assert [(p, str(v)) for p, v in fetch_monthly_series(db, "revenue", branch_id="BR-JED")] == [
        ("2023-01-01", "50.00")]
    assert [(p, str(v)) for p, v in fetch_monthly_series(db, "revenue", end_period="2024-12-01")] == [
        ("2023-01-01", "150.00"), ("2023-03-01", "200.00"), ("2024-01-01", "300.00")]
    assert DEFAULT_END_PERIOD == "2023-12-01"


def test_03_decimal_exactness_and_guards(db):
    series = fetch_monthly_series(db, "revenue")
    assert all(isinstance(v, Decimal) and v.as_tuple().exponent == -2 for _, v in series)
    with pytest.raises(TypeError):
        fetch_monthly_series(None, "revenue")
    with pytest.raises(TypeError):
        fetch_monthly_series(db, "")
    with pytest.raises(ValueError):
        fetch_monthly_series(db, "revenue", end_period="2023")
    with pytest.raises(ValueError):
        fetch_monthly_series(db, "revenue", end_period="Q1")
    with pytest.raises(TypeError):
        fetch_monthly_series(db, "revenue", branch_id=123)


def test_04_no_duckdb_import_and_empty_metric():
    import ast as _ast
    source = (ROOT / "src" / "extraction" / "series.py").read_text(encoding="utf-8")
    tree = _ast.parse(source)
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] != "duckdb"
        elif isinstance(node, _ast.ImportFrom):
            assert "duckdb" not in (node.module or "")
    con = connect(":memory:")
    try:
        init_db(con)
        assert fetch_monthly_series(con, "revenue") == []
    finally:
        con.close()
