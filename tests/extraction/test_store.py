"""Deterministic tests for the declarative DuckDB TARGET DESIGN.

Foundation milestone: ``src/extraction/store.py`` is declarative only —
no connections, no init, no inserts, no live database. These tests pin that
boundary (no ``duckdb`` import anywhere in the extraction package) plus the
DDL shape (table, checks, indexes, R6 view) and documented query patterns.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import extraction.store as S


def test_no_persistence_api():
    # Scope correction §2: premature persistence removed for this milestone.
    for name in ("connect", "init_db", "insert_records"):
        assert not hasattr(S, name), f"store.{name} must not exist in foundation milestone"
    assert "duckdb" not in dir(S)


def test_no_duckdb_dependency():
    import subprocess
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, 'src'); import extraction.store; print('ok')"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert out.returncode == 0, out.stderr
    src = (ROOT / "src" / "extraction" / "store.py").read_text(encoding="utf-8")
    assert "import duckdb" not in src
    assert "duckdb.connect" not in src


def test_ddl_shape():
    stmts = S.ddl_statements()
    assert len(stmts) == 5
    joined = "\n".join(stmts)
    assert "CREATE TABLE IF NOT EXISTS financial_records" in joined
    assert '"value" DECIMAL(18, 2) NOT NULL' in joined
    assert '"precision" VARCHAR NOT NULL' in joined
    assert "CHECK (currency = 'SAR')" in joined
    assert "BETWEEN 2015 AND 2024" in joined
    assert "idx_metric_period" in joined and "idx_branch_period" in joined
    assert "idx_document" in joined
    assert "v_monthly_company_totals" in joined
    assert "branch_id IS NOT NULL" in joined


def test_query_patterns_documented():
    assert "branch_id IS NOT NULL" in S.sql_company_total("revenue", "2023")
    assert "branch_id IS NOT NULL" in S.sql_branch_comparison("revenue", "2023-01-01")
    assert 'ROUND(SUM("value"), 2)' in S.sql_company_total("revenue", "2023")
    assert "budget" in S.sql_budget_variance_note().lower()
    assert S.TABLE_NAME == "financial_records"
    assert S.VIEW_NAME == "v_monthly_company_totals"
    assert S.COLUMNS[0] == "record_id" and "precision" in S.COLUMNS
