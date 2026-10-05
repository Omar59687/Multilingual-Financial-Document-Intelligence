"""Deterministic tests for the DuckDB TARGET DESIGN + persistence slices 1-2.

``src/extraction/store.py`` exposes declarative DDL plus ``connect``,
``init_db(connection)`` and ``insert_records`` (caller-owned connections, dedup
``ON CONFLICT``, Decimal passthrough, ordered writes). ``duckdb`` is an
optional extra: no top-level import, only function-local optional imports.
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import extraction.store as S
from extraction.schemas import FinancialRecord


def test_api_surface():
    # Persistence slices 1-2: exactly the approved entry points exist.
    for name in ("connect", "init_db", "insert_records"):
        assert callable(getattr(S, name, None)), f"store.{name} must exist"
    assert "duckdb" not in dir(S)


def _top_level_duckdb_imports(src: str) -> list[str]:
    import ast
    tree = ast.parse(src)
    hits = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                if (a.name or "").split(".")[0] == "duckdb":
                    hits.append(f"import {a.name} (line {node.lineno})")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] == "duckdb":
                hits.append(f"from duckdb (line {node.lineno})")
    return hits


def test_no_duckdb_dependency():
    import subprocess
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, 'src'); import extraction.store; print('ok')"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert out.returncode == 0, out.stderr
    src = (ROOT / "src" / "extraction" / "store.py").read_text(encoding="utf-8")
    # No top-level hard dependency: module must import with stdlib only.
    assert _top_level_duckdb_imports(src) == []
    # The driver is touched only where approved: `duckdb.connect` may appear
    # solely inside `connect()` (docstrings/comments may mention it freely);
    # no code path may close a caller-owned handle.
    import ast as _ast
    _tree = _ast.parse(src)

    def _is_duckdb_connect(n):
        return (isinstance(n, _ast.Attribute) and n.attr == "connect"
                and isinstance(getattr(n, "value", None), _ast.Name)
                and n.value.id == "duckdb")

    _all_connect = [n.lineno for n in _ast.walk(_tree) if _is_duckdb_connect(n)]
    _approved = []
    for _fn in [n for n in _ast.walk(_tree)
                if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))
                and n.name == "connect"]:
        _approved += [n.lineno for n in _ast.walk(_fn) if _is_duckdb_connect(n)]
    assert sorted(_all_connect) == sorted(_approved) and _all_connect != [], \
        f"duckdb.connect must live only in connect(): {_all_connect}"
    _module_level = [
        n.lineno for n in _tree.body
        if isinstance(n, _ast.Expr) and isinstance(getattr(n, "value", None), _ast.Call)
    ]
    assert _module_level == []  # no driver calls at module scope
    _closes = [
        n.lineno for n in _ast.walk(_tree)
        if isinstance(n, _ast.Attribute) and n.attr == "close"
    ]
    assert _closes == [], f"connection close() in store.py: {_closes}"


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


def test_init_db_rejects_bad_connection():
    with pytest.raises(TypeError):
        S.init_db(None)
    with pytest.raises(TypeError):
        S.init_db(object())


def test_init_db_creates_schema_live():
    duckdb = pytest.importorskip("duckdb", reason="optional extra not installed")
    con = duckdb.connect(":memory:")
    try:
        S.init_db(con)  # first run creates table + indexes + view
        S.init_db(con)  # second run proves IF NOT EXISTS / OR REPLACE idempotency
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        assert S.TABLE_NAME in tables
        assert S.VIEW_NAME in tables
        cols = [r[1] for r in con.execute(f"PRAGMA table_info('{S.TABLE_NAME}')").fetchall()]
        assert list(cols) == list(S.COLUMNS)
    finally:
        con.close()


# --- Slice 2 fixtures: synthetic validated records (no benchmark values) ---

def _prov(display="100.00", precision="exact-visible", label="Revenue", document_id="DEV-001"):
    from decimal import Decimal as _D
    return {
        "document_id": document_id,
        "source_label": label,
        "display_value": display,
        "normalized_value": _D(display),
        "precision": precision,
    }


def _record_dict(record_id, metric="revenue", period="2023-01-01", fiscal_year=2023,
                 value="100.00", branch_id="BR-RUH", document_id="DEV-001", page=1,
                 label="Revenue", display="100.00", precision="exact-visible",
                 department_id=None):
    return {
        "record_id": record_id,
        "metric": metric,
        "period": period,
        "fiscal_year": fiscal_year,
        "value": value,
        "currency": "SAR",
        "branch_id": branch_id,
        "department_id": department_id,
        "provenance": _prov(display, precision, label, document_id),
        "document_id": document_id,
        "page": page,
        "source_label": label,
        "display_value": display,
        "precision": precision,
    }


def _record(*args, **kwargs):
    return FinancialRecord.model_validate(_record_dict(*args, **kwargs))


def _live_con():
    duckdb = pytest.importorskip("duckdb", reason="optional extra not installed")
    con = S.connect()
    S.init_db(con)
    return con


def test_connect_roundtrip_live():
    pytest.importorskip("duckdb", reason="optional extra not installed")
    con = S.connect()
    try:
        assert hasattr(con, "execute")
        S.init_db(con)
        assert con.execute("SELECT COUNT(*) FROM financial_records").fetchall() == [(0,)]
    finally:
        con.close()
    with pytest.raises(TypeError):
        S.connect("")
    with pytest.raises(TypeError):
        S.connect(None)


def test_connect_file_path_live(tmp_path):
    pytest.importorskip("duckdb", reason="optional extra not installed")
    db = str(tmp_path / "mizan.duckdb")
    con = S.connect(db)
    try:
        S.init_db(con)
        assert S.insert_records(con, [_record("FIN-abcdef123456")], created_batch="B1") == 1
    finally:
        con.close()
    con2 = S.connect(db)
    try:
        assert con2.execute("SELECT COUNT(*) FROM financial_records").fetchall() == [(1,)]
    finally:
        con2.close()


def test_insert_records_roundtrip_live():
    con = _live_con()
    try:
        recs = [
            _record("FIN-abcdef123456"),
            _record_dict("FIN-000000000001", metric="cogs", value="50.25",
                         display="50.25", label="COGS", branch_id=None, page=2,
                         document_id="DEV-002"),
        ]
        assert S.insert_records(con, recs, created_batch="B7") == 2
        rows = con.execute(
            'SELECT record_id, metric, "value", created_batch FROM financial_records '
            "ORDER BY rowid"
        ).fetchall()
        assert [r[0] for r in rows] == ["FIN-abcdef123456", "FIN-000000000001"]
        assert rows[0][1] == "revenue" and rows[1][1] == "cogs"
        assert rows[0][2] == Decimal("100.00") and type(rows[0][2]).__name__ == "Decimal"
        assert rows[1][2] == Decimal("50.25")
        assert {r[3] for r in rows} == {"B7"}
        assert con.execute('SELECT SUM("value") FROM financial_records').fetchall() == [
            (Decimal("150.25"),)]
    finally:
        con.close()


def test_insert_records_dedup_live():
    con = _live_con()
    try:
        recs = [_record("FIN-abcdef123456"),
                _record("FIN-000000000001", metric="cogs", value="50.25",
                        display="50.25", label="COGS", branch_id=None,
                        document_id="DEV-002", page=2)]
        assert S.insert_records(con, recs) == 2
        assert S.insert_records(con, recs) == 0  # full-batch replay dedups
        mixed = recs[:1] + [_record("FIN-000000000002", metric="cogs", value="10.00",
                                    display="10.00", label="COGS", branch_id="BR-JED",
                                    document_id="DEV-003", page=3)]
        assert S.insert_records(con, mixed) == 1  # only the new row counts
        assert con.execute("SELECT COUNT(*) FROM financial_records").fetchall() == [(3,)]
        assert con.execute(
            'SELECT "value" FROM financial_records WHERE record_id = \'FIN-abcdef123456\''
        ).fetchall() == [(Decimal("100.00"),)]  # conflicting write did not clobber
    finally:
        con.close()


def test_insert_records_ordered_live():
    con = _live_con()
    try:
        recs = [
            _record("FIN-000000000003", period="2023-03-01"),
            _record("FIN-000000000001", period="2023-01-01"),
            _record("FIN-000000000002", period="2023-02-01"),
        ]
        assert S.insert_records(con, recs) == 3
        order = [r[0] for r in con.execute(
            "SELECT record_id FROM financial_records ORDER BY rowid").fetchall()]
        assert order == ["FIN-000000000003", "FIN-000000000001", "FIN-000000000002"]
    finally:
        con.close()


def test_insert_records_atomic_on_invalid_live():
    con = _live_con()
    try:
        good = [_record("FIN-abcdef123456"),
                _record("FIN-000000000001", metric="cogs", value="50.25",
                        display="50.25", label="COGS", branch_id=None,
                        document_id="DEV-002", page=2)]
        bad = _record_dict("FIN-000000000002", metric="not_a_metric")
        with pytest.raises(ValidationError):
            S.insert_records(con, good + [bad])
        assert con.execute("SELECT COUNT(*) FROM financial_records").fetchall() == [(0,)]
    finally:
        con.close()


def test_insert_records_rejects_bad_input():
    con = _live_con()
    try:
        with pytest.raises(TypeError):
            S.insert_records(None, [])
        with pytest.raises(TypeError):
            S.insert_records(con, _record_dict("FIN-abcdef123456"))  # single mapping
        with pytest.raises(TypeError):
            S.insert_records(con, "FIN-abcdef123456")
        with pytest.raises(TypeError):
            S.insert_records(con, [_record("FIN-abcdef123456"), 42])
        with pytest.raises(TypeError):
            S.insert_records(con, [], created_batch=7)
        assert S.insert_records(con, []) == 0
        assert con.execute("SELECT COUNT(*) FROM financial_records").fetchall() == [(0,)]
    finally:
        con.close()


def test_insert_records_created_batch_default_null_live():
    con = _live_con()
    try:
        assert S.insert_records(con, [_record("FIN-abcdef123456")]) == 1
        assert con.execute(
            "SELECT created_batch FROM financial_records").fetchall() == [(None,)]
    finally:
        con.close()


class _FlakyExecute:
    """Duck-typed connection wrapper failing the Nth execute call (fault injection).

    Delegates everything else to the real connection, so ROLLBACK still runs
    against live state and the test proves the handler rolls back real inserts.
    """

    def __init__(self, real, fail_on):
        self._real = real
        self._calls = 0
        self._fail_on = fail_on

    def execute(self, *args, **kwargs):
        self._calls += 1
        if self._calls == self._fail_on:
            raise RuntimeError("injected driver failure")
        return self._real.execute(*args, **kwargs)


def _two_records():
    return [_record("FIN-abcdef123456"),
            _record("FIN-000000000001", metric="cogs", value="50.25",
                    display="50.25", label="COGS", branch_id=None,
                    document_id="DEV-002", page=2)]


def test_insert_records_rolls_back_on_insert_failure_live():
    con = _live_con()
    try:
        # Call order: COUNT(1), BEGIN(2), INSERT(3 -> boom), ROLLBACK.
        flaky = _FlakyExecute(con, fail_on=3)
        with pytest.raises(RuntimeError, match="injected driver failure"):
            S.insert_records(flaky, _two_records())
        assert con.execute("SELECT COUNT(*) FROM financial_records").fetchall() == [(0,)]
    finally:
        con.close()


def test_insert_records_rolls_back_on_commit_failure_live():
    con = _live_con()
    try:
        # Call order: COUNT(1), BEGIN(2), INSERT(3), INSERT(4), COMMIT(5 -> boom), ROLLBACK.
        flaky = _FlakyExecute(con, fail_on=5)
        with pytest.raises(RuntimeError, match="injected driver failure"):
            S.insert_records(flaky, _two_records())
        assert con.execute("SELECT COUNT(*) FROM financial_records").fetchall() == [(0,)]
    finally:
        con.close()
