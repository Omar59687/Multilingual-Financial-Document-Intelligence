"""Native statement-tables adapter tests (bounded slice, deterministic).

Synthetic fixtures only: every numeric below is synthetic. No benchmark
or ground-truth reads, no network, no duckdb, no randomness.
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from extraction.runner_statements import (
    normalize_pl_label,
    run_statements,
)


def _cell(eid, page, ti, row, col, text):
    return {"element_id": eid, "element_type": "table_cell", "page": page,
            "row": row, "column": col, "table_index": ti, "text": text,
            "meta": {}}


def _stmt_doc():
    return {
        "document_id": "DEV-024",
        "filename": "synthetic.pdf",
        "file_type": "pdf",
        "language_hint": "en",
        "elements": [
            _cell("e-h0", 1, 0, 0, 0, "Line item"),
            _cell("e-h1", 1, 0, 0, 1, "FY 2023 (SAR)"),
            _cell("e-h2", 1, 0, 0, 2, "FY 2022 (SAR)"),
            _cell("e-h3", 1, 0, 0, 3, "Change"),
            _cell("e-r1c0", 1, 0, 1, 0, "Revenue"),
            _cell("e-r1c1", 1, 0, 1, 1, "10.00"),
            _cell("e-r1c2", 1, 0, 1, 2, "9.00"),
            _cell("e-r1c3", 1, 0, 1, 3, "+11.1%"),
            _cell("e-r2c0", 1, 0, 2, 0, "Cost of sales (COGS)"),
            _cell("e-r2c1", 1, 0, 2, 1, "6.00"),
            _cell("e-r2c2", 1, 0, 2, 2, "5.00"),
            _cell("e-r2c3", 1, 0, 2, 3, "+20.0%"),
            _cell("e-r3c0", 1, 0, 3, 0, "Gross profit"),
            _cell("e-r3c1", 1, 0, 3, 1, "4.00"),
            _cell("e-r3c2", 1, 0, 3, 2, "4.00"),
            _cell("e-r3c3", 1, 0, 3, 3, "+0.0%"),
        ],
    }


def test_01_statement_grid_emits_per_column_records():
    env = run_statements(_stmt_doc(), column_periods={1: "2023", 2: "2022"})
    assert env.ok is True
    assert [(r.metric.value, r.period, str(r.value)) for r in env.records] == [
        ("revenue", "2023", "10.00"), ("revenue", "2022", "9.00"),
        ("cogs", "2023", "6.00"), ("cogs", "2022", "5.00"),
        ("gross_profit", "2023", "4.00"), ("gross_profit", "2022", "4.00")]
    first = env.records[0]
    assert first.fiscal_year == 2023
    assert first.provenance.extraction_route == "native"
    assert first.provenance.element_id == "e-r1c1"
    assert first.page == 1
    assert first.branch_id is None
    # Header row self-skips via W_LABEL; Change% column silently skipped.
    assert any("W_LABEL" in w and "Line item" in w for w in env.warnings)
    assert not any("Change" in w for w in env.warnings)


def test_02_pl_alias_and_generic_rule():
    assert normalize_pl_label("Cost of sales (COGS)") == "cogs"
    assert normalize_pl_label("Gross profit") == "gross_profit"
    assert normalize_pl_label("VAT 15%") == "vat"
    assert normalize_pl_label("Total due") == "total"
    assert normalize_pl_label("TOTAL ASSETS") == "total_assets"
    assert normalize_pl_label("Line item") is None
    assert normalize_pl_label("") is None
    assert normalize_pl_label(None) is None


def test_03_unknown_and_bad_values():
    doc = {"document_id": "DEV-025", "filename": "s.pdf", "file_type": "pdf",
           "language_hint": "en",
           "elements": [
               _cell("u0", 1, 0, 0, 0, "Frobnicate"),
               _cell("u1", 1, 0, 0, 1, "10.00"),
               _cell("v0", 1, 0, 1, 0, "Revenue"),
               _cell("v1", 1, 0, 1, 1, "10.0"),
               _cell("f0", 1, 0, 2, 0, "Revenue"),
               _cell("f1", 1, 0, 2, 1, "10.00 USD"),
           ]}
    env = run_statements(doc, column_periods={1: "2023"})
    assert env.records == []
    assert any("W_LABEL" in w and "Frobnicate" in w for w in env.warnings)
    assert any("foreign-currency" in w for w in env.warnings)


def test_04_config_guards():
    import pytest
    with pytest.raises(TypeError):
        run_statements(_stmt_doc(), column_periods=[(1, "2023")])
    empty = run_statements(_stmt_doc(), column_periods={})
    assert empty.records == [] and empty.ok is True
    assert any("W_CONFIG:empty" in w for w in empty.warnings)
    with pytest.raises(TypeError):
        run_statements(_stmt_doc(), column_periods={1: "2023"}, label_column=True)
    with pytest.raises(ValueError):
        run_statements(_stmt_doc(), column_periods={1: "Q3"})
    for bad_key in (True, "1", -1, 1.5):
        with pytest.raises(TypeError):
            run_statements(_stmt_doc(), column_periods={bad_key: "2023"})
    with pytest.raises(TypeError):
        run_statements("nope", column_periods={1: "2023"})
    env = run_statements({"document_id": "DEV-026", "filename": "s.pdf",
                          "file_type": "pdf", "language_hint": "en",
                          "elements": [{"element_id": "t1", "element_type": "text",
                                        "page": 1, "text": "Revenue 10.00"}]},
                         column_periods={1: "2023"})
    assert env.records == [] and env.ok is True
    assert any("statements_empty" in w or "non-table" in w for w in env.warnings)


def test_05_branch_and_determinism():
    env = run_statements(_stmt_doc(), column_periods={1: "2023"},
                         branch_id="BR-JED")
    assert all(r.branch_id == "BR-JED" for r in env.records)
    assert len(env.records) == 3
    again = run_statements(_stmt_doc(), column_periods={1: "2023"},
                           branch_id="BR-JED")
    assert [r.record_id for r in env.records] == [r.record_id for r in again.records]


def test_06_no_duckdb_or_store_imports():
    source = (ROOT / "src" / "extraction" / "runner_statements.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in ("duckdb", "store")
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "")
            assert "duckdb" not in module and "store" not in module
    assert "duckdb.connect" not in source
    common = (ROOT / "src" / "extraction" / "adapter_common.py").read_text(encoding="utf-8")
    ctree = ast.parse(common)
    for node in ast.walk(ctree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in ("duckdb", "store")
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "")
            assert "duckdb" not in module and "store" not in module
            assert "runner" not in module
    from extraction import adapter_common as _C
    assert _C.map_identity_or_bs("Revenue") == "revenue"
    assert _C.map_identity_or_bs("Nope") is None
    assert _C.derive_period("2023") == ("2023", 2023)
    assert _C.display_for("10.00") == "10.00"
    assert _C.has_foreign_currency("10.00 USD") is True
    assert _C.has_foreign_currency("10.00 SAR") is False
