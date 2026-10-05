"""Paddle scanned-grid adapter tests (bounded slice, deterministic).

Synthetic fixtures only: every numeric below is synthetic. No benchmark
or ground-truth reads, no network, no duckdb, no randomness.
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from extraction.runner_paddle import run_paddle


def _bs_result():
    return {
        "document_id": "DEV-010",
        "model": "paddleocr-vl",
        "tables": [[
            ["ASSETS"],
            ["Cash", "100.00"],
            ["Accounts receivable", "50.00"],
            ["TOTAL ASSETS", "150.00"],
            ["LIABILITIES"],
            ["Debt", "60.00"],
            ["TOTAL LIABILITIES", "60.00"],
            ["EQUITY", "90.00"],
        ]],
    }


def test_01_pairs_emit_validated_records_in_order():
    env = run_paddle(_bs_result(), document_period="2020", page=1,
                     filename="synthetic.pdf", file_type="pdf")
    assert env.ok is True
    assert env.document_id == "DEV-010"
    assert len(env.records) == 6
    assert [r.metric.value for r in env.records] == [
        "cash", "accounts_receivable", "total_assets",
        "debt", "total_liabilities", "equity"]
    first = env.records[0]
    assert first.period == "2020" and first.fiscal_year == 2020
    assert first.branch_id is None and first.currency == "SAR"
    assert first.provenance.extraction_route == "paddle"
    assert first.provenance.table_index == 0 and first.provenance.row == 1
    assert first.page == 1
    assert str(first.value) == "100.00"


def test_02_structural_rows_silently_skipped():
    env = run_paddle(_bs_result(), document_period="2020")
    assert env.ok is True
    assert not any("ASSETS" in w and "W_LABEL" in w for w in env.warnings)
    assert env.records[0].provenance.page is None


def test_03_arabic_aliases_map():
    result = {"document_id": "DEV-004", "tables": [[
        ["المبلغ", "24.00"],
        ["الضريبة 15 %", "3.00"],
        ["الاجمالي", "27.00"],
        ["ختم الفرع"],
    ]]}
    env = run_paddle(result, document_period="2019-06-09", branch_id="BR-JED")
    assert env.ok is True
    assert [r.metric.value for r in env.records] == [
        "amount", "tax_amount", "total_amount"]
    assert all(r.branch_id == "BR-JED" for r in env.records)
    assert all(r.period == "2019-06-09" for r in env.records)


def test_04_unknown_labels_warn_without_inventing():
    result = {"document_id": "DEV-011", "tables": [[["Frobnicate", "10.00"]]]}
    env = run_paddle(result, document_period="2021")
    assert env.records == []
    assert any("W_LABEL" in w and "Frobnicate" in w for w in env.warnings)


def test_05_bad_values_never_become_zero():
    result = {"document_id": "DEV-012", "tables": [[
        ["Cash", "10.0"], ["Debt", True], ["Equity", "nope"]]]}
    env = run_paddle(result, document_period="2021")
    assert env.records == []
    assert any("W_VALUE" in w or "W_RECORD" in w for w in env.warnings)


def test_06_period_and_branch_guards():
    import pytest
    with pytest.raises(TypeError):
        run_paddle(_bs_result())
    with pytest.raises(TypeError):
        run_paddle(_bs_result(), document_period="")
    import pytest as _p
    with _p.raises(ValueError):
        run_paddle(_bs_result(), document_period="not-a-period")
    env = run_paddle(_bs_result(), document_period="2020", branch_id="BR-XX")
    assert env.records == []
    assert any("W_RECORD" in w or "W_VALIDATE" in w for w in env.warnings)


def test_07_input_type_guards_and_row_shapes():
    import pytest
    for bad in ("x", None, 123, ["x"]):
        with pytest.raises(TypeError):
            run_paddle(bad, document_period="2020")
    with pytest.raises(TypeError):
        run_paddle({"nope": 1}, document_period="2020")
    with pytest.raises(TypeError):
        run_paddle({"document_id": "DEV-010", "tables": "nope"},
                   document_period="2020")
    env = run_paddle({"document_id": "DEV-010", "tables": [[["a", "b", "c"]]]},
                     document_period="2020")
    assert env.records == [] and env.ok is True
    assert any("3 cells" in w for w in env.warnings)
    with pytest.raises(TypeError):
        run_paddle({"document_id": "DEV-010", "tables": (["Cash", "10.00"],)},
                   document_period="2020")
    env = run_paddle({"document_id": "DEV-010",
                      "tables": [("Cash", "11.00"), {("Debt", "22.00")}]},
                     document_period="2020")
    assert env.records == [] and env.ok is True
    assert sum("not a grid list" in w for w in env.warnings) == 2
    nested = run_paddle({"document_id": "DEV-010",
                         "tables": [[("Cash", "10.00"), ["Debt", "20.00"]]]},
                        document_period="2020")
    assert [r.metric.value for r in nested.records] == ["debt"]
    assert any("not a cell list" in w for w in nested.warnings)


def test_08_determinism_and_identical_pair_dedup():
    result = {"document_id": "DEV-013", "tables": [
        [["Cash", "10.00"]], [["Cash", "10.00"]]]}
    a = run_paddle(result, document_period="2021")
    b = run_paddle(result, document_period="2021")
    assert [r.record_id for r in a.records] == [r.record_id for r in b.records]
    assert a.records[0].record_id == a.records[1].record_id


def test_10_slash_currency_empty_and_provenance():
    result = {"document_id": "DEV-016", "tables": [[
        ["Cash", "1 0.00 SAR"], ["Frobnicate", "1.00"], ["Frobnicate", "2.00"]]]}
    env = run_paddle(result, document_period="01/06/2019",
                     filename="synthetic.pdf")
    assert env.ok is True
    assert len(env.records) == 1
    assert env.records[0].period == "2019-06-01"
    assert env.records[0].precision == "exact-visible"
    assert env.records[0].provenance.element_id is None
    assert env.records[0].provenance.filename == "synthetic.pdf"
    assert env.latency_ms >= 0 and env.latency_ms == env.latency_ms
    labels = [w for w in env.warnings if "W_LABEL" in w]
    assert len(labels) == 1 and "Frobnicate" in labels[0]
    empty = run_paddle({"document_id": "DEV-017", "tables": []},
                       document_period="2020")
    assert empty.records == [] and empty.ok is True
    assert any("paddle_empty" in w for w in empty.warnings)


def test_09_no_duckdb_or_store_or_paddle_imports():
    source = (ROOT / "src" / "extraction" / "runner_paddle.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in ("duckdb", "store", "paddleocr", "paddle")
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "")
            assert "duckdb" not in module and "store" not in module
            assert "paddle" not in module and "ocrbench" not in module
    assert "duckdb.connect" not in source


def test_11_foreign_currency_never_becomes_sar():
    result = {"document_id": "DEV-028", "tables": [[
        ["Cash", "10.00 USD"], ["Debt", "20.00 SAR"]]]}
    env = run_paddle(result, document_period="2021")
    assert [r.metric.value for r in env.records] == ["debt"]
    assert any("foreign-currency" in w for w in env.warnings)
