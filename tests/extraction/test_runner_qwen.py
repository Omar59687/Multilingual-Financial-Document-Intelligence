"""Qwen visual-pairs adapter tests (bounded slice, deterministic).

Synthetic fixtures only: every numeric below is synthetic. No benchmark
or ground-truth reads, no network, no duckdb, no randomness.
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from extraction.runner_qwen import run_qwen


def _pairs_result():
    return {
        "document_id": "DEV-009",
        "model": "qwen3-vl",
        "fields": {
            "Revenue": "23.90M SAR",
            "Cash": "100.00",
            "Verdict": "target met",
        },
    }


def test_01_pairs_emit_in_map_order_with_rounding():
    env = run_qwen(_pairs_result(), document_period="2023-Q4", page=1)
    assert env.ok is True
    assert [r.metric.value for r in env.records] == ["revenue", "cash"]
    rev, cash = env.records
    assert rev.precision == "display-rounded" and str(rev.value) == "23900000.00"
    assert rev.period == "2023-Q4" and rev.fiscal_year == 2023
    assert cash.precision == "exact-visible" and str(cash.value) == "100.00"
    assert all(r.provenance.extraction_route == "qwen-visual" for r in env.records)
    assert all(r.provenance.element_id is None for r in env.records)
    assert any("W_LABEL" in w and "Verdict" in w for w in env.warnings)


def test_02_coarse_suffix_skips_never_fabricates():
    result = {"document_id": "DEV-018", "fields": {"Revenue": "24M SAR"}}
    env = run_qwen(result, document_period="2023")
    assert env.records == []
    assert any("W_RECORD" in w or "W_VALUE" in w or "W_VALIDATE" in w
               for w in env.warnings)


def test_03_empty_fields_and_tables_note():
    env = run_qwen({"document_id": "DEV-008", "fields": {}},
                   document_period="2023")
    assert env.records == [] and env.ok is True
    assert any("qwen_empty" in w for w in env.warnings)
    noted = run_qwen({"document_id": "DEV-009", "fields": {"Cash": "10.00"},
                      "tables": [[["LABEL", "x"]]]},
                     document_period="2023-Q4")
    assert len(noted.records) == 1
    assert any("qwen_tables_ignored" in w for w in noted.warnings)


def test_04_bad_values_and_guards():
    import pytest
    result = {"document_id": "DEV-019",
              "fields": {"Cash": "10.0", "Debt": True, "Equity": "nope"}}
    env = run_qwen(result, document_period="2021")
    assert env.records == []
    for bad in ("x", None, 123, ["x"]):
        with pytest.raises(TypeError):
            run_qwen(bad, document_period="2020")
    with pytest.raises(TypeError):
        run_qwen({"document_id": "DEV-009"}, document_period="2020")
    with pytest.raises(TypeError):
        run_qwen({"document_id": "DEV-009", "fields": {}})
    with pytest.raises(TypeError):
        run_qwen({"document_id": "DEV-009", "fields": {}},
                 document_period="")
    with pytest.raises(ValueError):
        run_qwen({"document_id": "DEV-009", "fields": {}},
                 document_period="Q3")


def test_05_branch_and_determinism():
    result = {"document_id": "DEV-020", "fields": {"Revenue": "10.00"}}
    env = run_qwen(result, document_period="2022", branch_id="BR-RUH")
    assert env.records[0].branch_id == "BR-RUH"
    again = run_qwen(result, document_period="2022", branch_id="BR-RUH")
    assert [r.record_id for r in env.records] == [r.record_id for r in again.records]
    bad = run_qwen(result, document_period="2022", branch_id="BR-XX")
    assert bad.records == []


def test_06_foreign_currency_never_becomes_sar():
    foreign = {"document_id": "DEV-021",
               "fields": {"Revenue": "23.90M USD", "Cash": "$100.00",
                          "Debt": "10.00 AED", "Equity": "30.00 SAR"}}
    env = run_qwen(foreign, document_period="2023")
    assert [r.metric.value for r in env.records] == ["equity"]
    assert sum("foreign-currency" in w for w in env.warnings) == 3


def test_07_bs_alias_and_cross_adapter_fin_id():
    env = run_qwen({"document_id": "DEV-022",
                    "fields": {"Property and equipment (net)": "50.00"}},
                   document_period="2020")
    assert len(env.records) == 1
    assert env.records[0].metric.value == "property_plant_equipment"
    from extraction.runner_paddle import run_paddle
    qwen_rec = run_qwen({"document_id": "DEV-023", "fields": {"Cash": "10.00"}},
                        document_period="2021").records[0]
    paddle_rec = run_paddle({"document_id": "DEV-023",
                             "tables": [[["Cash", "10.00"]]]},
                            document_period="2021").records[0]
    assert qwen_rec.record_id == paddle_rec.record_id


def test_08_no_duckdb_or_store_or_qwen_imports():
    source = (ROOT / "src" / "extraction" / "runner_qwen.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in ("duckdb", "store", "transformers")
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "")
            assert "duckdb" not in module and "store" not in module
            assert "transformers" not in module and "ocrbench" not in module
            assert "qwen_adapter" not in module
    assert "duckdb.connect" not in source
