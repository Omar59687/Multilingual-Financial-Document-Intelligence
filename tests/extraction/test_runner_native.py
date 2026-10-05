"""Native-only runner tests (bounded slice, deterministic).

Synthetic fixtures only: every numeric below (10.00/20.00/30.00-style) is
synthetic. No benchmark expected_numeric_values, no ground-truth reads, no
network, no duckdb, no randomness, no wall-clock assertions beyond
finiteness.
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import extraction.runner_native as RN
from extraction.runner_native import run_native


def _csv_doc():
    return {
        "document_id": "DEV-001",
        "filename": "synthetic.csv",
        "file_type": "csv",
        "language_hint": "en",
        "elements": [
            {"element_id": "DEV-001:row:1", "element_type": "csv_row", "row": 1,
             "value": {"transaction_id": "TX-SYN-001", "date": "2023-04-15",
                       "branch_id": "BR-RUH", "department_id": "DEP-LOG",
                       "amount": "10.00", "tax_amount": "1.50",
                       "total_amount": "11.50", "vendor_customer": "Syn Vendor"},
             "data_type": "str", "meta": {}},
            {"element_id": "DEV-001:row:2", "element_type": "csv_row", "row": 2,
             "value": {"transaction_id": "TX-SYN-002", "date": "2023-04-16",
                       "branch_id": "BR-JED", "amount": "20.00",
                       "tax_amount": "3.00", "total_amount": "23.00"},
             "data_type": "str", "meta": {}},
        ],
    }


def _xlsx_doc():
    def cell(sheet, row, col, text=None, value=None):
        return {"element_id": f"DEV-002:{sheet}:r{row}c{col}",
                "element_type": "sheet_cell", "sheet": sheet,
                "row": row, "column": col, "text": text, "value": value,
                "data_type": "s" if text is not None else "n", "meta": {}}

    return {
        "document_id": "DEV-002",
        "filename": "synthetic.xlsx",
        "file_type": "xlsx",
        "language_hint": "en",
        "elements": [
            cell("tx", 1, 1, text="date"), cell("tx", 1, 2, text="branch_id"),
            cell("tx", 1, 3, text="amount"), cell("tx", 1, 4, text="tax_amount"),
            cell("tx", 1, 5, text="total_amount"),
            cell("tx", 2, 1, text="2023-05-01"), cell("tx", 2, 2, text="BR-RUH"),
            cell("tx", 2, 3, value=30.27), cell("tx", 2, 4, value=4.53),
            cell("tx", 2, 5, value=34.81),
            cell("tx", 3, 1, text="2023-05-02"), cell("tx", 3, 2, text="BR-JED"),
            cell("tx", 3, 3, value=40.13), cell("tx", 3, 4, value=6.07),
            cell("tx", 3, 5, value=46.21),
        ],
    }


def test_01_csv_rows_emit_ordered_validated_records():
    env = run_native(_csv_doc())
    assert env.ok is True
    assert env.document_id == "DEV-001"
    assert len(env.records) == 6
    first = env.records[0]
    assert first.metric.value == "amount"
    assert first.period == "2023-04-15" and first.fiscal_year == 2023
    assert first.branch_id == "BR-RUH" and first.department_id == "DEP-LOG"
    assert first.currency == "SAR"
    assert first.provenance.extraction_route == "native"
    assert first.provenance.element_id == "DEV-001:row:1"
    assert str(first.value) == "10.00"
    # Input order preserved: row1 amount/tax/total, then row2.
    assert [ (r.metric.value, str(r.value)) for r in env.records[:3]] == [
        ("amount", "10.00"), ("tax_amount", "1.50"), ("total_amount", "11.50")]
    assert env.latency_ms >= 0 and env.latency_ms == env.latency_ms


def test_02_determinism_same_ids_twice():
    a = run_native(_csv_doc())
    b = run_native(_csv_doc())
    assert [r.record_id for r in a.records] == [r.record_id for r in b.records]
    assert all(r.record_id.startswith("FIN-") for r in a.records)


def test_03_xlsx_row_groups():
    env = run_native(_xlsx_doc())
    assert env.ok is True
    assert len(env.records) == 6
    assert env.records[0].metric.value == "amount"
    assert str(env.records[0].value) == "30.27"
    assert env.records[0].provenance.sheet == "tx"
    assert env.records[0].provenance.row == 2


def test_04_bs_alias_maps():
    doc = {"document_id": "DEV-003", "filename": "s.csv", "file_type": "csv",
           "language_hint": "en",
           "elements": [{"element_id": "DEV-003:row:1", "element_type": "csv_row",
                         "row": 1, "value": {"date": "2020-01-01",
                                             "property_and_equipment": "50.00"}}]}
    env = run_native(doc)
    assert len(env.records) == 1
    assert env.records[0].metric.value == "property_plant_equipment"


def test_05_unknown_labels_warn_without_inventing():
    doc = {"document_id": "DEV-004", "filename": "s.csv", "file_type": "csv",
           "language_hint": "en",
           "elements": [{"element_id": "e1", "element_type": "csv_row", "row": 1,
                         "value": {"date": "2023-01-01", "frobnicate": "10.00",
                                   "amount": "10.00"}}]}
    env = run_native(doc)
    assert len(env.records) == 1
    assert env.records[0].metric.value == "amount"
    assert any("W_LABEL" in w and "frobnicate" in w for w in env.warnings)
    # Structural columns stay silent: no warnings for date/IDs.
    assert not any("transaction_id" in w for w in env.warnings)


def test_06_missing_date_skips_row_with_warning():
    doc = {"document_id": "DEV-005", "filename": "s.csv", "file_type": "csv",
           "language_hint": "en",
           "elements": [{"element_id": "e1", "element_type": "csv_row", "row": 1,
                         "value": {"amount": "10.00"}}]}
    env = run_native(doc)
    assert env.records == [] and env.ok is True
    assert any("W_PERIOD" in w for w in env.warnings)


def test_07_bad_values_never_become_zero():
    doc = {"document_id": "DEV-006", "filename": "s.csv", "file_type": "csv",
           "language_hint": "en",
           "elements": [
               {"element_id": "e1", "element_type": "csv_row", "row": 1,
                "value": {"date": "2023-01-01", "amount": "10.0"}},
               {"element_id": "e2", "element_type": "csv_row", "row": 2,
                "value": {"date": "2023-01-02", "amount": True}},
               {"element_id": "e3", "element_type": "csv_row", "row": 3,
                "value": {"date": "2023-01-03", "amount": "not-a-number"}},
           ]}
    env = run_native(doc)
    assert env.records == []
    assert any("W_VALUE" in w or "W_RECORD" in w or "W_VALIDATE" in w
               for w in env.warnings)


def test_08_invalid_branch_and_dammam_gate_skip():
    doc = {"document_id": "DEV-007", "filename": "s.csv", "file_type": "csv",
           "language_hint": "en",
           "elements": [
               {"element_id": "e1", "element_type": "csv_row", "row": 1,
                "value": {"date": "2023-01-01", "branch_id": "BR-XX",
                          "amount": "10.00"}},
               {"element_id": "e2", "element_type": "csv_row", "row": 2,
                "value": {"date": "2018-12-01", "branch_id": "BR-DMM",
                          "amount": "10.00"}},
           ]}
    env = run_native(doc)
    assert env.records == []
    assert any("W_RECORD" in w or "W_VALIDATE" in w for w in env.warnings)


def test_09_unsupported_pdf_elements_deferred():
    doc = {"document_id": "DEV-008", "filename": "s.pdf", "file_type": "pdf",
           "language_hint": "en",
           "elements": [
               {"element_id": "e1", "element_type": "text", "page": 1,
                "text": "Revenue 10.00"},
               {"element_id": "e2", "element_type": "table_cell", "page": 1,
                "row": 0, "column": 0, "text": "Revenue"},
           ]}
    env = run_native(doc)
    assert env.records == [] and env.ok is True
    assert any("native_deferred" in w for w in env.warnings)


def test_10_input_type_guards():
    import pytest
    for bad in ("DEV-001", None, 123, ["x"], {"nope": 1}):
        with pytest.raises(TypeError):
            run_native(bad)
    with pytest.raises(TypeError):
        run_native({"document_id": "DEV-001", "elements": "nope"})


def test_12_sheet_without_row1_skipped_fail_closed():
    els = [
        {"element_id": "DEV-009:s:r2c1", "element_type": "sheet_cell",
         "sheet": "s", "row": 2, "column": 1, "text": "date",
         "value": "date", "data_type": "s", "meta": {}},
        {"element_id": "DEV-009:s:r2c2", "element_type": "sheet_cell",
         "sheet": "s", "row": 2, "column": 2, "text": "10.00",
         "value": "10.00", "data_type": "s", "meta": {}},
    ]
    doc = {"document_id": "DEV-009", "filename": "s.xlsx", "file_type": "xlsx",
           "language_hint": "en", "elements": els}
    env = run_native(doc)
    assert env.records == [] and env.ok is True
    assert any("header row 1" in w for w in env.warnings)


def test_13_sheets_emit_in_first_seen_order():
    def cell(sheet, row, col, text=None, value=None):
        return {"element_id": f"DEV-010:{sheet}:r{row}c{col}",
                "element_type": "sheet_cell", "sheet": sheet,
                "row": row, "column": col, "text": text, "value": value,
                "data_type": "s" if text is not None else "n", "meta": {}}
    # "zeta" sheet appears first in input though sorting would put "alpha" first.
    els = [
        cell("zeta", 1, 1, text="date"), cell("zeta", 1, 2, text="amount"),
        cell("zeta", 2, 1, text="2023-06-01"), cell("zeta", 2, 2, text="10.01"),
        cell("alpha", 1, 1, text="date"), cell("alpha", 1, 2, text="amount"),
        cell("alpha", 2, 1, text="2023-06-02"), cell("alpha", 2, 2, text="20.02"),
    ]
    doc = {"document_id": "DEV-010", "filename": "s.xlsx", "file_type": "xlsx",
           "language_hint": "en", "elements": els}
    env = run_native(doc)
    assert [str(r.value) for r in env.records] == ["10.01", "20.02"]
    assert env.records[0].provenance.sheet == "zeta"


def test_14_inner_spacing_preserved_in_provenance():
    doc = {"document_id": "DEV-011", "filename": "s.csv", "file_type": "csv",
           "language_hint": "en",
           "elements": [{"element_id": "e1", "element_type": "csv_row", "row": 1,
                         "value": {"date": "2023-01-01", "amount": "1 0.00"}}]}
    env = run_native(doc)
    assert len(env.records) == 1
    assert env.records[0].display_value == "1 0.00"
    assert str(env.records[0].value) == "10.00"


def test_15_date_key_priority_date_over_period():
    doc = {"document_id": "DEV-012", "filename": "s.csv", "file_type": "csv",
           "language_hint": "en",
           "elements": [{"element_id": "e1", "element_type": "csv_row", "row": 1,
                         "value": {"date": "2023-03-05", "period": "2021",
                                   "amount": "10.00"}}]}
    env = run_native(doc)
    assert len(env.records) == 1
    assert env.records[0].period == "2023-03-05"


def test_17_padded_header_still_maps():
    doc = {"document_id": "DEV-013", "filename": "s.csv", "file_type": "csv",
           "language_hint": "en",
           "elements": [{"element_id": "e1", "element_type": "csv_row", "row": 1,
                         "value": {"date": "2023-01-01", "  amount  ": "10.00"}}]}
    env = run_native(doc)
    assert len(env.records) == 1
    assert env.records[0].metric.value == "amount"


def test_18_undated_row_still_reports_unknown_label():
    doc = {"document_id": "DEV-014", "filename": "s.csv", "file_type": "csv",
           "language_hint": "en",
           "elements": [{"element_id": "e1", "element_type": "csv_row", "row": 1,
                         "value": {"frobnicate": "10.00"}}]}
    env = run_native(doc)
    assert env.records == []
    assert any("W_PERIOD" in w for w in env.warnings)
    assert any("W_LABEL" in w and "frobnicate" in w for w in env.warnings)


def test_19_unknown_label_order_deterministic():
    doc = {"document_id": "DEV-015", "filename": "s.csv", "file_type": "csv",
           "language_hint": "en",
           "elements": [{"element_id": "e1", "element_type": "csv_row", "row": 1,
                         "value": {"date": "2023-01-01", "foo": "1.00",
                                   "Foo": "2.00", "bars": "3.00"}}]}
    first = [w for w in run_native(doc).warnings if "W_LABEL" in w]
    second = [w for w in run_native(doc).warnings if "W_LABEL" in w]
    assert first == second
    assert [w for w in first if "'Foo'" in w] < [w for w in first if "'foo'" in w]


def test_16_no_duckdb_or_store_imports():
    source = (ROOT / "src" / "extraction" / "runner_native.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in ("duckdb", "store")
                assert "store" not in alias.name
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "")
            assert "duckdb" not in module and "store" not in module
    assert "duckdb.connect" not in source


def test_17_foreign_currency_never_becomes_sar():
    doc = {"document_id": "DEV-027", "filename": "s.csv", "file_type": "csv",
           "language_hint": "en",
           "elements": [{"element_id": "e1", "element_type": "csv_row", "row": 1,
                         "value": {"date": "2023-01-01", "amount": "10.00 USD",
                                   "tax_amount": "1.00", "total_amount": "11.00 SAR"}}]}
    env = run_native(doc)
    assert [r.metric.value for r in env.records] == ["tax_amount", "total_amount"]
    assert any("foreign-currency" in w for w in env.warnings)
