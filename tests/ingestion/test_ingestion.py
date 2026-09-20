"""Phase-1 native ingestion tests: routing, parsers, GT-anchored extraction,
failure isolation. Ground-truth JSONs are used here for EVALUATION only;
production parser code (src/ingestion) never imports them.
"""

import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ingestion.model import (ElementType, FileType, IngestionStatus,  # noqa: E402
                             LanguageHint, document_id_for)
from ingestion.router import route  # noqa: E402
from ingestion.service import ingest_directory, ingest_document  # noqa: E402

DOCS = ROOT / "data" / "dev" / "dataset_v0.1" / "documents"
GT = ROOT / "data" / "dev" / "dataset_v0.1" / "ground_truth"

FILES = {
    "DEV-001": "DEV-001_income_statement_2023_EN.pdf",
    "DEV-002": "DEV-002_branch_expenses_2022_AR.pdf",
    "DEV-003": "DEV-003_supplier_invoice_2023_MIX.pdf",
    "DEV-004": "DEV-004_scanned_receipt_2019_AR.png",
    "DEV-005": "DEV-005_management_commentary_2023_MIX.docx",
    "DEV-006": "DEV-006_monthly_transactions_2023_EN.xlsx",
    "DEV-007": "DEV-007_transactions_extract_2024_MIX.csv",
    "DEV-008": "DEV-008_revenue_trend_2015_2024_EN.png",
    "DEV-009": "DEV-009_kpi_dashboard_Q4-2023_MIX.jpg",
    "DEV-010": "DEV-010_balance_sheet_2020_EN_scanned.pdf",
}


def gt_record(dev_id):
    return json.loads((GT / f"{dev_id}.json").read_text(encoding="utf-8"))


def full_text(result):
    return "\n".join(e.text or "" for e in result.document.elements)


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------
def test_routing_by_signature_and_extension():
    assert route(DOCS / FILES["DEV-001"]).file_type == FileType.PDF
    assert route(DOCS / FILES["DEV-005"]).file_type == FileType.DOCX
    assert route(DOCS / FILES["DEV-006"]).file_type == FileType.XLSX
    assert route(DOCS / FILES["DEV-007"]).file_type == FileType.CSV
    assert route(DOCS / FILES["DEV-004"]).file_type == FileType.PNG
    assert route(DOCS / FILES["DEV-009"]).file_type == FileType.JPG


def test_signature_wins_over_extension(tmp_path):
    disguised = tmp_path / "renamed.bin"
    disguised.write_bytes((DOCS / FILES["DEV-001"]).read_bytes())
    assert route(disguised).file_type == FileType.PDF


def test_unsupported_format_controlled(tmp_path):
    weird = tmp_path / "notes.xyz"
    weird.write_bytes(b"\x00\x01\x02not a supported format")
    result = ingest_document(weird)
    assert not result.ok
    assert result.document.status == IngestionStatus.UNSUPPORTED


def test_missing_file_controlled(tmp_path):
    result = ingest_document(tmp_path / "absent.pdf")
    assert not result.ok
    assert result.document.status == IngestionStatus.FAILED


# --------------------------------------------------------------------------
# Deterministic IDs
# --------------------------------------------------------------------------
def test_dev_document_ids_stable():
    first = ingest_document(DOCS / FILES["DEV-001"])
    second = ingest_document(DOCS / FILES["DEV-001"])
    assert first.document.document_id == "DEV-001"
    assert second.document.document_id == "DEV-001"


def test_content_addressed_ids_for_general_files(tmp_path):
    copy_a = tmp_path / "report.pdf"
    copy_a.write_bytes((DOCS / FILES["DEV-001"]).read_bytes())
    copy_b = tmp_path / "other-name.pdf"
    copy_b.write_bytes((DOCS / FILES["DEV-001"]).read_bytes())
    assert document_id_for(copy_a) == document_id_for(copy_b)
    assert document_id_for(copy_a).startswith("DOC-")


# --------------------------------------------------------------------------
# PDF native extraction + scanned classification
# --------------------------------------------------------------------------
def test_dev001_native_ok_with_table():
    result = ingest_document(DOCS / FILES["DEV-001"])
    doc = result.document
    assert result.ok and doc.status == IngestionStatus.NATIVE_OK
    assert doc.page_count == 1 and not doc.requires_ocr
    text = full_text(result)
    for anchor in ("Net income", "11,413,708.69", "EVT-2023-001"):
        assert anchor in text
    tables = [e for e in doc.elements
              if e.element_type == ElementType.TABLE]
    assert len(tables) >= 1
    cells = [e for e in doc.elements
             if e.element_type == ElementType.TABLE_CELL]
    assert any("11,413,708.69" in (c.text or "") for c in cells)
    assert doc.language_hint == LanguageHint.EN


def test_dev010_requires_ocr_no_false_success():
    result = ingest_document(DOCS / FILES["DEV-010"])
    doc = result.document
    assert doc.status == IngestionStatus.REQUIRES_OCR
    assert doc.requires_ocr and not doc.requires_visual
    assert result.ok  # classification itself succeeded
    assert not [e for e in doc.elements
                if e.element_type in (ElementType.TEXT, ElementType.TABLE)]


def test_dev002_arabic_preserved():
    import arabic_reshaper
    result = ingest_document(DOCS / FILES["DEV-002"])
    doc = result.document
    assert result.ok and doc.status == IngestionStatus.NATIVE_OK
    assert doc.language_hint == LanguageHint.AR
    text = full_text(result)
    assert "6,472,903.83" in text and "EVT-2022-001" in text
    for word in ("تقرير", "المصروفات", "جدة"):
        assert arabic_reshaper.reshape(word) in text


def test_dev003_invoice_anchors():
    result = ingest_document(DOCS / FILES["DEV-003"])
    doc = result.document
    assert result.ok and doc.status == IngestionStatus.NATIVE_OK
    assert doc.language_hint == LanguageHint.MIXED
    text = full_text(result)
    assert "INV-2023-0106" in text and "82,209.85" in text


# --------------------------------------------------------------------------
# DOCX / XLSX / CSV
# --------------------------------------------------------------------------
def test_dev005_docx_paragraphs_and_rtl():
    result = ingest_document(DOCS / FILES["DEV-005"])
    doc = result.document
    assert result.ok and doc.status == IngestionStatus.NATIVE_OK
    paras = [e for e in doc.elements
             if e.element_type == ElementType.TEXT]
    assert len(paras) == 11  # all Normal style; headings only if styled so
    assert sum(1 for e in doc.elements if e.meta.get("rtl")) == 5
    text = full_text(result)
    assert "EVT-2023-001" in text and "تعليق الإدارة" in text
    assert doc.language_hint == LanguageHint.MIXED


def test_dev006_xlsx_native_cells():
    result = ingest_document(DOCS / FILES["DEV-006"])
    doc = result.document
    assert result.ok and doc.status == IngestionStatus.NATIVE_OK
    assert doc.sheet_count == 1
    assert doc.metadata["sheets"][0]["name"] == "transactions_2023"
    assert doc.metadata["data_cells"] > 40000
    first = next(e for e in doc.elements
                 if e.sheet == "transactions_2023" and e.row == 2)
    assert first.column == 1
    amounts = [e for e in doc.elements
               if e.sheet == "transactions_2023" and e.column == 9
               and isinstance(e.value, float)]
    assert amounts, "numeric amounts must stay native numbers"
    assert all(isinstance(e.value, float) for e in amounts)


def test_dev007_csv_rows_and_arabic():
    result = ingest_document(DOCS / FILES["DEV-007"])
    doc = result.document
    assert result.ok and doc.status == IngestionStatus.NATIVE_OK
    rows = [e for e in doc.elements
            if e.element_type == ElementType.CSV_ROW]
    assert len(rows) == 346
    assert rows[0].row == 1 and rows[-1].row == 346
    assert "description_ar" in rows[0].value
    assert any("مع" in (v or "") for e in rows
               for v in e.value.values())


def test_images_metadata_only_and_flagged():
    expectations = {"DEV-004": (1038, 1288), "DEV-008": (1600, 1000),
                    "DEV-009": (1600, 1000)}
    for dev_id, (width, height) in expectations.items():
        result = ingest_document(DOCS / FILES[dev_id])
        doc = result.document
        assert doc.status == IngestionStatus.REQUIRES_VISUAL, dev_id
        assert doc.requires_visual and not doc.requires_ocr, dev_id
        assert result.ok, dev_id  # detection succeeded; nothing to extract
        assert doc.metadata["width"] == width, dev_id
        assert doc.metadata["height"] == height, dev_id
        assert doc.language_hint == LanguageHint.UNKNOWN, dev_id


# --------------------------------------------------------------------------
# Provenance, serialization, directory ingestion, isolation
# --------------------------------------------------------------------------
def test_source_locations_present():
    result = ingest_document(DOCS / FILES["DEV-001"])
    for element in result.document.elements:
        assert element.element_id.startswith("DEV-001:"), element.element_id
        if element.element_type == ElementType.TABLE_CELL:
            assert element.page == 1
            assert element.table_index is not None
            assert element.row is not None and element.column is not None
    xlsx = ingest_document(DOCS / FILES["DEV-006"])
    cell = next(e for e in xlsx.document.elements
                if e.element_type == ElementType.SHEET_CELL)
    assert cell.sheet and cell.row and cell.column


def test_serialization_roundtrip(tmp_path):
    result = ingest_document(DOCS / FILES["DEV-003"])
    payload = result.document.to_json()
    parsed = json.loads(payload)
    assert parsed["document_id"] == "DEV-003"
    assert parsed["status"] == "native_ok"
    assert parsed["elements"], "must serialize elements, not just metadata"
    out = tmp_path / "dev003.json"
    out.write_text(payload, encoding="utf-8")
    assert json.loads(out.read_text(encoding="utf-8")) == parsed


def test_directory_ingestion_skips_ground_truth():
    report = ingest_directory(DOCS.parent)  # dataset_v0.1 root
    summary = report["summary"]
    assert summary["ingested"] == 10 and not summary["failed"]
    assert summary["files_found"] == 21  # 10 docs + 10 GT JSONs + manifest
    for dev_id in FILES:
        assert f"{dev_id}.json" in summary["skipped_sidecars"]
    assert "manifest.json" in summary["skipped_sidecars"]
    assert summary["by_status"]["native_ok"] == 6
    assert summary["by_status"]["requires_visual"] == 3
    assert summary["by_status"]["requires_ocr"] == 1
    ids = [r.document.document_id for r in report["results"]]
    assert ids == sorted(ids), "deterministic directory ordering"


def test_failure_isolation_in_directory(tmp_path):
    good = tmp_path / FILES["DEV-007"]
    good.write_bytes((DOCS / FILES["DEV-007"]).read_bytes())
    (tmp_path / "corrupt.pdf").write_bytes(b"not a pdf at all")
    (tmp_path / "notes.xyz").write_bytes(b"\x00\x01junk")
    report = ingest_directory(tmp_path)
    assert report["summary"]["ingested"] == 2  # csv + corrupt(pdf->FAILED)
    assert report["summary"]["failed"] == ["corrupt.pdf"]
    ok_ids = [r.document.document_id for r in report["results"] if r.ok]
    assert len(ok_ids) == 1


# --------------------------------------------------------------------------
# Ground-truth-anchored checks (evaluation side; never in parser code)
# --------------------------------------------------------------------------
def test_gt_dev001_net_income_recoverable():
    expected = gt_record("DEV-001")["expected_numeric_values"][
        "fy2023_net_income"]
    assert f"{Decimal(expected):,.2f}" in full_text(
        ingest_document(DOCS / FILES["DEV-001"]))


def test_gt_dev003_invoice_total_recoverable():
    expected = gt_record("DEV-003")["expected_numeric_values"]["total"]
    text = full_text(ingest_document(DOCS / FILES["DEV-003"]))
    assert "INV-2023-0106" in text
    assert f"{Decimal(expected):,.2f}" in text


def test_gt_dev006_row_count_and_value():
    expected_rows = gt_record("DEV-006")["expected_numeric_values"][
        "row_count"]
    result = ingest_document(DOCS / FILES["DEV-006"])
    data_rows = result.document.metadata["sheets"][0]["max_row"] - 1
    assert str(data_rows) == expected_rows == "4020"
    first_id = gt_record("DEV-006")["expected_identifiers"][
        "first_transaction_id"]
    assert any(e.text == first_id or
               (e.sheet and e.column == 1 and e.value == first_id)
               for e in result.document.elements)


def test_gt_dev007_total_recoverable():
    expected = gt_record("DEV-007")["expected_numeric_values"][
        "total_amount_sum"]
    result = ingest_document(DOCS / FILES["DEV-007"])
    total = sum((Decimal(e.value["total_amount"]) for e in
                 result.document.elements
                 if e.element_type == ElementType.CSV_ROW),
                Decimal("0.00"))
    assert total == Decimal(expected)


def test_no_ocr_or_ai_imports_in_production_code():
    banned = ("paddleocr", "pytesseract", "transformers", "torch", "qdrant",
              "langchain", "llama_index", "rank_bm25", "sentence_transformers",
              "sklearn", "streamlit")
    for module in ("model", "router", "parsers", "service"):
        source = (ROOT / "src" / "ingestion" / f"{module}.py").read_text(
            encoding="utf-8")
        for name in banned:
            assert name not in source, f"{module}.py references {name}"
