"""Phase-4 R1 tests: deterministic chunking, normalizer, index build.

Per docs/RETRIEVAL_DESIGN.md sections 2, 3, 4, 8. Uses synthetic
ingestion.model Documents plus small real DEV files copied to tmp dirs
(never ground_truth/ or canonical/ content).
"""

import ast
import json
import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ingestion.model import (Document, Element, ElementType, FileType,  # noqa: E402
                             LanguageHint)
from retrieval.chunking import (build_index, chunk_corpus,  # noqa: E402
                                chunk_document, chunk_id_for,
                                make_chunk_record, normalize_text,
                                ocr_fingerprint_for)

DOCS = ROOT / "data" / "dev" / "dataset_v0.1" / "documents"

RECORD_KEYS = {"chunk_id", "document_id", "kind", "index", "text",
               "filename", "file_type", "language", "page", "sheet",
               "section", "element_ids", "spans", "length"}


def make_el(eid, etype, text=None, value=None, page=None, sheet=None,
            row=None, column=None, table_index=None):
    return Element(element_id=eid, element_type=etype, text=text,
                   value=value, page=page, sheet=sheet, row=row,
                   column=column, table_index=table_index)


def make_doc(doc_id="DEV-T01", elements=(), filename="t.pdf",
             file_type=FileType.PDF, lang=LanguageHint.EN):
    return Document(document_id=doc_id, filename=filename,
                    file_type=file_type, language_hint=lang,
                    elements=list(elements))


# ---------------------------------------------------------------------------
# TEXT / HEADING units + section rule
# ---------------------------------------------------------------------------
def test_text_one_chunk_verbatim():
    doc = make_doc(elements=[make_el("e1", ElementType.TEXT, text="  keep  spaces ")])
    (chunk,) = chunk_document(doc)
    assert chunk["kind"] == "text"
    assert chunk["text"] == "  keep  spaces "
    assert chunk["element_ids"] == ["e1"]
    assert chunk["spans"] == [{"element_id": "e1", "start": 0, "end": 15}]
    assert chunk["length"] == 15 == len(chunk["text"])
    assert chunk["section"] is None  # no preceding heading
    assert set(chunk) == RECORD_KEYS


def test_heading_one_chunk_section_none():
    doc = make_doc(elements=[make_el("h1", ElementType.HEADING, text="Results")])
    (chunk,) = chunk_document(doc)
    assert chunk["kind"] == "heading"
    assert chunk["text"] == "Results"
    assert chunk["section"] is None


def test_heading_section_attach_nearest_preceding():
    doc = make_doc(elements=[
        make_el("t0", ElementType.TEXT, text="lead"),
        make_el("h1", ElementType.HEADING, text="First"),
        make_el("t1", ElementType.TEXT, text="under first"),
        make_el("h2", ElementType.HEADING, text="Second"),
        make_el("t2", ElementType.TEXT, text="under second"),
    ])
    chunks = chunk_document(doc)
    assert [c["section"] for c in chunks] == [None, None, "First", None, "Second"]
    assert [c["index"] for c in chunks] == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# TABLE_CELL grouping
# ---------------------------------------------------------------------------
def _table_doc():
    return make_doc(doc_id="DEV-T02", elements=[
        make_el("tbl", ElementType.TABLE, page=2, table_index=0),
        make_el("r", ElementType.TABLE_ROW, page=2, table_index=0, row=0),
        make_el("c00", ElementType.TABLE_CELL, text="ab", page=2, row=0, column=0, table_index=0),
        make_el("c01", ElementType.TABLE_CELL, text="cd", page=2, row=0, column=1, table_index=0),
        make_el("c10", ElementType.TABLE_CELL, text="ab", page=2, row=1, column=0, table_index=0),
        make_el("c11", ElementType.TABLE_CELL, text="", page=2, row=1, column=1, table_index=0),
    ])


def test_table_grouping_and_cell_order():
    (chunk,) = chunk_document(_table_doc())
    assert chunk["kind"] == "table"
    assert chunk["text"] == "ab | cd | ab | "
    assert chunk["page"] == 2
    assert chunk["sheet"] is None
    assert chunk["element_ids"] == ["c00", "c01", "c10", "c11"]
    # Arithmetic spans: repeated "ab" resolves positionally, not via find.
    assert chunk["spans"] == [
        {"element_id": "c00", "start": 0, "end": 2},
        {"element_id": "c01", "start": 5, "end": 7},
        {"element_id": "c10", "start": 10, "end": 12},
        {"element_id": "c11", "start": 15, "end": 15},
    ]


def test_table_group_key_splits_page_and_sheet():
    doc = make_doc(doc_id="DEV-T03", elements=[
        make_el("a", ElementType.TABLE_CELL, text="p1", page=1, row=0, column=0, table_index=0),
        make_el("b", ElementType.TABLE_CELL, text="p2", page=2, row=0, column=0, table_index=0),
        make_el("c", ElementType.TABLE_CELL, text="s", page=None, row=0, column=0,
                table_index=0, sheet="S1"),
        make_el("d", ElementType.TABLE_CELL, text="t", page=None, row=0, column=0,
                table_index=0, sheet="S2"),
    ])
    chunks = chunk_document(doc)
    assert [(c["text"], c["page"], c["sheet"]) for c in chunks] == [
        ("p1", 1, None), ("p2", 2, None), ("s", None, "S1"), ("t", None, "S2")]


# ---------------------------------------------------------------------------
# SHEET_CELL grouping
# ---------------------------------------------------------------------------
def test_sheet_grouping_header_pairs_and_order():
    doc = make_doc(doc_id="DEV-T04", file_type=FileType.XLSX, elements=[
        make_el("h1", ElementType.SHEET_CELL, text="transaction_id", value="transaction_id",
                sheet="tx", row=1, column=1),
        make_el("h2", ElementType.SHEET_CELL, text="amount", value="amount",
                sheet="tx", row=1, column=2),
        make_el("a1", ElementType.SHEET_CELL, text="TX-1", value="TX-1",
                sheet="tx", row=2, column=1),
        make_el("a2", ElementType.SHEET_CELL, text=None, value=2107.27,
                sheet="tx", row=2, column=2),
        make_el("b2", ElementType.SHEET_CELL, text="note", value="note",
                sheet="tx", row=3, column=5),
    ])
    chunks = chunk_document(doc)
    assert len(chunks) == 2  # header row 1 supplies names only, not chunked
    first, second = chunks
    assert first["kind"] == "sheet-row"
    assert first["sheet"] == "tx" and first["page"] is None
    assert first["text"] == "transaction_id: TX-1 | amount: 2107.27"
    assert first["element_ids"] == ["a1", "a2"]
    # Value-substring spans: "amount: 2107.27" starts at 23; value at 31.
    assert first["spans"][1] == {"element_id": "a2", "start": 31, "end": 38}
    # Missing header for an unheaded column falls back to empty header.
    assert second["text"] == ": note"


# ---------------------------------------------------------------------------
# CSV_ROW mapping order
# ---------------------------------------------------------------------------
def test_csv_mapping_order():
    doc = make_doc(doc_id="DEV-T05", file_type=FileType.CSV, elements=[
        make_el("r1", ElementType.CSV_ROW, value={"b": "2", "a": "1"}, row=1),
        make_el("r2", ElementType.CSV_ROW, value={"b": "", "a": "x"}, row=2),
    ])
    chunks = chunk_document(doc)
    assert [c["kind"] for c in chunks] == ["csv-row", "csv-row"]
    assert chunks[0]["text"] == "b: 2 | a: 1"
    assert chunks[1]["text"] == "b:  | a: x"
    assert chunks[0]["element_ids"] == ["r1"]


# ---------------------------------------------------------------------------
# OCR chunks (DEV-004 pageless, DEV-010 page 1; allowlist enforced)
# ---------------------------------------------------------------------------
def test_ocr_chunks_page_anchoring():
    d4 = make_doc(doc_id="DEV-004", filename="r.png", file_type=FileType.PNG,
                  elements=[make_el("img", ElementType.IMAGE_REF)])
    (chunk4,) = chunk_document(d4, ocr_text="ocr body")
    assert chunk4["kind"] == "ocr-text"
    assert chunk4["text"] == "ocr body"
    assert chunk4["page"] is None and chunk4["sheet"] is None
    assert chunk4["element_ids"] == [] and chunk4["spans"] == []

    d10 = make_doc(doc_id="DEV-010", elements=[])
    (chunk10,) = chunk_document(d10, ocr_text="balance sheet")
    assert chunk10["kind"] == "ocr-text"
    assert chunk10["page"] == 1


def test_ocr_allowlist_enforced():
    doc = make_doc(doc_id="DEV-001", elements=[
        make_el("t", ElementType.TEXT, text="hi")])
    with pytest.raises(TypeError):
        chunk_document(doc, ocr_text="not allowed here")


# ---------------------------------------------------------------------------
# Exclusions: charts, containers, image refs
# ---------------------------------------------------------------------------
def test_chart_container_imageref_exclusion():
    doc = make_doc(doc_id="DEV-T06", file_type=FileType.PNG, elements=[
        make_el("tbl", ElementType.TABLE, page=1, table_index=0),
        make_el("row", ElementType.TABLE_ROW, page=1, table_index=0, row=0),
        make_el("img", ElementType.IMAGE_REF),
    ])
    assert chunk_document(doc) == []


# ---------------------------------------------------------------------------
# IDs: format + determinism
# ---------------------------------------------------------------------------
def test_chunk_id_format_and_determinism():
    first = chunk_id_for("DEV-001", "text", 0, "Revenue")
    assert re.fullmatch(r"CHK-[0-9a-f]{12}", first)
    assert chunk_id_for("DEV-001", "text", 0, "Revenue") == first
    assert chunk_id_for("DEV-001", "text", 0, "revenue") != first
    assert chunk_id_for("DEV-001", "text", 1, "Revenue") != first
    assert chunk_id_for("DEV-002", "text", 0, "Revenue") != first


def test_rebuild_identity_same_inputs_twice():
    doc = _table_doc()
    run_a = chunk_document(doc)
    run_b = chunk_document(doc)
    assert run_a == run_b


def test_make_chunk_record_rejects_bad_kind_and_text():
    with pytest.raises(TypeError):
        make_chunk_record("D", "nope", 0, "x")
    with pytest.raises(TypeError):
        make_chunk_record("D", "text", 0, None)


# ---------------------------------------------------------------------------
# Normalizer (design section 4)
# ---------------------------------------------------------------------------
def test_normalize_arabic_indic_digits():
    assert normalize_text("TX-٢٠١٩-013526") == ["tx", "2019", "013526"]


def test_normalize_arabic_punctuation():
    assert normalize_text("مبلغ،123") == normalize_text("مبلغ,123")
    assert normalize_text("س؟") == normalize_text("س?")
    assert normalize_text("س؟") == ["س"]  # "?" maps but emits no token
    assert normalize_text("أ؛ب") == normalize_text("أ,ب")


def test_normalize_casefold_and_nfkc():
    assert normalize_text("Noor RETAIL") == ["noor", "retail"]
    assert normalize_text("ＡＢＣ１２３") == ["abc123"]  # NFKC; \w+ keeps it one token


def test_normalize_entity_parity_across_scripts():
    assert normalize_text("TX-٢٠١٩-013526") == normalize_text("TX-2019-013526")


def test_normalize_no_stopwords_no_stemming():
    assert normalize_text("the totals and assets") == ["the", "totals", "and", "assets"]


def test_normalize_type_error():
    with pytest.raises(TypeError):
        normalize_text(None)
    with pytest.raises(TypeError):
        normalize_text(123)


# ---------------------------------------------------------------------------
# Malformed / empty guards
# ---------------------------------------------------------------------------
def test_empty_elements_zero_chunks():
    assert chunk_document(make_doc(elements=[])) == []


def test_non_document_type_error():
    for bad in (None, "x", 42, {"document_id": "D"}, [make_doc()]):
        with pytest.raises(TypeError):
            chunk_document(bad)
    with pytest.raises(TypeError):
        chunk_corpus("not-a-list")
    with pytest.raises(TypeError):
        chunk_corpus([make_doc(), "nope"])


# ---------------------------------------------------------------------------
# Index roundtrip + determinism + failure isolation (tmp corpus)
# ---------------------------------------------------------------------------
def _stage_tmp_corpus(tmp_path):
    src = tmp_path / "docs"
    src.mkdir()
    shutil.copy(DOCS / "DEV-007_transactions_extract_2024_MIX.csv", src)
    shutil.copy(DOCS / "DEV-005_management_commentary_2023_MIX.docx", src)
    return src


def test_index_roundtrip_write_read_identical(tmp_path):
    docs_dir = _stage_tmp_corpus(tmp_path)
    out = tmp_path / "out"
    summary = build_index(docs_dir=docs_dir,
                          ocr_path=tmp_path / "missing.json", out_dir=out)
    assert (out / "chunks.json").is_file()
    assert (out / "index_meta.json").is_file()
    assert "vectors.json" not in [p.name for p in out.iterdir()]
    reread = json.loads((out / "chunks.json").read_text(encoding="utf-8"))
    assert reread == summary["chunks"]
    meta = json.loads((out / "index_meta.json").read_text(encoding="utf-8"))
    assert meta["embedding_model"]["id"] == "pending"
    assert meta["embedding_model"]["revision"] == "pending"
    assert meta["generated_at"]
    assert meta["counts"]["chunks_total"] == len(reread)
    assert any("ocr" in w["warning"] for w in meta["warnings"])
    assert summary["counts_by_doc"]
    assert set(summary["counts_by_doc"]) <= {"DEV-005", "DEV-007"}


def test_two_builds_identical_except_generated_at(tmp_path):
    docs_dir = _stage_tmp_corpus(tmp_path)
    first = build_index(docs_dir=docs_dir,
                        ocr_path=tmp_path / "missing.json",
                        out_dir=tmp_path / "o1")
    second = build_index(docs_dir=docs_dir,
                         ocr_path=tmp_path / "missing.json",
                         out_dir=tmp_path / "o2")
    assert first["chunks"] == second["chunks"]
    meta_a = dict(first["meta"])
    meta_b = dict(second["meta"])
    meta_a.pop("generated_at")
    meta_b.pop("generated_at")
    meta_a["timings_ms"] = {}
    meta_b["timings_ms"] = {}
    assert meta_a == meta_b


def test_build_index_skips_failed_doc_with_warning(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    shutil.copy(DOCS / "DEV-007_transactions_extract_2024_MIX.csv", docs_dir)
    (docs_dir / "empty.csv").write_text("a,b\n", encoding="utf-8")
    summary = build_index(docs_dir=docs_dir,
                          ocr_path=tmp_path / "missing.json",
                          out_dir=tmp_path / "o")
    assert summary["meta"]["counts"]["docs_skipped"] == ["empty.csv"]
    assert any(w["file"] == "empty.csv" for w in summary["warnings"])
    assert summary["counts_by_doc"].get("DEV-007", 0) > 0


def test_build_index_rejects_non_directory(tmp_path):
    with pytest.raises(NotADirectoryError):
        build_index(docs_dir=tmp_path / "nope", out_dir=tmp_path / "o")


# ---------------------------------------------------------------------------
# Import hygiene: no torch/transformers/network in R1 files
# ---------------------------------------------------------------------------
def test_no_heavy_or_network_imports():
    targets = [ROOT / "src" / "retrieval" / "__init__.py",
               ROOT / "src" / "retrieval" / "chunking.py",
               ROOT / "scripts" / "build_retrieval_index.py"]
    forbidden = {"torch", "transformers", "urllib", "requests", "socket",
                 "http", "httpx", "socketserver"}
    for target in targets:
        tree = ast.parse(target.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not (imported & forbidden), f"{target}: {imported & forbidden}"

# ---------------------------------------------------------------------------
# Recovery R1: out-of-order ingestion positions, OCR fingerprint, perf
# (additive; existing tests above untouched)
# ---------------------------------------------------------------------------
def _out_of_order_table_doc():
    # Cells ingested out of row/column order with a TEXT strictly between
    # the earliest-ingested member (c11) and the row-major-first cell
    # (c00): the group must emit at c11's position, not c00's.
    return make_doc(doc_id="DEV-R01", elements=[
        make_el("c11", ElementType.TABLE_CELL, text="d", page=1, row=1,
                column=1, table_index=0),
        make_el("mid", ElementType.TEXT, text="mid", page=1),
        make_el("c00", ElementType.TABLE_CELL, text="a", page=1, row=0,
                column=0, table_index=0),
        make_el("c01", ElementType.TABLE_CELL, text="b", page=1, row=0,
                column=1, table_index=0),
        make_el("c10", ElementType.TABLE_CELL, text="c", page=1, row=1,
                column=0, table_index=0),
        make_el("tail", ElementType.TEXT, text="tail", page=1),
    ])


def test_table_group_emits_at_earliest_ingestion_position():
    chunks = chunk_document(_out_of_order_table_doc())
    assert [(c["kind"], c["index"]) for c in chunks] == [
        ("table", 0), ("text", 1), ("text", 2)]
    table = chunks[0]
    assert [c["text"] for c in chunks] == ["a | b | c | d", "mid", "tail"]
    assert table["element_ids"] == ["c00", "c01", "c10", "c11"]
    assert table["spans"] == [
        {"element_id": "c00", "start": 0, "end": 1},
        {"element_id": "c01", "start": 4, "end": 5},
        {"element_id": "c10", "start": 8, "end": 9},
        {"element_id": "c11", "start": 12, "end": 13},
    ]
    # ID stability: same inputs twice -> identical records incl. IDs.
    assert chunk_document(_out_of_order_table_doc()) == chunks


def test_group_position_stable_under_member_reordering():
    cells = [
        make_el("c00", ElementType.TABLE_CELL, text="a", page=1, row=0,
                column=0, table_index=0),
        make_el("c01", ElementType.TABLE_CELL, text="b", page=1, row=0,
                column=1, table_index=0),
        make_el("c10", ElementType.TABLE_CELL, text="c", page=1, row=1,
                column=0, table_index=0),
    ]
    variants = []
    # TEXT sits strictly between earliest-ingested and row-major-first
    # members in every variant; only member order varies.
    for order in ((2, 1, 0), (1, 2, 0), (2, 0, 1)):
        doc = make_doc(doc_id="DEV-R02", elements=[
            cells[order[0]],
            make_el("t", ElementType.TEXT, text="t", page=1),
        ] + [cells[i] for i in order[1:]])
        variants.append(chunk_document(doc))
    # Chunk text always row-major, group always at index 0, records
    # (hence IDs) identical regardless of ingestion order of members.
    assert all(v[0]["text"] == "a | b | c" for v in variants)
    assert all([(c["kind"], c["index"]) for c in v] == [("table", 0),
                                                        ("text", 1)]
               for v in variants)
    assert all(v == variants[0] for v in variants)


def test_sheet_group_emits_at_earliest_ingestion_position():
    # Earliest-ingested data cell (v2, column 2) precedes a TEXT that
    # precedes the column-sorted-first cell (v1): the row chunk must emit
    # at v2's position, not v1's.
    doc = make_doc(doc_id="DEV-R03", file_type=FileType.XLSX, elements=[
        make_el("v2", ElementType.SHEET_CELL, text="20", value="20",
                sheet="s", row=2, column=2),
        make_el("mid", ElementType.TEXT, text="mid"),
        make_el("h1", ElementType.SHEET_CELL, text="id", value="id",
                sheet="s", row=1, column=1),
        make_el("h2", ElementType.SHEET_CELL, text="n", value="n",
                sheet="s", row=1, column=2),
        make_el("v1", ElementType.SHEET_CELL, text="X", value="X",
                sheet="s", row=2, column=1),
        make_el("w1", ElementType.CSV_ROW, value={"k": "v"}, row=1),
    ])
    chunks = chunk_document(doc)
    assert [(c["kind"], c["index"]) for c in chunks] == [
        ("sheet-row", 0), ("text", 1), ("csv-row", 2)]
    assert chunks[0]["text"] == "id: X | n: 20"
    assert chunks[0]["element_ids"] == ["v1", "v2"]
    assert chunk_document(doc) == chunks


def test_ocr_text_change_alters_fingerprint(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "DEV-004_mini.csv").write_text("a\n1\n", encoding="utf-8")
    sidecar = tmp_path / "ocr.json"
    sidecar.write_text(json.dumps([{"document_id": "DEV-004",
                                    "text": "first read"}]),
                       encoding="utf-8")
    run_a = build_index(docs_dir=docs_dir, ocr_path=sidecar,
                        out_dir=tmp_path / "a")
    sidecar.write_text(json.dumps([{"document_id": "DEV-004",
                                    "text": "second read"}]),
                       encoding="utf-8")
    run_b = build_index(docs_dir=docs_dir, ocr_path=sidecar,
                        out_dir=tmp_path / "b")
    fp_a = run_a["meta"]["corpus"]["ocr_fingerprint"]
    fp_b = run_b["meta"]["corpus"]["ocr_fingerprint"]
    assert fp_a != fp_b
    assert (run_a["meta"]["corpus"]["manifest_sha256"]
            != run_b["meta"]["corpus"]["manifest_sha256"])
    assert ocr_fingerprint_for({"DEV-004": "first read"}) == fp_a


def test_missing_sidecar_empty_marker_fingerprint(tmp_path):
    import hashlib
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "DEV-004_mini.csv").write_text("a\n1\n", encoding="utf-8")
    summary = build_index(docs_dir=docs_dir,
                          ocr_path=tmp_path / "absent.json",
                          out_dir=tmp_path / "o")
    assert summary["meta"]["corpus"]["ocr_fingerprint"] == hashlib.sha256(
        b"{}").hexdigest()
    assert any("ocr" in w["warning"] for w in summary["warnings"])
    assert all(c["kind"] != "ocr-text" for c in summary["chunks"])
