"""Deterministic tests for the REAL PaddleOCR-VL 1.6 API shape.

Synthetic fixtures mirror the Kaggle observation (PaddleOCR 3.7.0,
``PaddleOCRVL(pipeline_version="v1.6")`` returning
``PaddleOCRVLResult``-like page objects with
``page.json == {"res": {"parsing_res_list": [...]}}`` and table
``block_content`` as HTML). No Paddle import, no weights, no network.

Covers the task gates:
  A. paragraph_title extraction
  B. Arabic text preservation (exact, no normalization in adapter)
  C. parsing_res_list text extraction (order)
  D. HTML table -> row grid
  E. colspan row behavior (documented expansion)
  F. empty image block does not pollute recognition text
  G. numeric values remain character-faithful
  H. no semantic fields invented
  I. malformed/unknown result shape fails safely with warnings
  J. multiple page result objects combine deterministically
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ocrbench import paddle_adapter, schema  # noqa: E402


class FakePage:
    """Minimal PaddleOCRVLResult-like stand-in (.json + .markdown)."""

    def __init__(self, parsing_blocks, markdown=None):
        self.json = {"res": {"parsing_res_list": parsing_blocks}}
        self.markdown = markdown if markdown is not None else {"res": ""}


def _blocks_page(blocks):
    return {"res": {"parsing_res_list": blocks}}


TABLE_HTML = (
    "<table>"
    "<tr><td>X</td><td>24,371.25 SAR</td></tr>"
    "<tr><td>Cash</td><td>5,087,893.71</td></tr>"
    "</table>"
)


# A. paragraph_title extraction -------------------------------------------
def test_a_paragraph_title_extraction():
    raw = _blocks_page([
        {"block_label": "paragraph_title", "block_content": "T",
         "block_bbox": [0, 0, 1, 1]},
        {"block_label": "text", "block_content": "body line",
         "block_bbox": [0, 0, 1, 1]},
    ])
    text, fields, tables, visual, notes = paddle_adapter.adapt_output(raw)
    assert "T" in text.splitlines()
    assert "body line" in text
    assert fields == {}
    assert any("parsing_res_list" in n for n in notes)


# B. Arabic text preservation ----------------------------------------------
def test_b_arabic_text_preserved_exactly():
    arabic = "T"
    raw = _blocks_page([
        {"block_label": "text", "block_content": arabic,
         "block_bbox": [0, 0, 1, 1]},
    ])
    text, _, _, _, _ = paddle_adapter.adapt_output(raw)
    assert arabic in text  # adapter never normalizes/folds Arabic
    assert text.strip() == arabic


# C. parsing_res_list text extraction (order) -------------------------------
def test_c_parsing_res_list_order():
    raw = _blocks_page([
        {"block_label": "paragraph_title", "block_content": "first",
         "block_bbox": [0, 0, 1, 1]},
        {"block_label": "text", "block_content": "second",
         "block_bbox": [0, 0, 1, 1]},
        {"block_label": "text", "block_content": "third",
         "block_bbox": [0, 0, 1, 1]},
    ])
    text, _, _, _, _ = paddle_adapter.adapt_output(raw)
    assert text.split("\n") == ["first", "second", "third"]


# D. HTML table -> row grid -------------------------------------------------
def test_d_html_table_to_row_grid():
    raw = _blocks_page([
        {"block_label": "table", "block_content": TABLE_HTML,
         "block_bbox": [0, 0, 1, 1]},
    ])
    text, fields, tables, _, notes = paddle_adapter.adapt_output(raw)
    assert tables == [[["X", "24,371.25 SAR"],
                       ["Cash", "5,087,893.71"]]]
    # Each grid row contributes one " | "-joined recognition line.
    assert "X | 24,371.25 SAR" in text
    assert "Cash | 5,087,893.71" in text
    assert fields == {}
    assert any("table rows=2" in n for n in notes)


# E. colspan row behavior (documented) --------------------------------------
def test_e_colspan_expands_with_empty_padding():
    html = ('<table><tr><td colspan="2">TOTAL ASSETS</td></tr>'
            '<tr><td>Cash</td><td>5,087,893.71</td></tr></table>')
    raw = _blocks_page([
        {"block_label": "table", "block_content": html,
         "block_bbox": [0, 0, 1, 1]},
    ])
    _, _, tables, _, _ = paddle_adapter.adapt_output(raw)
    # Documented rule: colspan=N -> N cells, first holds text, rest "".
    # Rectangular without duplicating content into invented cells.
    assert tables == [[["TOTAL ASSETS", ""], ["Cash", "5,087,893.71"]]]


# F. empty image block does not pollute --------------------------------------
def test_f_empty_image_block_ignored():
    raw = _blocks_page([
        {"block_label": "text", "block_content": "hello",
         "block_bbox": [0, 0, 1, 1]},
        {"block_label": "image", "block_content": "",
         "block_bbox": [0, 0, 1, 1]},
        {"block_label": "image", "block_content": "   ",
         "block_bbox": [0, 0, 1, 1]},
    ])
    text, _, tables, visual, notes = paddle_adapter.adapt_output(raw)
    assert text.strip() == "hello"
    assert tables == []
    assert visual == ""
    assert any("ignored empty image block" in n for n in notes)


# G. numeric character-faithful ----------------------------------------------
def test_g_numeric_values_character_faithful():
    raw = _blocks_page([
        {"block_label": "table",
         "block_content": ("<table><tr><td>X</td>"
                           "<td>24,371.25 SAR</td></tr></table>"),
         "block_bbox": [0, 0, 1, 1]},
    ])
    text, _, tables, _, _ = paddle_adapter.adapt_output(raw)
    assert "24,371.25 SAR" in text  # grouping comma kept, not 24371.25
    assert tables[0][0][1] == "24,371.25 SAR"
    assert "24371.25" not in tables[0][0][1]


# H. no semantic fields invented ----------------------------------------------
def test_h_no_semantic_fields_invented():
    raw = _blocks_page([
        {"block_label": "table",
         "block_content": ("<table><tr><td>X</td>"
                           "<td>24,371.25 SAR</td></tr></table>"),
         "block_bbox": [0, 0, 1, 1]},
    ])
    _, fields, _, _, notes = paddle_adapter.adapt_output(raw)
    assert fields == {}
    assert "amount" not in fields
    built = paddle_adapter.build_result(
        "DEV-004", raw, 1.0,
        paddle_adapter.identity_for("3.7.0", "3.2.1", "kaggle-T4"))
    assert built["fields"] == {}
    assert schema.validate_result(built) == []


# I. malformed/unknown shape fails safely --------------------------------------
def test_i_malformed_shapes_fail_safely():
    for bad in ({"res": "oops"},
                {"res": {"parsing_res_list": "not-a-list"}},
                {"weird": 1},
                12345,
                None):
        text, fields, tables, visual, notes = paddle_adapter.adapt_output(bad)
        assert isinstance(text, str)
        assert isinstance(fields, dict)
        assert isinstance(tables, list)
        assert notes  # at least one warning, never silent, never crash
    # Non-dict block inside a valid list is stringified, not dropped.
    raw = _blocks_page(["just a string", 7])
    text, _, _, _, notes = paddle_adapter.adapt_output(raw)
    assert "just a string" in text and "7" in text
    assert any("non-dict" in n for n in notes)


# J. multiple pages combine deterministically -----------------------------------
def test_j_multiple_pages_combine_in_order():
    p1 = FakePage([
        {"block_label": "text", "block_content": "page-one",
         "block_bbox": [0, 0, 1, 1]},
        {"block_label": "table",
         "block_content": ("<table><tr><td>A</td><td>1.00</td></tr></table>"),
         "block_bbox": [0, 0, 1, 1]},
    ])
    p2 = FakePage([
        {"block_label": "text", "block_content": "page-two",
         "block_bbox": [0, 0, 1, 1]},
    ])
    text, fields, tables, _, notes = paddle_adapter.adapt_output([p1, p2])
    assert text.split("\n") == ["page-one", "A | 1.00", "page-two"]
    assert tables == [[["A", "1.00"]]]
    assert fields == {}
    assert any("pages ×2" in n for n in notes)
    # Same input as raw dict payloads gives the same result (both accepted).
    text2, _, tables2, _, _ = paddle_adapter.adapt_output(
        [p1.json, p2.json])
    assert text2 == text and tables2 == tables
    # Reversed input reverses output (order-sensitive, deterministic).
    rev_text, _, _, _, _ = paddle_adapter.adapt_output([p2, p1])
    assert rev_text.split("\n") == ["page-two", "page-one", "A | 1.00"]
