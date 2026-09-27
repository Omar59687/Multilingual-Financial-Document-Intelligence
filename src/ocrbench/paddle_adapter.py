"""PaddleOCR-VL-1.6 benchmark adapter (evaluation-side, stdlib only).

Covers the Kaggle-side integration contract WITHOUT importing Paddle here:

- identity constants (candidate / model id / pipeline version — locked),
- PaddleSession: lazy, once-per-session pipeline creation with init
  latency measurement (importer + clock injectable for tests),
- adapt_output: defensive normalization of structured pipeline outputs
  into (text, fields, tables, visual_description, notes),
- blocker_result / build_result: schema-conformant result construction.

Real v1.6 API observed in Kaggle (PaddleOCR 3.7.0, PaddlePaddle 3.2.1,
Tesla T4, ``PaddleOCRVL(pipeline_version="v1.6")``):

- ``predict(...)`` returns an iterable of page result objects of type
  ``paddlex.inference.pipelines.paddleocr_vl.result.PaddleOCRVLResult``.
- Each page has ``page.json`` (dict) with top-level key ``"res"`` and
  useful content under ``page.json["res"]["parsing_res_list"]``.
- Each block looks like
  ``{"block_label": ..., "block_content": ..., "block_bbox": [...]}``
  with labels observed: ``paragraph_title``, ``text``, ``table``,
  ``image``.
- Table ``block_content`` is HTML (e.g. ``<table><tr><td>…``).
- ``page.markdown`` is also a dict but the canonical adapter prefers
  ``parsing_res_list`` because it preserves block structure.

Adapter rules (canonical):

- Accept real ``PaddleOCRVLResult``-like objects (anything with a
  ``.json`` dict attribute, ``.markdown`` optional) and/or raw
  ``page.json`` dict payloads (``{"res": {"parsing_res_list": [...]}}``),
  single pages or lists/generators of pages combined in order.
- Recognition text comes from ``parsing_res_list`` verbatim (Arabic
  preserved exactly, numbers character-faithful, no normalization).
- ``table`` blocks are parsed from HTML deterministically into
  list-of-row grids (stdlib ``html.parser`` only); each grid row also
  contributes one ``" | "``-joined line to recognition text so numeric
  and label recall work without inventing semantic fields.
- ``image`` blocks never contribute to recognition text; empty ones are
  ignored, non-empty ones are kept as visual description only.
- ``fields`` is ALWAYS ``{}`` for v1.6 shapes: Paddle emits recognition
  + structure, never semantic fields like ``amount``. Recognition and
  table structure stay separate from downstream canonical field
  extraction by design.
- Raw Paddle output preservation is the caller's job (notebook writes
  ``paddle_raw_<DEV>.json``; :func:`build_result` records ``raw_path``
  in notes). Bboxes stay in the raw file, not in the normalized text.
- Legacy/simple shapes (plain strings, ``rec_texts``, ``blocks``,
  ``tables``, ``markdown`` …) are still handled where safe for backward
  compatibility. Unknown shapes are stringified + noted, never dropped,
  never hand-corrected.

No Paddle import, no weights, no inference in this module — the real
``from paddleocr import PaddleOCRVL`` call happens only inside the Kaggle
notebook (mirrored logic) via an injected importer.
"""

import html as _html
import time
from html.parser import HTMLParser

from . import schema

CANDIDATE = "paddleocr-vl"
MODEL_ID = "PaddleOCR-VL-1.6"
PIPELINE_VERSION = "v1.6"

# First-baseline pipeline posture: official v1.6 defaults, NO overrides.
# Recorded (not tuned): orientation classification, unwarping, layout
# detection, chart recognition all stay at whatever the official pipeline
# does by default. Any future non-default option must be added here and
# recorded in the exported identity.
PIPELINE_OPTIONS = {
    "doc_orientation_classify": "official-default (not overridden)",
    "doc_unwarping": "official-default (not overridden)",
    "layout_detection": "official-default (not overridden)",
    "chart_recognition": "official-default (not overridden)",
}


def identity_for(paddleocr_version=None, paddle_version=None, device=None):
    """Reproducibility identity stamped on every Paddle result/meta."""
    return {
        "candidate": CANDIDATE,
        "model_id": MODEL_ID,
        "pipeline_version": PIPELINE_VERSION,
        "paddleocr_version": paddleocr_version or "unknown",
        "paddle_version": paddle_version or "unknown",
        "device": device or "unknown",
        "pipeline_options": dict(PIPELINE_OPTIONS),
    }


class PaddleSession:
    """Lazily creates the v1.6 pipeline exactly once per session.

    importer: zero-arg callable returning the PaddleOCRVL CLASS
      (default performs the real ``from paddleocr import PaddleOCRVL``).
      Tests inject a fake; the Kaggle notebook uses the default.
    clock: time.perf_counter-compatible (injectable for tests).
    """

    def __init__(self, importer=None, clock=None):
        self._importer = importer or self._default_importer
        self._clock = clock or time.perf_counter
        self._pipeline = None
        self._init_latency_ms = None
        self._identity_note = None

    @staticmethod
    def _default_importer():
        from paddleocr import PaddleOCRVL  # Kaggle-side only, lazy
        return PaddleOCRVL

    def ensure(self):
        """Return the cached pipeline, creating + timing it on first use."""
        if self._pipeline is None:
            started = self._clock()
            pipeline_cls = self._importer()
            # Authoritative v1.6 entry point; no version substitution.
            self._pipeline = pipeline_cls(pipeline_version=PIPELINE_VERSION)
            self._init_latency_ms = (self._clock() - started) * 1000.0
        return self._pipeline

    @property
    def init_latency_ms(self):
        """Model init/download time — reported separately, NEVER counted
        as per-document OCR latency."""
        return self._init_latency_ms

    @property
    def created(self):
        return self._pipeline is not None


def _coerce_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(str(v) for v in value)
    return str(value)


def _coerce_grid(table):
    """Best-effort grid normalization; returns None when unusable.

    NOTE: raw HTML strings are kept verbatim as a single cell here.
    That is the legacy fallback only. The canonical v1.6 table path uses
    :func:`_parse_html_tables` instead (deterministic HTML -> grids).
    """
    if isinstance(table, str):  # e.g. raw HTML kept verbatim, one cell
        return [[table]]
    if isinstance(table, dict):
        for key in ("rows", "cells", "data", "grid"):
            if isinstance(table.get(key), list):
                table = table[key]
                break
        else:
            return None
    if not isinstance(table, list):
        return None
    grid = []
    for row in table:
        if isinstance(row, (list, tuple)):
            grid.append([str(c) for c in row])
        elif isinstance(row, dict):
            cells = row.get("cells", row.get("values"))
            if isinstance(cells, list):
                grid.append([str(c) for c in cells])
        elif isinstance(row, str):
            grid.append([row])
    return grid or None


# --------------------------------------------------------------------------
# Real v1.6 result shape: parsing_res_list + HTML tables (stdlib only)
# --------------------------------------------------------------------------
class _TableHTMLParser(HTMLParser):
    """Deterministic ``<table>`` -> grids using stdlib only.

    - ``<td>``/``<th>`` become one cell each; text is outer-stripped,
      HTML entities resolved by the parser, inner spacing preserved
      (Arabic exactly as returned, numbers character-faithful).
    - ``colspan=N`` expands to N cells: first holds the text, the
      remaining N-1 are ``""``. Rectangular without duplicating content.
    - ``rowspan`` is NOT vertically expanded (treated as a single cell);
      its presence is reported by the caller via substring check so the
      behavior stays explicit and deterministic.
    - Orphan ``<tr>``/``<td>`` outside ``<table>`` form an implicit table
      (tolerant to fragments). ``<br>`` inside a cell becomes a space.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables = []
        self._table_stack = []
        self._row_stack = []
        self._cell_stack = []

    def _ensure_table(self):
        if not self._table_stack:
            self._table_stack.append([])

    def _ensure_row(self):
        self._ensure_table()
        if not self._row_stack:
            self._row_stack.append([])

    def handle_starttag(self, tag, attrs):
        t = tag.lower()
        if t == "table":
            self._table_stack.append([])
        elif t == "tr":
            self._ensure_table()
            self._row_stack.append([])
        elif t in ("td", "th"):
            self._ensure_row()
            colspan = 1
            for key, value in attrs or []:
                if key.lower() == "colspan":
                    try:
                        colspan = max(1, int(str(value).strip()))
                    except (ValueError, TypeError, AttributeError):
                        colspan = 1
            self._cell_stack.append({"chunks": [], "colspan": colspan})
        elif t == "br":
            if self._cell_stack:
                self._cell_stack[-1]["chunks"].append(" ")

    def handle_startendtag(self, tag, attrs):
        t = tag.lower()
        if t in ("td", "th"):
            self.handle_starttag(tag, attrs)
            self.handle_endtag(tag)
        elif t == "br":
            if self._cell_stack:
                self._cell_stack[-1]["chunks"].append(" ")

    def handle_data(self, data):
        if self._cell_stack:
            self._cell_stack[-1]["chunks"].append(data)

    def handle_endtag(self, tag):
        t = tag.lower()
        if t in ("td", "th"):
            if not self._cell_stack:
                return
            cell = self._cell_stack.pop()
            text = "".join(cell["chunks"]).strip()
            if not self._row_stack:
                self._ensure_row()
            row = self._row_stack[-1]
            row.append(text)
            for _ in range(cell["colspan"] - 1):
                row.append("")
        elif t == "tr":
            if not self._row_stack:
                return
            row = self._row_stack.pop()
            if row:  # skip <tr></tr> with zero cells; keep 1+-cell rows
                self._ensure_table()
                self._table_stack[-1].append(row)
        elif t == "table":
            if not self._table_stack:
                return
            table = self._table_stack.pop()
            # Flush tags left open inside this table (malformed HTML).
            while self._cell_stack:
                cell = self._cell_stack.pop()
                text = "".join(cell["chunks"]).strip()
                if not self._row_stack:
                    self._row_stack.append([])
                crow = self._row_stack[-1]
                crow.append(text)
                for _ in range(cell["colspan"] - 1):
                    crow.append("")
            while self._row_stack:
                orow = self._row_stack.pop()
                if orow:
                    table.append(orow)
            if table:
                self.tables.append(table)


def _parse_html_tables(html_text):
    """Parse an HTML table string into a list of grids.

    Returns [] when no parseable table exists. Deterministic: document
    order, no sorting, no content normalization beyond outer strip +
    entity resolution. Tables whose cells are all empty are skipped.
    """
    if not isinstance(html_text, str) or not html_text.strip():
        return []
    lowered = html_text.lower()
    if "<td" not in lowered and "<th" not in lowered \
            and "<tr" not in lowered:
        return []
    parser = _TableHTMLParser()
    try:
        parser.feed(html_text)
        parser.close()
    except Exception:
        return []
    try:
        while parser._cell_stack:
            cell = parser._cell_stack.pop()
            text = "".join(cell["chunks"]).strip()
            if not parser._row_stack:
                parser._ensure_row()
            parser._row_stack[-1].append(text)
            for _ in range(cell["colspan"] - 1):
                parser._row_stack[-1].append("")
        while parser._row_stack:
            row = parser._row_stack.pop()
            if row:
                parser._ensure_table()
                parser._table_stack[-1].append(row)
        while parser._table_stack:
            table = parser._table_stack.pop()
            if table:
                parser.tables.append(table)
    except Exception:
        return []
    out = []
    for table in parser.tables:
        if table and any(any(c.strip() for c in row) for row in table):
            out.append([[str(c) for c in row] for row in table])
    return out


def _page_json_and_markdown(obj):
    """Split a page-like object into (page_json, page_markdown).

    Accepts real ``PaddleOCRVLResult``-like objects (``.json`` dict
    attribute, optional ``.markdown``) and raw ``page.json`` dict payloads
    (``{"res": {"parsing_res_list": [...]}}`` or tolerant
    ``{"parsing_res_list": [...]}``), plus serialized
    ``{"json": {...}, "markdown": {...}}`` forms. Returns
    ``(None, None)`` when not page-like.
    """
    page_json, page_md = None, None
    if not isinstance(obj, (str, bytes, dict, list, tuple)):
        if hasattr(obj, "json"):
            try:
                value = getattr(obj, "json")
                if callable(value):
                    try:
                        value = value()
                    except TypeError:
                        value = None
                if isinstance(value, dict):
                    page_json = value
            except Exception:
                page_json = None
        if hasattr(obj, "markdown"):
            try:
                value = getattr(obj, "markdown")
                if callable(value):
                    try:
                        value = value()
                    except TypeError:
                        value = None
                if isinstance(value, (dict, str)):
                    page_md = value
            except Exception:
                page_md = None
        if page_json is not None:
            return page_json, page_md
    if isinstance(obj, dict):
        res = obj.get("res")
        if isinstance(res, dict) \
                and isinstance(res.get("parsing_res_list"), list):
            md = obj.get("markdown")
            return obj, (md if isinstance(md, (dict, str)) else None)
        if isinstance(obj.get("parsing_res_list"), list):
            return obj, None
        if isinstance(obj.get("json"), dict):
            md = obj.get("markdown")
            return obj["json"], (md if isinstance(md, (dict, str))
                                 else None)
    return None, None


def _is_page_like(obj):
    page_json, _ = _page_json_and_markdown(obj)
    return page_json is not None


def _collect_v16_pages(raw):
    """Collect (page_json, page_markdown) pairs for real v1.6 shapes.

    Returns a list for single pages, lists/tuples/generators of pages
    (combined deterministically in order), else None (not a v1.6 shape —
    caller falls back to legacy handling).
    """
    page_json, page_md = _page_json_and_markdown(raw)
    if page_json is not None:
        return [(page_json, page_md)]
    if isinstance(raw, (list, tuple)):
        if raw and all(_is_page_like(e) for e in raw):
            return [_page_json_and_markdown(e) for e in raw]
        return None
    if hasattr(raw, "__iter__") \
            and not isinstance(raw, (str, bytes, dict, bytearray)):
        try:
            items = list(raw)
        except Exception:
            return None
        if items and all(_is_page_like(e) for e in items):
            return [_page_json_and_markdown(e) for e in items]
        return None
    return None


def _extract_parsing_res_list(page_json):
    """Canonical block list location, tolerant to a flat variant."""
    if not isinstance(page_json, dict):
        return None
    res = page_json.get("res")
    if isinstance(res, dict) \
            and isinstance(res.get("parsing_res_list"), list):
        return res["parsing_res_list"]
    if isinstance(page_json.get("parsing_res_list"), list):
        return page_json["parsing_res_list"]
    return None


def _extract_markdown_text(page_markdown):
    """Best-effort text from a ``page.markdown`` dict (fallback only)."""
    if isinstance(page_markdown, str):
        return page_markdown if page_markdown.strip() else None
    if isinstance(page_markdown, dict):
        res = page_markdown.get("res")
        if isinstance(res, str) and res.strip():
            return res
        parts = [v for v in page_markdown.values()
                 if isinstance(v, str) and v.strip()]
        if parts:
            return "\n".join(parts)
        if isinstance(res, dict):
            sub = [v for v in res.values()
                   if isinstance(v, str) and v.strip()]
            if sub:
                return "\n".join(sub)
    return None


def _adapt_v16_pages(pages, doc_id=""):
    """Normalize real v1.6 pages into (text, fields, tables, visual, notes).

    ``fields`` is intentionally ALWAYS ``{}`` here: v1.6 emits recognition
    + structure, never semantic canonical fields. A row like
    ``المبلغ | 24,371.25 SAR`` stays as recognition text + a table grid
    row — no ``fields["amount"]`` is invented.
    """
    texts, fields, tables, visuals, notes = [], {}, [], [], []
    notes.append(f"shape: paddle-vl-1.6 pages ×{len(pages)}")
    for pidx, (page_json, page_md) in enumerate(pages):
        blocks = _extract_parsing_res_list(page_json)
        if not isinstance(blocks, list):
            md_text = _extract_markdown_text(page_md)
            if md_text:
                texts.append(md_text)
                notes.append(f"page {pidx}: fallback to markdown "
                             "(no parsing_res_list)")
            else:
                keys = sorted(page_json.keys()) \
                    if isinstance(page_json, dict) else type(page_json).__name__
                notes.append(f"page {pidx}: no parsing_res_list; "
                             f"keys={keys}")
            continue
        notes.append(f"page {pidx}: parsing_res_list ×{len(blocks)}")
        label_counts = {}
        for bidx, block in enumerate(blocks):
            if not isinstance(block, dict):
                texts.append(str(block))
                notes.append(f"page {pidx} block {bidx}: "
                             "non-dict stringified")
                continue
            label = block.get("block_label",
                              block.get("label", block.get("type", "")))
            if not isinstance(label, str):
                label = str(label)
            label_counts[label] = label_counts.get(label, 0) + 1
            if "block_content" in block:
                content = block["block_content"]
            elif "content" in block:
                content = block["content"]
            elif "text" in block:
                content = block["text"]
            else:
                content = ""
            if label == "image":
                if content is None or content == "" or (
                        isinstance(content, str) and not content.strip()):
                    notes.append(f"page {pidx}: ignored empty image block")
                    continue
                visuals.append(content if isinstance(content, str)
                               else str(content))
                notes.append(f"page {pidx}: non-empty image block kept "
                             "as visual, not text")
                continue
            if label == "table":
                if isinstance(content, str) and content.strip():
                    grids = _parse_html_tables(content)
                    if grids:
                        for grid in grids:
                            tables.append(grid)
                            for row in grid:
                                texts.append(" | ".join(row))
                        nrows = sum(len(g) for g in grids)
                        notes.append(f"page {pidx}: table rows={nrows}")
                        if "rowspan" in content.lower():
                            notes.append(
                                f"page {pidx}: rowspan present; "
                                "kept as single cell (no vertical expand)")
                    else:
                        texts.append(content)  # verbatim, never dropped
                        notes.append(f"page {pidx}: table HTML unparseable, "
                                     "kept verbatim")
                elif isinstance(content, (list, dict)):
                    grid = _coerce_grid(content)
                    if grid:
                        tables.append(grid)
                        for row in grid:
                            texts.append(" | ".join(row))
                        notes.append(f"page {pidx}: table (structured) "
                                     f"rows={len(grid)}")
                    else:
                        notes.append(f"page {pidx}: empty structured table "
                                     "skipped")
                else:
                    notes.append(f"page {pidx}: empty table block skipped")
                continue
            # Text-like blocks (paragraph_title, text, unknown): verbatim.
            if isinstance(content, str):
                if content.strip():
                    texts.append(content)
                else:
                    notes.append(f"page {pidx}: skipped empty "
                                 f"{label or 'unlabeled'} block")
            elif isinstance(content, (list, tuple)):
                joined = "\n".join(str(c) for c in content)
                if joined.strip():
                    texts.append(joined)
                else:
                    notes.append(f"page {pidx}: skipped empty "
                                 f"{label or 'unlabeled'} block")
            elif content is None or content == "":
                notes.append(f"page {pidx}: skipped empty "
                             f"{label or 'unlabeled'} block")
            else:
                texts.append(str(content))
        if label_counts:
            dist = ", ".join(f"{k or 'unlabeled'}×{v}"
                             for k, v in sorted(label_counts.items()))
            notes.append(f"page {pidx} labels: {dist}")
        if not blocks and page_md:
            md_text = _extract_markdown_text(page_md)
            if md_text:
                texts.append(md_text)
                notes.append(f"page {pidx}: empty blocks, "
                             "fallback to markdown")
    notes.append("fields: none emitted by v1.6 "
                 "(semantic extraction is downstream)")
    return "\n".join(texts), fields, tables, "\n".join(visuals), notes


def adapt_output(raw, doc_id=""):
    """Normalize structured pipeline output.

    Returns (text, fields, tables, visual_description, notes). The real
    v1.6 ``parsing_res_list`` shape is tried FIRST (canonical); legacy
    plain strings, Paddle-style dicts (rec_texts/rec_scores, blocks,
    tables, markdown, fields), and lists of blocks/strings are kept where
    safe. Unknown shapes are stringified into text + noted — never
    silently dropped, never hand-corrected.
    """
    try:
        pages = _collect_v16_pages(raw)
    except Exception:
        pages = None
    if pages is not None:
        try:
            return _adapt_v16_pages(pages, doc_id)
        except Exception as exc:
            return "", {}, [], "", [
                f"v1.6 adapt failed safely: {type(exc).__name__}: {exc}"]
    texts, fields, tables, visuals, notes = [], {}, [], [], []
    if isinstance(raw, str):
        texts.append(raw)
        notes.append("shape: plain string")
        return "\n".join(texts), fields, tables, "", notes
    if isinstance(raw, (list, tuple)):
        if raw and all(isinstance(b, str) for b in raw):
            notes.append(f"shape: string list ×{len(raw)}")
            return "\n".join(raw), fields, tables, "", notes
        raw = {"blocks": list(raw)}
        notes.append("shape: block list -> blocks")
    if not isinstance(raw, dict):
        texts.append(str(raw))
        notes.append(f"shape: unrecognized {type(raw).__name__}; "
                     "stringified verbatim")
        return "\n".join(texts), fields, tables, "", notes

    for key in ("markdown", "md"):
        if isinstance(raw.get(key), str) and raw[key].strip():
            texts.append(raw[key])
            notes.append(f"used key: {key}")
    for key in ("text", "transcription", "content", "ocr_text"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            texts.append(value)
            notes.append(f"used key: {key}")
        elif isinstance(value, list) and value:
            texts.append(_coerce_text(value))
            notes.append(f"used key: {key} (list)")
    if isinstance(raw.get("rec_texts"), list) and raw["rec_texts"]:
        texts.append("\n".join(str(t) for t in raw["rec_texts"]))
        notes.append(f"used rec_texts ×{len(raw['rec_texts'])}")
        if "rec_scores" in raw:
            notes.append("rec_scores present (not scored, not in fields)")
    for key in ("blocks", "layout", "elements", "regions"):
        blocks = raw.get(key)
        if not isinstance(blocks, list) or not blocks:
            continue
        notes.append(f"used key: {key} ×{len(blocks)}")
        for block in blocks:
            if not isinstance(block, dict):
                texts.append(str(block))
                continue
            label = block.get("block_label", block.get("label",
                                                       block.get("type", "")))
            if "block_content" in block:
                raw_body = block["block_content"]
            else:
                raw_body = block.get("text", block.get("content", ""))
            body = _coerce_text(raw_body)
            if body.strip():
                texts.append(f"[{label}] {body}" if label else body)
            for tkey in ("table", "tables", "cells", "rows", "html"):
                if tkey in block and block[tkey] is not None:
                    grid = _coerce_grid(block[tkey])
                    if grid:
                        tables.append(grid)
                        notes.append(f"block table via {tkey}")
    raw_tables = raw.get("tables", raw.get("table"))
    if raw_tables is not None:
        batch = raw_tables if isinstance(raw_tables, list) else [raw_tables]
        for table in batch:
            grid = _coerce_grid(table)
            if grid:
                tables.append(grid)
        notes.append("used key: tables/table")
    for key in ("fields", "key_values"):
        mapping = raw.get(key)
        if isinstance(mapping, dict):
            for fkey, fval in mapping.items():
                fields[str(fkey)] = fval if isinstance(fval, str) \
                    else str(fval)
            notes.append(f"used key: {key} ×{len(mapping)}")
    for key in ("chart", "figures", "chart_result", "visual_description",
                "description"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            visuals.append(value)
            notes.append(f"used key: {key}")
        elif isinstance(value, (list, dict)) and value:
            visuals.append(str(value))
            notes.append(f"used key: {key} (structured, stringified)")
    if not texts and not tables and not fields:
        notes.append("no recognized content keys; raw stringified to text")
        texts.append(str(raw))
    return "\n".join(texts), fields, tables, "\n".join(visuals), notes


def blocker_result(doc_id, reason, device=None):
    """Schema-valid result recording an import/runtime blocker (no crash)."""
    result = schema.blank_result(CANDIDATE, MODEL_ID,
                                 device or "unknown", doc_id)
    result["warnings"] = [reason]
    return result


def build_result(doc_id, raw, latency_ms, identity, raw_path=None,
                 notes_extra=None):
    """Schema-valid result from adapted pipeline output (no hand edits)."""
    text, fields, tables, visual, notes = adapt_output(raw, doc_id)
    if raw_path:
        notes.append(f"raw output preserved at: {raw_path}")
    if notes_extra:
        notes.extend(notes_extra)
    return {
        "model": identity.get("candidate", CANDIDATE),
        "model_version": identity.get("model_id", MODEL_ID),
        "device": identity.get("device", "unknown"),
        "document_id": doc_id,
        "latency_ms": latency_ms,
        "text": text,
        "fields": fields,
        "tables": tables,
        "visual_description": visual,
        "warnings": notes,
    }
