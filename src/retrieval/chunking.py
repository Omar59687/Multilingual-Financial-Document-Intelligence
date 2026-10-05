"""Deterministic DEV-corpus chunking + shared normalizer + local index build.

Implements docs/RETRIEVAL_DESIGN.md sections 2, 3, 4, 8 (Phase 4 R1).
Text memory only: chunk records carry verbatim text plus provenance
metadata; no embeddings/vectors are produced here (later package).

Stdlib only (hashlib/json/re/unicodedata plus ingestion imports).
No network, no torch/transformers.
"""

import hashlib
import json
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from ingestion.model import Document, ElementType
from ingestion.service import ingest_document

CHUNKER_NAME = "mizaniq-text-chunker"
CHUNKER_VERSION = "1.0.0"
CHUNK_ID_PREFIX = "CHK-"

# Joiners frozen for R1 (recorded in index_meta chunker params).
TABLE_JOINER = " | "
PAIR_JOINER = " | "

# XLSX header row supplying "header: value" names for sheet-row chunks.
SHEET_HEADER_ROW = 1

# OCR corpus allowlist (design section 2: Paddle route texts for scanned
# documents DEV-004 + DEV-010 ONLY) with per-doc page anchoring
# (DEV-004 pageless/image-level, DEV-010 page 1).
OCR_DOC_PAGES = {"DEV-004": None, "DEV-010": 1}

CHUNK_KINDS = ("text", "heading", "table", "sheet-row", "csv-row", "ocr-text")

_DEFAULTS_ANCHOR = Path(__file__).resolve().parents[2]
DEFAULT_DOCS_DIR = _DEFAULTS_ANCHOR / "data" / "dev" / "dataset_v0.1" / "documents"
DEFAULT_OCR_PATH = _DEFAULTS_ANCHOR / "data" / "benchmark" / "results" / "paddle_results.json"
DEFAULT_OUT_DIR = _DEFAULTS_ANCHOR / "data" / "retrieval"

# Design section 4 maps.
_ARABIC_INDIC_DIGITS = {chr(0x0660 + i): str(i) for i in range(10)}
_ARABIC_PUNCT = {"\u060c": ",", "\u061b": ",", "\u061f": "?"}
_TOKEN_RE = re.compile(r"\w+")


# ---------------------------------------------------------------------------
# Normalizer (design section 4)
# ---------------------------------------------------------------------------
def normalize_text(text):
    """Tokenize per the frozen normalization contract.

    NFKC -> Latin casefold -> Arabic-Indic digits to Western ->
    Arabic punctuation to ASCII (,,?) -> ``\\w+`` Unicode word tokens.
    No stopword removal, no stemming.
    """
    if not isinstance(text, str):
        raise TypeError(f"normalize_text expects str, got {type(text).__name__}")
    out = unicodedata.normalize("NFKC", text)
    out = out.casefold()
    out = "".join(_ARABIC_INDIC_DIGITS.get(ch, ch) for ch in out)
    out = "".join(_ARABIC_PUNCT.get(ch, ch) for ch in out)
    return _TOKEN_RE.findall(out)


# ---------------------------------------------------------------------------
# Chunk ID + record builder
# ---------------------------------------------------------------------------
def chunk_id_for(document_id, kind, index, text):
    """Stable ID: CHK-<first 12 lowercase hex of sha256 over canonical JSON.

    Canonical form: json.dumps({document_id, kind, index, text},
    sort_keys=True, compact separators) encoded UTF-8.
    """
    canonical = json.dumps(
        {"document_id": document_id, "kind": kind, "index": index, "text": text},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return CHUNK_ID_PREFIX + hashlib.sha256(canonical).hexdigest()[:12]


def make_chunk_record(document_id, kind, index, text, filename=None,
                       file_type=None, language=None, page=None, sheet=None,
                       section=None, element_ids=(), spans=()):
    """Build one chunk record dict and assign its deterministic ID."""
    if kind not in CHUNK_KINDS:
        raise TypeError(f"unknown chunk kind: {kind!r}")
    if not isinstance(text, str):
        raise TypeError(f"chunk text must be str, got {type(text).__name__}")
    element_ids = list(element_ids)
    spans = [dict(s) for s in spans]
    return {
        "chunk_id": chunk_id_for(document_id, kind, index, text),
        "document_id": document_id,
        "kind": kind,
        "index": index,
        "text": text,
        "filename": filename,
        "file_type": file_type,
        "language": language,
        "page": page,
        "sheet": sheet,
        "section": section,
        "element_ids": element_ids,
        "spans": spans,
        "length": len(text),
    }


# ---------------------------------------------------------------------------
# Element helpers
# ---------------------------------------------------------------------------
def _element_kind(element):
    et = element.element_type
    return et.value if isinstance(et, ElementType) else et


def _verbatim(element):
    """Verbatim cell/text rendering: text as-is; non-string native values
    via str(); nothing present -> empty string."""
    if isinstance(element.text, str):
        return element.text
    if element.value is not None:
        return str(element.value)
    return ""


def _doc_base(document):
    file_type = document.file_type
    return {
        "filename": document.filename,
        "file_type": file_type.value if file_type is not None
        and hasattr(file_type, "value") else file_type,
        "language": document.language_hint.value
        if hasattr(document.language_hint, "value")
        else document.language_hint,
    }


def _spans_for_joined(pieces):
    """Positional spans for joiner-joined pieces.

    pieces: list of (element_id, substring). Joiner is TABLE_JOINER.
    Offsets are arithmetic (never str.find) so repeated text is exact.
    """
    spans = []
    offset = 0
    for i, (element_id, piece) in enumerate(pieces):
        if i:
            offset += len(TABLE_JOINER)
        spans.append({"element_id": element_id, "start": offset,
                      "end": offset + len(piece)})
        offset += len(piece)
    return spans


# ---------------------------------------------------------------------------
# Per-kind chunk planners (index assigned later, in emission order)
# ---------------------------------------------------------------------------
def _plan_text_heading(document):
    """One chunk per TEXT/HEADING element. TEXT carries section = nearest
    preceding HEADING text in the same document, else None; HEADING
    chunks carry section None."""
    plans = []
    last_heading = None
    for el in document.elements:
        kind = _element_kind(el)
        if kind == "heading":
            plans.append({
                "kind": "heading", "text": _verbatim(el), "page": el.page,
                "sheet": None, "section": None,
                "element_ids": [el.element_id],
                "spans": [{"element_id": el.element_id, "start": 0,
                           "end": len(_verbatim(el))}],
            })
            if isinstance(el.text, str):
                last_heading = el.text
        elif kind == "text":
            text = _verbatim(el)
            plans.append({
                "kind": "text", "text": text, "page": el.page,
                "sheet": None, "section": last_heading,
                "element_ids": [el.element_id],
                "spans": [{"element_id": el.element_id, "start": 0,
                           "end": len(text)}],
            })
    return plans


def _table_group_key(element):
    return (element.page, element.table_index, element.sheet)


def _plan_tables(document, positions):
    """TABLE_CELL grouped by (page, table_index) [+ sheet where present]
    into ONE chunk; cells row-major, TABLE_JOINER-joined.

    Text assembly stays row-major, but the group's emission position is
    the earliest ORIGINAL ingestion position of its members (independent
    of content sorting), so out-of-order cells cannot shift indices.
    """
    groups = {}
    order = []
    for el in document.elements:
        if _element_kind(el) != "table_cell":
            continue
        key = _table_group_key(el)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(el)
    plans = []
    for key in order:
        cells = sorted(groups[key],
                       key=lambda e: ((e.row if e.row is not None else 0),
                                      (e.column if e.column is not None else 0)))
        first = min(cells, key=lambda c: positions[c.element_id])
        pieces = [(c.element_id, _verbatim(c)) for c in cells]
        text = TABLE_JOINER.join(piece for _, piece in pieces)
        page, _, sheet = key
        plans.append({
            "kind": "table", "text": text, "page": page,
            "sheet": sheet, "section": None,
            "element_ids": [c.element_id for c in cells],
            "spans": _spans_for_joined(pieces),
            "_first_pos": positions[first.element_id],
            "_first_id": first.element_id,
        })
    return plans


def _plan_sheets(document, positions):
    """SHEET_CELL grouped by (sheet, row) into ONE chunk; header row 1
    supplies names; lines are "header: value" PAIR_JOINER-joined.
    The header row itself is not chunked (it supplies names only).
    Emission position is the earliest ORIGINAL ingestion position of the
    row's members (independent of column sorting)."""
    by_sheet_row = {}
    headers = {}
    for el in document.elements:
        if _element_kind(el) != "sheet_cell":
            continue
        if el.row == SHEET_HEADER_ROW:
            headers.setdefault((el.sheet, el.column), _verbatim(el))
        else:
            by_sheet_row.setdefault((el.sheet, el.row), []).append(el)
    plans = []
    for (sheet, _row), cells in by_sheet_row.items():
        cells = sorted(cells, key=lambda e: e.column
                       if e.column is not None else 0)
        first = min(cells, key=lambda c: positions[c.element_id])
        pairs = []
        spans = []
        offset = 0
        for i, cell in enumerate(cells):
            header = headers.get((cell.sheet, cell.column), "")
            value = _verbatim(cell)
            pair = f"{header}: {value}"
            if i:
                offset += len(PAIR_JOINER)
            value_start = offset + len(header) + 2
            spans.append({"element_id": cell.element_id,
                          "start": value_start,
                          "end": value_start + len(value)})
            pairs.append(pair)
            offset += len(pair)
        plans.append({
            "kind": "sheet-row", "text": PAIR_JOINER.join(pairs),
            "page": None, "sheet": sheet, "section": None,
            "element_ids": [c.element_id for c in cells],
            "spans": spans,
            "_first_pos": positions[first.element_id],
            "_first_id": first.element_id,
        })
    plans.sort(key=lambda p: p["_first_pos"])
    return plans


def _plan_csv(document):
    """CSV_ROW: one chunk per row; "header: value" pairs PAIR_JOINER-joined
    in mapping order."""
    plans = []
    for el in document.elements:
        if _element_kind(el) != "csv_row":
            continue
        mapping = el.value if isinstance(el.value, dict) else {}
        pairs = [f"{k}: {(v if v is not None else '')}"
                 for k, v in mapping.items()]
        text = PAIR_JOINER.join(pairs)
        plans.append({
            "kind": "csv-row", "text": text, "page": None,
            "sheet": None, "section": None,
            "element_ids": [el.element_id],
            "spans": [{"element_id": el.element_id, "start": 0,
                       "end": len(text)}],
        })
    return plans


def _plan_ocr(document, ocr_text):
    """OCR texts: ONE chunk per scanned document (kind ocr-text)."""
    if ocr_text is None:
        return []
    if not isinstance(ocr_text, str):
        raise TypeError(
            f"ocr_text must be str or None, got {type(ocr_text).__name__}")
    if document.document_id not in OCR_DOC_PAGES:
        raise TypeError(
            f"OCR chunk not allowed for {document.document_id!r}: "
            "allowlist is DEV-004 + DEV-010 only")
    return [{
        "kind": "ocr-text", "text": ocr_text,
        "page": OCR_DOC_PAGES[document.document_id],
        "sheet": None, "section": None,
        "element_ids": [], "spans": [],
    }]


# ---------------------------------------------------------------------------
# Public chunking API
# ---------------------------------------------------------------------------
def chunk_document(document, ocr_text=None):
    """Chunk one ingested Document. Returns chunk-record dicts.

    Raises TypeError for non-Document inputs; empty element lists yield
    zero chunks (never an error). Containers (TABLE, TABLE_ROW) and
    IMAGE_REF carry no text and are skipped; charts are excluded.
    Per-doc sequence ``index`` follows ingestion order (grouped chunks
    take the position of their first member; the OCR chunk is last).
    """
    if not isinstance(document, Document):
        raise TypeError(
            f"chunk_document expects ingestion.model.Document, "
            f"got {type(document).__name__}")
    base = _doc_base(document)
    # Earliest-ingestion position per element_id (O(n), single pass) so
    # grouped chunks emit at their first member's original position.
    positions = {}
    for pos, el in enumerate(document.elements):
        positions.setdefault(el.element_id, pos)
    # Emission order = ingestion order: walk elements once, flushing the
    # planned single/group chunks at each member's first occurrence.
    singles = {}
    for plan in _plan_text_heading(document) + _plan_csv(document):
        singles[plan["element_ids"][0]] = plan
    group_plans = {}
    for plan in (_plan_tables(document, positions)
                 + _plan_sheets(document, positions)):
        group_plans[plan["_first_id"]] = plan
    seen_groups = set()
    ordered = []
    for el in document.elements:
        if el.element_id in singles:
            ordered.append(singles[el.element_id])
        elif (el.element_id in group_plans
                and el.element_id not in seen_groups):
            seen_groups.add(el.element_id)
            ordered.append(group_plans[el.element_id])
    ordered.extend(_plan_ocr(document, ocr_text))
    chunks = []
    for index, plan in enumerate(ordered):
        plan = dict(plan)
        plan.pop("_first_pos", None)
        plan.pop("_first_id", None)
        chunks.append(make_chunk_record(
            document.document_id, plan["kind"], index, plan["text"],
            filename=base["filename"], file_type=base["file_type"],
            language=base["language"], page=plan["page"],
            sheet=plan["sheet"], section=plan["section"],
            element_ids=plan["element_ids"], spans=plan["spans"]))
    return chunks


def chunk_corpus(documents, ocr_texts=None):
    """Chunk an ordered corpus of Documents. ocr_texts maps
    document_id -> OCR text (honored for allowlisted docs only)."""
    if not isinstance(documents, (list, tuple)):
        raise TypeError(
            f"chunk_corpus expects a list of Document, "
            f"got {type(documents).__name__}")
    ocr_texts = ocr_texts or {}
    chunks = []
    for document in documents:
        if not isinstance(document, Document):
            raise TypeError(
                f"chunk_corpus expects Document items, "
                f"got {type(document).__name__}")
        chunks.append(chunk_document(
            document, ocr_text=ocr_texts.get(document.document_id)))
    return [chunk for per_doc in chunks for chunk in per_doc]


# ---------------------------------------------------------------------------
# OCR sidecar + index build (design section 8)
# ---------------------------------------------------------------------------
def load_ocr_texts(ocr_path):
    """Load {document_id: text} from a paddle_results.json sidecar.

    Only DEV-004 + DEV-010 entries with non-empty text are returned.
    """
    ocr_path = Path(ocr_path)
    records = json.loads(ocr_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise TypeError("paddle results sidecar must be a JSON list")
    texts = {}
    for record in records:
        doc_id = record.get("document_id")
        text = record.get("text")
        if doc_id in OCR_DOC_PAGES and isinstance(text, str) and text:
            texts[doc_id] = text
    return texts


def ocr_fingerprint_for(ocr_texts):
    """Stable fingerprint of the SELECTED OCR texts ({doc_id: text}).

    A missing/unreadable sidecar yields ``ocr_texts == {}`` upstream, so
    it fingerprints as the empty-selection marker ``sha256("{}")`` —
    identical to a sidecar that selects nothing, since both contribute
    zero OCR bytes to the corpus. Deterministic either way.
    """
    canonical = json.dumps(
        {doc_id: ocr_texts[doc_id] for doc_id in sorted(ocr_texts)},
        sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_index(docs_dir=None, ocr_path=None, out_dir=None):
    """Ingest the DEV documents via ingestion.service and write the R1 index.

    Writes ``chunks.json`` + ``index_meta.json`` under ``out_dir``
    (makedirs as needed). R1 carries no vectors: model fields in meta are
    marked pending. Unreadable/unparsable files surface as ingestion
    controlled-failures: the doc is skipped with a warning entry and the
    build continues. Returns a summary dict.
    """
    docs_dir = Path(docs_dir) if docs_dir is not None else DEFAULT_DOCS_DIR
    ocr_path = Path(ocr_path) if ocr_path is not None else DEFAULT_OCR_PATH
    out_dir = Path(out_dir) if out_dir is not None else DEFAULT_OUT_DIR
    if not docs_dir.is_dir():
        raise NotADirectoryError(f"not a directory: {docs_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    files = sorted((p for p in docs_dir.iterdir() if p.is_file()),
                   key=lambda p: p.name)
    documents = []
    file_hashes = {}
    skipped_files = []
    warnings = []
    for path in files:
        result = ingest_document(path)
        if not result.ok:
            skipped_files.append(path.name)
            warnings.append({"file": path.name,
                             "warning": f"ingestion failed: {result.error}"})
            continue
        documents.append(result.document)
        file_hashes[result.document.document_id] = _file_sha256(path)
    ingest_ms = (time.perf_counter() - started) * 1000.0

    try:
        ocr_texts = load_ocr_texts(ocr_path)
    except (OSError, ValueError, TypeError) as exc:
        warnings.append({"file": Path(ocr_path).name,
                         "warning": f"ocr sidecar unavailable: {exc}"})
        ocr_texts = {}

    chunk_started = time.perf_counter()
    chunks = chunk_corpus(documents, ocr_texts=ocr_texts)
    chunk_ms = (time.perf_counter() - chunk_started) * 1000.0

    write_started = time.perf_counter()
    chunks_path = out_dir / "chunks.json"
    meta_path = out_dir / "index_meta.json"
    chunks_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    by_kind = {}
    by_doc = {}
    for chunk in chunks:
        by_kind[chunk["kind"]] = by_kind.get(chunk["kind"], 0) + 1
        by_doc[chunk["document_id"]] = by_doc.get(chunk["document_id"], 0) + 1
    manifest = [{"document_id": doc_id, "filename": name,
                 "sha256": file_hashes[doc_id]}
                for doc_id, name in sorted(
                    ((d.document_id, d.filename) for d in documents))]
    ocr_fingerprint = ocr_fingerprint_for(ocr_texts)
    manifest_sha256 = hashlib.sha256(json.dumps(
        {"files": manifest, "ocr_texts": {doc_id: ocr_texts[doc_id]
                                          for doc_id in sorted(ocr_texts)}},
        sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")).hexdigest()
    meta = {
        "chunker": {
            "name": CHUNKER_NAME,
            "version": CHUNKER_VERSION,
            "params": {
                "table_group_by": ["page", "table_index", "sheet"],
                "table_joiner": TABLE_JOINER,
                "sheet_group_by": ["sheet", "row"],
                "sheet_header_row": SHEET_HEADER_ROW,
                "csv_pair_joiner": PAIR_JOINER,
                "pair_format": "{header}: {value}",
                "chunk_id": "CHK-<12hex sha256 of canonical JSON "
                            "{document_id,kind,index,text}>",
                "storage": "verbatim (normalization applies at "
                           "tokenization only)",
            },
        },
        "embedding_model": {
            "id": "pending",
            "revision": "pending",
            "note": "R1 text-only: no vectors; embeddings are a later package",
        },
        "corpus": {
            "docs_dir": str(docs_dir),
            "document_ids": sorted(d.document_id for d in documents),
            "ocr_path": str(ocr_path),
            "ocr_fingerprint": ocr_fingerprint,
            "manifest_sha256": manifest_sha256,
        },
        "counts": {
            "chunks_total": len(chunks),
            "by_kind": by_kind,
            "by_doc": by_doc,
            "docs_ingested": len(documents),
            "docs_skipped": sorted(skipped_files),
        },
        "warnings": warnings,
        "timings_ms": {
            "ingest": ingest_ms,
            "chunk": chunk_ms,
            "write": (time.perf_counter() - write_started) * 1000.0,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return {
        "chunks": chunks,
        "meta": meta,
        "chunks_path": str(chunks_path),
        "meta_path": str(meta_path),
        "counts_by_doc": by_doc,
        "excluded": sorted(d.document_id for d in documents
                           if by_doc.get(d.document_id, 0) == 0),
        "warnings": warnings,
    }
