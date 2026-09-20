# Ingestion Design — MizanIQ Phase 1 (Native Document Ingestion)

> How native files become a normalized representation. No OCR, no embeddings,
> no retrieval here — this layer answers only "what does this file natively
> contain, and is that enough?"
> Related docs: [ARCHITECTURE.md](ARCHITECTURE.md) (ingestion box) ·
> [DEV_DOCUMENT_SPEC.md](DEV_DOCUMENT_SPEC.md) · [DATASET_V0_1_FREEZE.md](DATASET_V0_1_FREEZE.md)

## 1. Normalized representation (`src/ingestion/model.py`)

One model for all formats, built on stdlib **dataclasses** (Pydantic would add
a dependency for no Phase-1 benefit; dataclasses + `to_dict()` cover typed
construction and JSON debugging output).

**Document** — `document_id, filename, file_type, language_hint, page_count,
sheet_count, status, requires_ocr, requires_visual, metadata, elements,
error, latency_ms`. Exactly one of `page_count` / `sheet_count` is set,
depending on format (both `None` for CSV/images).

**Element** — flat (no nesting): `element_id, element_type, page, sheet, row,
column, table_index, text, value, data_type, meta`. Unused coordinates stay
`None`, so every element keeps the same shape for future chunking/citation
code. Element types (closed set, 8): `text, heading, table, table_row,
table_cell, sheet_cell, csv_row, image_ref`.
`table` carries shape in `meta` (`n_rows`, `n_cols`); cells carry
`table_index/row/column`; `sheet_cell` carries native `value + data_type`
(numbers stay numbers — never formatted display strings); `csv_row` carries
`row_number` + full field `mapping`.

## 2. Router (`router.py`)

Extension map (pdf/docx/xlsx/csv/png/jpg/jpeg) **plus magic-byte check**:
`%PDF`, ZIP (`PK\x03\x04`, disambiguated via `[Content_Types].xml`:
wordprocessingml → docx, spreadsheetml → xlsx), PNG (`\x89PNG`), JPEG
(`\xFF\xD8\xFF`). Signature wins over extension; unknown → controlled
`UNSUPPORTED` (never an exception). CSV: `.csv` extension + UTF-8(S) decodable
sample (BOM stripped via `utf-8-sig`).

## 3. Parsers (`parsers.py`) — one function per format, same return contract

- **PDF** (`parse_pdf`, pypdf): per-page text + `has_images` (XObject scan).
  Tables: **pdfplumber, lattice strategy only** — our native PDFs use ruled
  grids, where lattice is reliable; no text-strategy guessing. If lattice
  finds nothing, page text is preserved and table recovery is honestly
  deferred (`tables_found: 0`, no fabricated structure).
- **DOCX** (python-docx): paragraph order + text, headings (`Heading*`
  styles), tables (cell text + coordinates), inline-image count
  (`image_ref` elements only as existence markers — no OCR).
- **XLSX** (openpyxl, read-only): sheet names, dimensions, per-cell
  coordinate/value/`data_type`. Formulas: recorded as `data_type: "f"` with
  the formula string in `meta` and no invented value. No LLM math, ever.
- **CSV** (stdlib): `utf-8-sig` header + `csv_row` elements with 1-based
  `row_number`. Reader-based (streaming-capable); v0.1 collects the list.
- **Images** (Pillow): format/size/mode from bytes only. One `image_ref`
  element for the whole frame. Content NEVER interpreted here.

## 4. Status model (closed enum, explicit definitions)

- `NATIVE_OK` — every page/sheet yielded meaningful native content
  (PDF page threshold: ≥ 50 stripped chars; sheets/CSV: ≥ 1 data row).
- `PARTIAL_NATIVE` — some pages natively fine, others image-only
  (per-page detail in `metadata.pages`). `requires_ocr = true`.
- `REQUIRES_OCR` — scanned-style PDF: no page meets the threshold AND at
  least one page is image-dominant. Parser does NOT pretend success.
- `REQUIRES_VISUAL` — raster image files (PNG/JPG): metadata only.
- `FAILED` — missing/unreadable/corrupt file or parser exception, with
  `error` set and partial elements preserved where available.
- `UNSUPPORTED` — unknown type; routing-level, no parser invoked.

`requires_ocr` is true for `PARTIAL_NATIVE`/`REQUIRES_OCR`; `requires_visual`
is true for `REQUIRES_VISUAL` (and, as a hint, for image-dominant PDF pages
recorded in page metadata). Expected DEV mapping: 001/002/003 NATIVE_OK ·
004/008/009 REQUIRES_VISUAL · 005/006/007 NATIVE_OK · 010 REQUIRES_OCR.

## 5. OCR/visual boundary

Phase 1 **detects insufficiency, never remedies it**. Anything with
`requires_ocr`/`requires_visual` is complete output for this phase; Phase 2
consumes exactly these flags. No OCR/AI/vision packages are introduced
(dependency check at freeze must show none).

## 6. Provenance + IDs (deterministic, no UUIDs)

Every element ID derives from its document: `{doc_id}:p{page}:{type}:{n}`
(PDF/DOCX), `{doc_id}:{sheet}:r{row}c{col}` (XLSX),
`{doc_id}:row{n}` (CSV), `{doc_id}:img{n}` (images).
Document IDs: frozen DEV files use their `DEV-\d{3}` filename prefix
(`DEV-001`…); general files use `DOC-` + 12 hex chars of the file-content
SHA-256 (content-addressed, stable across moves/copies). Same bytes →
same IDs, every run.

## 7. Language hint (explicitly NOT detection)

Script-ratio heuristic over extracted text: Arabic-block share vs
Latin-letter share → `ar` / `en` / `mixed` (both ≥ 20%) / `unknown`
(negligible text). Preserved verbatim text is the contract; the hint only
routes future work. Documented thresholds live in `parsers.language_hint`.

## 8. Service (`service.py`)

`ingest_document(path)` → one `IngestedDocument` (dataclass: `document` +
`ok` + `error` + `latency_ms`; never raises for domain failures — missing
file, unsupported type, corrupt bytes, parser exceptions all become
controlled `FAILED`/`UNSUPPORTED` results). `ingest_directory(path)` →
sorted discovery of supported files, per-file isolation (one bad file
cannot fail the run), `.json` ground-truth sidecars and `ground_truth/`
dirs skipped (never ingested), deterministic ordering, summary counts.

## 9. Serialization + debugging output

`to_dict()` / `to_json()` on the dataclasses (enums as values, tuples as
lists). `scripts/ingest_dev_corpus.py` ingests `data/dev/dataset_v0.1/
documents/`, writes normalized JSON per DEV file under
`data/ingestion/` (git-ignored), and prints the Phase-1 baseline table
(parser, status, elements, chars, tables/rows, latency, flags).

## 10. Dependency evaluations (recorded decisions)

- **PyMuPDF: NOT added.** pypdf covers Phase-1 needs (page text, page count,
  image-presence via `/XObject`); pdfium (already pinned) covers any future
  rasterization. Revisit only with a concrete gap.
- **pdfplumber (0.11.9, pre-installed): justified** — lattice table
  extraction on ruled native PDFs with full provenance
  (file/page/table-index/rows/cells). Text-strategy tables deliberately OUT
  of scope (deferred, not guessed).
- Nothing else added: stdlib csv/dataclasses/hashlib/mimetypes,
  python-docx, openpyxl, Pillow, pypdf. NumPy/pandas/matplotlib environment
  issues untouched (Phase 1 does not need them).

## 11. Known Phase-1 limitations

- PDF tables require ruling lines (lattice); unruled native tables yield
  page text only (deferred to later phases).
- Reshaped-Arabic PDFs: digit/Latin runs inside RTL paragraphs are
  invisible to naive extractors (frozen v0.1 bidi-isolation keeps such runs
  in standalone LTR lines/cells — see DEV_DOCUMENT_SPEC §0); table digit
  cells extract exactly.
- XLSX floats: equality with canonical Decimals holds via
  `Decimal(str(cell))` (same convention as the DEV generator).
- CSV: full-file read (no chunked streaming yet; reader-based design allows
  it later). No multi-GB hardening.
- Language hint is heuristic; mixed-threshold (20%) is a starting point,
  not a tuned classifier.
