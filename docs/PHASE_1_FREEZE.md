# Phase 1 Freeze Record — Native Document Ingestion

> What Phase 1 built, what it proved, and where it stops. Generated ingestion
> output is git-ignored and reproducible; THIS FILE is tracked. No
> ground-truth answers are copied into this document.

- **Purpose:** route supported files to deterministic native parsers and emit
  one normalized Document/Element representation with provenance — answering
  only "what does this file natively contain, and is that enough?"
- **Supported native formats:** PDF, DOCX, XLSX, CSV, PNG, JPG/JPEG (magic
  bytes + extension; signature wins; unknown → controlled UNSUPPORTED).
- **Normalized representation:** stdlib dataclasses — `Document` (stable id,
  filename, type, language hint, page/sheet counts, status, ocr/visual
  flags, metadata, elements, error, latency) + flat `Element` (deterministic
  id, 8-type closed set, page/sheet/row/column/table coordinates, verbatim
  text or native value). JSON-serializable for debugging.
- **Parser choices:** pypdf page text + XObject image scan; pdfplumber
  **lattice-only** tables (APPROVED; unruled tables deferred, never guessed);
  python-docx (paras/headings/tables/RTL flags); openpyxl read-only native
  values (formulas recorded, never evaluated); stdlib csv (utf-8-sig);
  Pillow metadata-only for images. PyMuPDF evaluated and rejected. Zero new
  dependencies.
- **Status definitions:** `native_ok` (all units sufficient; PDF page ≥ 50
  chars) · `partial_native` (mixed; ocr flag) · `requires_ocr`
  (scanned-style, no false success) · `requires_visual` (rasters, metadata
  only) · `failed` (controlled, with error) · `unsupported` (router-level).
- **Frozen DEV results (dataset_v0.1 documents/):** 001/002/003/005/006/007
  `native_ok` (001: 1 lattice table; 002: 2; 003: 3; 005: 11 paras/5 RTL;
  006: 48,493 native cells; 007: 346 rows) · 004/008/009 `requires_visual`
  (dims verified) · 010 `requires_ocr` (0 elements, correctly escalated).
- **Latency baseline** (`python scripts/ingest_dev_corpus.py`, 10 docs):
  QA run 1: total 2661.5 ms (wall 2662.7 ms) · QA run 2 (fresh process):
  total 8475.0 ms — cross-process wall time is machine-noise dominated, so
  single totals are baselines, not promises. In-process split isolates the
  real effects: DEV-001 456 ms cold (one-off pdfplumber import ~380 ms) →
  ~75 ms warm; DEV-006 ~1.5–3.0 s genuine XLSX parse cost (48,493 cells);
  all others < 0.3 s each. No optimization warranted.
- **Tests at freeze:** 56 passed (33 Phase-0 + 23 Phase-1), 0 failed —
  routing, signatures, deterministic IDs, per-format parsers, Arabic
  preservation, provenance, serialization, directory isolation, GT-anchored
  extraction, banned OCR/AI-import guard. Plus 31-check independent exit
  probe (evaluation-side script, not committed): all PASS.
- **Known limitations:** lattice-only PDF tables; reshaped-Arabic digit runs
  need bidi-isolated layouts (frozen corpus complies); XLSX equality via
  `Decimal(str(cell))`; CSV full-read; heuristic language hint; failure
  messages name files, not stack traces, by design.
- **Explicitly deferred (NOT Phase 1):** OCR, vision models, embeddings, RAG,
  Qdrant, BM25, reranking, answer generation, forecasting, Streamlit —
  nothing from that list was implemented or installed.
- **Baseline command:** `python scripts/ingest_dev_corpus.py`
  (reads ONLY `data/dev/dataset_v0.1/documents/`; writes git-ignored
  `data/ingestion/dataset_v0.1/` + prints the table).

**OCR/vision interpretation begins in Phase 2**, consuming exactly the
`requires_ocr` / `requires_visual` flags and per-page image metadata
produced here.
