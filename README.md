# MizanIQ
AI-Powered Finance Document Intelligence System

> Portfolio-quality AI project for multilingual (Arabic/English) financial document understanding, grounded RAG with citations, exact structured calculations, visual evidence retrieval, and time-series forecasting.

## Current Status

- Current phase: **Phase 3 — Structured Extraction: DONE** (architect gate 2026-10-04: G1 PASS under Option B, P 1.000 / R 0.904; G4 7/7 PASS).
- Completed phases: **Phase 0**, **Phase 1**, **Phase 2**, and **Phase 3**.
- Phase 3 delivered: foundation, Phase 3A, DuckDB persistence, all four
  extraction routes (native tabular, native statement tables, Paddle
  scanned grids, Qwen visual pairs), calculation-query library,
  forecasting handoff (360 tests; G1/G4 baselines in
  `docs/G1_G4_BASELINE.md`). Arabic/visual/prose limitations remain
  mandatory follow-ups before G6 — the Arabic slice (0.200) is explicitly
  not production-quality.
- Next phase: **Phase 4 — Text RAG** (not started).
- Development is incremental: the pipeline must prove itself on ~10 representative documents before scaling toward ~150 files / ~10 years.

## Phase-1 native ingestion

Deterministic parsers (PDF/DOCX/XLSX/CSV/PNG/JPG) producing one normalized
representation with source provenance, native-vs-OCR classification, and
controlled failures. No OCR/AI/RAG in this phase. Run the baseline locally —
output is git-ignored:

```text
python scripts/ingest_dev_corpus.py
python -m pytest tests/ -q
```

Produces `data/ingestion/dataset_v0.1/`: one normalized JSON per DEV document
plus `baseline_summary.json`. Design: `docs/INGESTION_DESIGN.md`. Expected
DEV behavior: 001/002/003/005/006/007 native-ok; 004/008/009 visual-only;
010 OCR-required.

## Canonical dataset_v0.1 (Phase 0 implementation)

Deterministic structured business truth for Noor Retail & Distribution Co.
(2015–2024, SAR, seed 42). Generate locally — output is git-ignored:

```text
pip install -r requirements.txt
python scripts/generate_canonical_data.py
python -m pytest tests/ -q
```

Produces `data/canonical/dataset_v0.1/`: `branches.csv`, `departments.csv`,
`monthly_financials.csv` (312 rows), `transactions.csv` (~35.9k),
`budgets.csv` (quarterly, budget truth only), `invoices.csv` (2,400),
`business_events.csv` (10), `annual_balance_sheet.csv` (10) plus
`manifest.json` (seed, counts, hashes, simplification notes).

Intentionally NOT contained: RAG chunks/embeddings, Qdrant data, Streamlit UI,
AI model calls. See `docs/CANONICAL_DATA_MODEL.md`.

## DEV document set v0.1 (Phase 0 implementation)

The 10 representative development documents (`DEV-001…DEV-010`) plus
machine-readable ground truth, generated deterministically from canonical
truth. Generate locally — output is git-ignored:

```text
python scripts/generate_dev_documents.py
python -m pytest tests/ -q
```

Produces `data/dev/dataset_v0.1/`: `documents/` (the 10 DEV files ONLY —
the future ingestion corpus), `ground_truth/` (per-document JSON for
evaluation tooling, never ingested), `manifest.json` (hashes, canonical
pointer, no-leakage rule). Spec: `docs/DEV_DOCUMENT_SPEC.md`.

Coverage: EN native PDF, Arabic PDF, mixed invoice PDF, scanned Arabic
receipt PNG, mixed DOCX commentary, XLSX + CSV extracts, revenue chart PNG,
KPI dashboard JPG, scanned balance-sheet PDF.

## Documentation

Authoritative docs (read these before contributing):

- [docs/PROJECT_SPEC.md](docs/PROJECT_SPEC.md) — WHAT we are building
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — HOW the system works (text + visual + DuckDB, router, reranking, forecasting)
- [docs/ROADMAP.md](docs/ROADMAP.md) — development phases (Phase 0 → Phase 11) and current status
- [docs/DEVELOPMENT_RULES.md](docs/DEVELOPMENT_RULES.md) — mandatory rules for coding agents
- [docs/DATASET_DESIGN.md](docs/DATASET_DESIGN.md) — dataset design (DRAFT, pending review)
- [docs/CANONICAL_DATA_MODEL.md](docs/CANONICAL_DATA_MODEL.md) — canonical truth model (DRAFT, pending review)
 - [docs/EVALUATION_PLAN.md](docs/EVALUATION_PLAN.md) — evaluation contract (DRAFT, thresholds TBD)
 - [docs/DEV_DOCUMENT_SPEC.md](docs/DEV_DOCUMENT_SPEC.md) — DEV-001…010 document/ground-truth specification (ACTIVE for dataset_v0.1)
- [docs/DATASET_V0_1_FREEZE.md](docs/DATASET_V0_1_FREEZE.md) — Phase 0 freeze record
- [docs/INGESTION_DESIGN.md](docs/INGESTION_DESIGN.md) — completed Phase 1 native-ingestion design; see [docs/PHASE_1_FREEZE.md](docs/PHASE_1_FREEZE.md)
