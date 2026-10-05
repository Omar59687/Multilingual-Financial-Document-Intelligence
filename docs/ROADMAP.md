# Roadmap — MizanIQ
AI-Powered Finance Document Intelligence System

> Development phases.
> Related docs: [PROJECT_SPEC.md](PROJECT_SPEC.md) · [ARCHITECTURE.md](ARCHITECTURE.md) · [DEVELOPMENT_RULES.md](DEVELOPMENT_RULES.md) · [DATASET_DESIGN.md](DATASET_DESIGN.md) · [CANONICAL_DATA_MODEL.md](CANONICAL_DATA_MODEL.md) · [EVALUATION_PLAN.md](EVALUATION_PLAN.md)

## Current Status

- Current phase: **Phase 3 — Structured Financial Extraction: DONE**
  (architect gate 2026-10-04; evidence `docs/G1_G4_BASELINE.md`)
- Completed phases: **Phase 0**, **Phase 1**, **Phase 2**, and **Phase 3**
- Phase 3 foundation: **COMPLETE**
- Phase 3A: **COMPLETE**
- Phase 3 persistence / routes / query library / handoff: **COMPLETE**
- G1/G4 baselines: **G1 PASS (Option B: P 1.000 ≥ 0.99, R 0.904 ≥ 0.90)**;
  **G4 PASS (7/7)**; Arabic/visual/prose follow-ups mandatory before G6
- Next phase: **Phase 4 — Text RAG (not started)**
- Phase freeze/exit evidence: [DATASET_V0_1_FREEZE.md](DATASET_V0_1_FREEZE.md),
  [PHASE_1_FREEZE.md](PHASE_1_FREEZE.md), and
  [PHASE_2_OCR_VISION_EXIT.md](PHASE_2_OCR_VISION_EXIT.md)
- Dataset design status: FROZEN for dataset_v0.1 — [DATASET_DESIGN.md](DATASET_DESIGN.md) + [DEV_DOCUMENT_SPEC.md](DEV_DOCUMENT_SPEC.md) (v0.2+ changes need a new dataset version)
- Canonical model status: FROZEN for dataset_v0.1 — [CANONICAL_DATA_MODEL.md](CANONICAL_DATA_MODEL.md)
- Canonical dataset status: IMPLEMENTED — deterministic generator + dataset_v0.1 truth tables validated (seed 42)
- DEV document status: IMPLEMENTED — 10 DEV documents + ground truth generated and validated per [DEV_DOCUMENT_SPEC.md](DEV_DOCUMENT_SPEC.md)
- Evaluation plan status: PROCEDURE-FROZEN, thresholds TBD — [EVALUATION_PLAN.md](EVALUATION_PLAN.md) (thresholds await baseline experiments; explicitly not a Phase 0 blocker)

## Phase 0 — Foundation and Dataset Design

Status: **COMPLETE** — freeze record [DATASET_V0_1_FREEZE.md](DATASET_V0_1_FREEZE.md).

Goals:

- establish project documentation
- inspect/select source datasets
- define the simulated/client-style document dataset
- create approximately 10 representative development documents
- define evaluation strategy
- finalize initial schemas

No full 150-document processing yet.

## Phase 1 — Native Document Ingestion

Status: **COMPLETE** — freeze record [PHASE_1_FREEZE.md](PHASE_1_FREEZE.md).

Support basic reading of:

- PDF
- DOCX
- XLSX
- CSV
- common images

Focus first on normal/native extraction.

Build metadata/manifests.

## Phase 2 — OCR and Visual Document Understanding

Status: **COMPLETE** — exit record
[PHASE_2_OCR_VISION_EXIT.md](PHASE_2_OCR_VISION_EXIT.md).

Add support for:

- scanned documents
- Arabic OCR
- English OCR
- mixed-language pages
- difficult tables
- charts/images

Benchmark candidate OCR/document-understanding models.

GPU experimentation may happen in Kaggle.

Do not select large models blindly.

## Phase 3 — Structured Financial Extraction

Status: **DONE** — foundation, Phase 3A, DuckDB persistence,
extraction routes (native tabular / statement tables, Paddle grids,
Qwen pairs), calculation-query library, and forecasting handoff
complete (360 tests); G1 PASS under Option B, G4 7/7 PASS
(`docs/G1_G4_BASELINE.md` + §9 gate resolution 2026-10-04). Arabic/
visual/prose follow-ups mandatory before G6. See
[STRUCTURED_EXTRACTION_DESIGN.md](STRUCTURED_EXTRACTION_DESIGN.md).

Define Pydantic models.

Extract normalized records such as:

- metric
- date/year
- value
- currency
- source document
- source page

Validate all AI-generated structured data.

Store approved financial records in DuckDB.

## Phase 4 — Text RAG

Implement:

- chunking
- multilingual embeddings
- metadata
- vector retrieval
- BM25
- hybrid retrieval

Build an evaluation question set.

Measure retrieval quality objectively.

## Phase 5 — Reranking and Evidence Building

Add reranking.

Build evidence packages.

Measure whether reranking improves retrieval results.

## Phase 6 — Visual Retrieval

Add visual document/page retrieval for:

- charts
- scanned pages
- images
- visually structured documents

Use a VLM only when necessary.

## Phase 7 — Question Router and Answer Generation

Route user questions to the correct source(s).

Support combination questions.

Generate grounded Arabic/English answers.

Add citations.

Add insufficient-evidence behavior.

## Phase 8 — Forecasting

Create time series from validated DuckDB data.

Start with baselines.

Evaluate statistical models such as:

- seasonal naive where relevant
- exponential smoothing
- SARIMA where appropriate

Use holdout/backtesting.

Use appropriate forecasting metrics.

Do not select a forecasting model merely because it is more complex.

## Phase 9 — Streamlit Application

Build the user interface.

Planned sections:

- Documents
- Ask
- Analyze
- Forecast

## Phase 10 — Scaling

Only after the pipeline works reliably on the development set:

- scale toward approximately 150 files
- add batch processing
- retries
- checkpointing
- incremental ingestion
- parallelization where safe

Modal may later be used for repeatable GPU workloads/model serving if needed.

## Phase 11 — Final Evaluation and Portfolio Polish

Measure:

- extraction accuracy
- retrieval performance
- grounded-answer quality
- citation correctness
- hallucination rate
- forecasting error
- latency
- reliability

Produce:

- architecture diagram
- benchmark results
- final README
- reproducible setup
- demo scenario
