# Roadmap — MizanIQ
AI-Powered Finance Document Intelligence System

> Development phases.
> Related docs: [PROJECT_SPEC.md](PROJECT_SPEC.md) · [ARCHITECTURE.md](ARCHITECTURE.md) · [DEVELOPMENT_RULES.md](DEVELOPMENT_RULES.md) · [DATASET_DESIGN.md](DATASET_DESIGN.md) · [CANONICAL_DATA_MODEL.md](CANONICAL_DATA_MODEL.md) · [EVALUATION_PLAN.md](EVALUATION_PLAN.md)

## Current Status

- Current phase: **Phase 0 — COMPLETE** (freeze record:
  [DATASET_V0_1_FREEZE.md](DATASET_V0_1_FREEZE.md), 2026-09-20; 33/33 tests
  passing; all 14 exit criteria met)
- Next phase: Phase 1 — Native Document Ingestion
- Current task: Native parser foundation and baseline evaluation
- Implementation status: Phase-1 native ingestion implemented
  (`src/ingestion/`, design in [INGESTION_DESIGN.md](INGESTION_DESIGN.md));
  OCR/RAG/forecasting/UI explicitly out of scope until later phases
- Dataset design status: FROZEN for dataset_v0.1 — [DATASET_DESIGN.md](DATASET_DESIGN.md) + [DEV_DOCUMENT_SPEC.md](DEV_DOCUMENT_SPEC.md) (v0.2+ changes need a new dataset version)
- Canonical model status: FROZEN for dataset_v0.1 — [CANONICAL_DATA_MODEL.md](CANONICAL_DATA_MODEL.md)
- Canonical dataset status: IMPLEMENTED — deterministic generator + dataset_v0.1 truth tables validated (seed 42)
- DEV document status: IMPLEMENTED — 10 DEV documents + ground truth generated and validated per [DEV_DOCUMENT_SPEC.md](DEV_DOCUMENT_SPEC.md)
- Evaluation plan status: PROCEDURE-FROZEN, thresholds TBD — [EVALUATION_PLAN.md](EVALUATION_PLAN.md) (thresholds await baseline experiments; explicitly not a Phase 0 blocker)

## Phase 0 — Foundation and Dataset Design

Goals:

- establish project documentation
- inspect/select source datasets
- define the simulated/client-style document dataset
- create approximately 10 representative development documents
- define evaluation strategy
- finalize initial schemas

No full 150-document processing yet.

## Phase 1 — Native Document Ingestion

Support basic reading of:

- PDF
- DOCX
- XLSX
- CSV
- common images

Focus first on normal/native extraction.

Build metadata/manifests.

## Phase 2 — OCR and Visual Document Understanding

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
