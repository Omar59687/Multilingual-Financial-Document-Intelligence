# MizanIQ
AI-Powered Finance Document Intelligence System

> Portfolio-quality AI project for multilingual (Arabic/English) financial document understanding, grounded RAG with citations, exact structured calculations, visual evidence retrieval, and time-series forecasting.

## Current Status

- Current phase: **Phase 0 — Foundation and Dataset Design**
- Implementation status: AI/data pipeline has not started yet.
- Development is incremental: the pipeline must prove itself on ~10 representative documents before scaling toward ~150 files / ~10 years.

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

Intentionally NOT contained: PDF/DOCX/XLSX documents, scans/images,
DEV-001…010 files, RAG chunks/embeddings, Qdrant data, Streamlit UI,
AI model calls. See `docs/CANONICAL_DATA_MODEL.md`.

## Documentation

Authoritative docs (read these before contributing):

- [docs/PROJECT_SPEC.md](docs/PROJECT_SPEC.md) — WHAT we are building
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — HOW the system works (text + visual + DuckDB, router, reranking, forecasting)
- [docs/ROADMAP.md](docs/ROADMAP.md) — development phases (Phase 0 → Phase 11) and current status
- [docs/DEVELOPMENT_RULES.md](docs/DEVELOPMENT_RULES.md) — mandatory rules for coding agents
- [docs/DATASET_DESIGN.md](docs/DATASET_DESIGN.md) — dataset design (DRAFT, pending review)
- [docs/CANONICAL_DATA_MODEL.md](docs/CANONICAL_DATA_MODEL.md) — canonical truth model (DRAFT, pending review)
- [docs/EVALUATION_PLAN.md](docs/EVALUATION_PLAN.md) — evaluation contract (DRAFT, thresholds TBD)
