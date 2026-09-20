# MizanIQ
AI-Powered Finance Document Intelligence System

> Authoritative description of WHAT we are building.
> Related docs: [ARCHITECTURE.md](ARCHITECTURE.md) · [ROADMAP.md](ROADMAP.md) · [DEVELOPMENT_RULES.md](DEVELOPMENT_RULES.md) · [DATASET_DESIGN.md](DATASET_DESIGN.md) · [CANONICAL_DATA_MODEL.md](CANONICAL_DATA_MODEL.md) · [EVALUATION_PLAN.md](EVALUATION_PLAN.md)
>
> Current status: Phase 0 — Foundation and Dataset Design. Implementation has not started yet.

## 1. Project Goal

Build an AI-powered financial document intelligence system capable of processing approximately 150 documents spanning around 10 years.

The system must be capable of:

- ingesting heterogeneous financial documents
- understanding English documents
- understanding Arabic documents
- understanding mixed Arabic/English documents
- understanding documents containing images
- understanding scanned pages
- extracting tables
- understanding charts and financial visuals
- extracting structured financial values
- answering questions using RAG
- answering in Arabic or English depending on the user's language
- providing file/page citations
- performing exact financial calculations
- retrieving relevant visual evidence
- forecasting financial time-series values

This should be treated as a serious portfolio-quality AI/data engineering project, not a toy chatbot.

## 2. Expected Document Types

The architecture should eventually support:

- PDF
- DOCX
- XLSX
- CSV
- PNG
- JPG/JPEG

Documents may contain:

- digital/selectable text
- scanned text
- Arabic text
- English text
- mixed language
- tables
- charts
- invoices
- receipts
- accounting policies
- management commentary
- financial statements
- transactions

## 3. Dataset Scale

Target final dataset:

- Approximately 150 files
- Approximately 10 years of financial information

Development strategy:

DO NOT begin development using all 150 files.

Start with approximately 10 representative documents covering different cases such as:

- English digital PDF
- Arabic document
- mixed Arabic/English document
- scanned PDF
- document containing tables
- document containing charts/images
- Excel financial data
- CSV financial data
- Word document
- standalone image

The pipeline must prove itself on the small representative set before scaling.

## 4. Major Capabilities

### Document ingestion

Identify file format and metadata and send documents to the correct parser.

### Native document extraction

Use normal parsers whenever content can be accurately extracted without AI.

### OCR/document understanding

Use OCR/document-understanding models for scanned or visually complex content.

### Structured financial extraction

Convert financial information into validated structured records.

### Text retrieval

Use multilingual semantic retrieval.

### Keyword retrieval

Use keyword/exact retrieval for values such as invoice numbers, identifiers and exact terminology.

### Visual retrieval

Allow relevant pages, images, tables and charts to be retrieved when visual evidence is required.

### Structured analytics

Use a structured database for calculations instead of asking an LLM to calculate important financial values.

### Question routing

Determine whether a question requires:

- text retrieval
- keyword retrieval
- visual retrieval
- structured database calculations
- forecasting
- or a combination

### Reranking

Rerank retrieved evidence before sending it to the answer-generation model.

### Answer generation

Generate natural Arabic or English answers based only on retrieved/verified evidence.

### Citations

Answers must identify their supporting source, including file and page when available.

### Forecasting

Forecasting should use statistical/time-series methods rather than asking the LLM to invent predictions.

## 5. Quality Goals

This project must optimize for measurable quality, not only functionality.

We care about:

- document extraction accuracy
- OCR accuracy
- financial-field extraction accuracy
- retrieval recall
- retrieval precision
- reranking quality
- answer correctness
- answer grounding
- citation correctness
- hallucination rate
- structured calculation correctness
- forecasting accuracy
- response latency
- ingestion reliability
- multilingual performance

We will create evaluation datasets and benchmarks as the project develops.

## 6. Performance Goals

The system should NOT reread all 150 files for every question.

Documents are processed/indexed ahead of time.

A normal question should search prepared indexes/databases and provide an answer in a reasonable interactive AI application response time.

Exact latency targets will be benchmarked rather than guessed.

We should separately measure:

- retrieval latency
- reranking latency
- LLM generation latency
- visual question latency
- database query latency
- end-to-end latency

## 7. User Experience

Final application should allow users to:

- browse processed documents
- ask Arabic or English questions
- inspect sources/citations
- view relevant images/charts/pages
- request calculations/comparisons
- view historical financial values
- request forecasts

Streamlit is currently the planned interface for the first version.

## 8. Guiding Engineering Principle

Use deterministic/non-AI methods whenever they are more accurate.

Examples:

- use parsers for digital text
- use SQL for calculations
- use statistical models for forecasting
- use AI where semantic understanding or visual understanding is genuinely required

Do not use an LLM simply because an LLM is available.
