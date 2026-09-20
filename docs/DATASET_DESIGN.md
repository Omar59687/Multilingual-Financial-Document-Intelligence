# MizanIQ Dataset Design

AI-Powered Finance Document Intelligence System

> Authoritative specification for the dataset (design only — no data collected yet).
> Related docs: [PROJECT_SPEC.md](PROJECT_SPEC.md) · [ARCHITECTURE.md](ARCHITECTURE.md) · [ROADMAP.md](ROADMAP.md) · [DEVELOPMENT_RULES.md](DEVELOPMENT_RULES.md) · [CANONICAL_DATA_MODEL.md](CANONICAL_DATA_MODEL.md) · [EVALUATION_PLAN.md](EVALUATION_PLAN.md)
>
> Status: DRAFT for architect review. See [Section 12](#12-open-decisions) for unresolved decisions.

## 1. Purpose

The MizanIQ dataset must simulate a realistic finance/accounting client environment rather than simply collecting random financial files.

A real client engagement does not arrive as a neat folder of identical spreadsheets. It arrives as a heterogeneous accumulation of documents produced over many years, by different people, in different tools, in two languages, with inconsistent formatting, scanned paperwork, embedded charts, and partial records. The dataset must reproduce that reality in a controlled way.

The final system will encounter heterogeneous documents spread across approximately 10 years. The dataset must therefore support evaluation of:

- native text extraction
- OCR
- Arabic understanding
- English understanding
- cross-language retrieval
- table extraction
- chart understanding
- financial field extraction
- exact calculations
- RAG
- visual retrieval
- forecasting

Each of these capabilities needs dedicated coverage in the dataset: if no scanned Arabic invoice exists in the data, Arabic OCR cannot be measured; if no monthly series exists, forecasting cannot be evaluated. Coverage must be designed, not hoped for.

## 2. Company Scenario

### Proposed scenario: single fictional food retail and distribution company

**Noor Retail & Distribution Co.** — a fictional mid-sized food retail and wholesale distribution company operating three branches (Riyadh, Jeddah, Dammam) with departments including Sales, Procurement, Warehouse & Logistics, and Administration.

The company sells packaged foods and beverages through its own retail outlets and wholesale distribution to small grocers. It purchases from suppliers, holds inventory, pays branch operating expenses (rent, salaries, utilities), and reports annual financial statements with management commentary.

> This scenario is PROPOSED (see [Section 12](#12-open-decisions)). The binding constraint is: exactly ONE coherent fictional company, fully synthetic, with no dependence on private or proprietary company data.

### Why this scenario fits

- **RAG questions:** annual statements, quarterly commentary, accounting policies, and audit notes provide long-form Arabic/English prose with explainable events (e.g., a rent increase, a new branch opening, a supply-cost spike) that questions can ask about.
- **Calculations:** revenue, cost of sales (COGS), operating expenses, and net income across branches, departments, and years give natural aggregation, margin, variance, and comparison questions with verifiable answers.
- **Trend analysis:** 10 years of monthly revenue/expenses with seasonality (e.g., Ramadan demand uplift, summer beverage peaks) and a growth trend produce meaningful multi-year comparisons.
- **Forecasting:** monthly granularity over 10 years (120 monthly points per metric) is sufficient history for seasonal-naive, exponential-smoothing, and SARIMA-style models with holdout backtesting.
- **Arabic/English documentation:** a Gulf-region retailer naturally produces Arabic invoices, receipts, and regulatory-style documents alongside English-language financial statements and management commentary, plus mixed-language operational documents — exactly the trilingual coverage (AR / EN / mixed) the system must handle.
- **Document variety from one business:** statements, branch reports, budgets, transaction extracts, invoices, receipts, policies, audit notes, KPI dashboards, and charts all arise organically from this single scenario, avoiding 150 near-identical files.

## 3. Time Range

Recommended fixed 10-year period: **FY 2015 – FY 2024** (calendar years ending 31 December).

- **Proposed range (explicit):** 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024.
- **Rationale:** ten complete historical years; recent enough to be realistic (includes a COVID-period demand shock in 2020 as an explainable trend event); avoids an incomplete current year, which would complicate forecasting holdouts.
- **Status:** PROPOSED — requires architect approval (see [Section 12](#12-open-decisions)).

All dated artifacts (statements, transactions, budgets, commentary) must fall inside this range. The forecasting holdout should use the final 12 months (2024) as the default out-of-sample window.

## 4. Canonical Financial Truth Dataset

### Concept

The "canonical financial truth dataset" is a clean, structured source of truth containing the REAL financial values — defined BEFORE any messy PDF/Word/image representation is generated.

Document generation flows one way:

```text
canonical truth (structured values)
    → document renderers (PDF / DOCX / XLSX / CSV / PNG / JPG)
        → messy client-style artifacts (formatted, scanned, bilingual)
            → pipeline extraction → comparison against truth
```

The pipeline never sees the truth tables at query time; they exist only for generation consistency and evaluation scoring.

### Conceptual tables (schema NOT final)

**monthly_financials**

- date (month)
- revenue
- cogs
- opex
- net_income
- branch
- department
- currency

**transactions**

- transaction_id
- date
- category
- amount
- branch
- department
- description
- vendor

**budgets**

- period
- department
- budget_amount
- actual_amount

Final schemas (keys, types, constraints, currency handling) will be fixed during Phase 0 schema finalization and Phase 3 Pydantic modeling. Keys must be stable so every rendered document can reference `document_id` / period / source line back to truth rows.

### Why the truth dataset is useful

- synthetic documents can be generated consistently (every artifact renders from the same numbers; no contradictions between a chart and its statement)
- we know correct answers (evaluation answers are read from truth, not guessed)
- extraction accuracy can be measured (extracted fields vs. truth rows)
- RAG answers can be validated (numbers in answers checked against truth)
- calculations can be checked (aggregations recomputed deterministically from truth)
- forecasting has reliable historical data (clean monthly series with known events and holdouts)

## 5. Source Data Strategy

Do NOT download data yet. Strategies under consideration:

**A. Suitable open-source / Kaggle dataset** — adopt a real public financial dataset as the underlying numbers.

**B. Combination of multiple compatible public datasets** — stitch several public sources (e.g., retail sales + expense records) into one company view.

**C. Controlled synthetic financial dataset** — generate the canonical truth programmatically with realistic relationships (growth, seasonality, shocks, branch structure).

**D. Hybrid approach** — synthetic truth as the primary source, calibrated against public datasets (shapes, ratios, seasonal patterns checked against real retail data for realism).

### Recommendation: investigate D first, with C as the backbone

Investigate the hybrid approach first because it is the only option that simultaneously satisfies all criteria:

- **Legal/public usage:** fully synthetic truth carries zero licensing or privacy risk; public data is used only as a realism reference, never redistributed as client data.
- **10-year coverage:** public datasets rarely span 10 clean, consistent years for one company; synthesis guarantees the full 2015–2024 range.
- **Temporal granularity:** synthesis guarantees 120 monthly points per metric plus transaction-level rows; public data at that granularity for a single firm is rarely available.
- **Realistic financial relationships:** calibration against public retail data (margins, seasonality, expense ratios) keeps synthetic numbers plausible without inheriting a foreign schema.
- **Enough records for many document types:** truth tables can emit statements, branch reports, budgets, extracts, invoices, and charts from one consistent source.
- **No sensitive/private data:** nothing real ever enters the corpus.

Do NOT select a dataset solely because it has "finance" in the title. Any candidate public reference must be judged on granularity, time coverage, license, and structural fit — not its name. If no suitable public reference is found during investigation, fall back to pure C (controlled synthetic with accountant-reviewed assumptions) and record that decision.

## 6. Final ~150-Document Distribution

PROPOSED distribution for approximately 150 files. Figures are targets, not exact counts; the three cuts below (format, language, purpose) each describe the same ~150 files from a different angle.

### By format (target: 150)

| Format  | Count | Notes                                              |
|---------|-------|----------------------------------------------------|
| PDF     | 68    | statements, reports, commentary, scanned paperwork |
| XLSX    | 26    | branch reports, budgets, transaction workbooks     |
| CSV     | 18    | transaction extracts, monthly series               |
| DOCX    | 20    | commentary, policies, audit notes, branch reports  |
| PNG/JPG | 18    | standalone charts, dashboards, receipt photos      |
| **Total** | **150** |                                                |

### By language (target: 150)

| Language            | Count | Share |
|---------------------|-------|-------|
| English             | 60    | ~40%  |
| Arabic              | 45    | ~30%  |
| Mixed Arabic/English| 45    | ~30%  |
| **Total**           | **150** | 100% |

### By visual complexity (overlapping — not additive)

| Characteristic    | Approx. count | Notes                                      |
|-------------------|---------------|--------------------------------------------|
| Native digital    | ~95           | selectable text, parser-readable           |
| Scanned / photo   | ~35           | requires OCR                               |
| Contains tables   | ~95           | statements, reports, extracts, invoices    |
| Contains charts   | ~35           | KPI summaries, dashboards, commentary      |
| Embedded images   | ~40           | logos, signatures, stamps, receipt photos  |

### By document purpose (target: 150)

| Purpose                          | Count | Typical format/language        |
|----------------------------------|-------|--------------------------------|
| Annual income statements         | 10    | PDF, EN + AR                   |
| Annual balance sheets            | 10    | PDF, EN + AR                   |
| Quarterly branch reports         | 20    | PDF/DOCX/XLSX, EN/AR/mixed     |
| Management commentary            | 12    | DOCX/PDF, EN + mixed           |
| Budget vs. actual reports        | 10    | XLSX, EN + mixed               |
| Monthly transaction extracts     | 24    | CSV/XLSX, mixed                |
| Invoices (supplier/wholesale)    | 20    | PDF scans, AR + mixed          |
| Expense receipts                 | 10    | JPG/PNG scans, AR + mixed      |
| Audit notes                      | 6     | DOCX/PDF, EN                   |
| Accounting policies              | 6     | DOCX/PDF, EN + AR              |
| KPI summaries / dashboards       | 12    | PNG/JPG + PDF, mixed           |
| Cash-flow summaries              | 10    | PDF/XLSX, EN                   |
| **Total**                        | **150** |                              |

Counts: 10+10+20+12+10+24+20+10+6+6+12+10 = 150.

### Anti-uniformity rule

Avoid generating 150 nearly identical documents. Vary explicitly across: year, branch, author/template style, Arabic vs. English terminology, number formats (Arabic-Indic vs. Western digits in scanned items), table layouts, chart types, scanners/photo quality, and narrative events. At least two layout templates per repeated purpose (e.g., pre-2020 vs. post-2020 statement style after a rebrand) must be planned at generation time.

## 7. Ten Representative Development Documents

Exactly 10 stable development documents. IDs DEV-001…DEV-010 are permanent: if a file is regenerated, the ID stays with its role.

### DEV-001 — English native annual income statement (PDF)

- Proposed filename: `DEV-001_income_statement_2023_EN.pdf`
- Format / year / language: PDF / 2023 / English
- Native / scanned: native digital
- Contents: full-year income statement (revenue, COGS, gross profit, opex lines, net income) with prior-year comparatives
- Table: yes · Chart/image: no
- Capability tested: native text + table extraction, financial field extraction
- Example questions: "What was net income in 2023?"; "What was gross margin in 2023 vs 2022?"

### DEV-002 — Arabic branch expense report (PDF)

- Proposed filename: `DEV-002_branch_expenses_2022_AR.pdf`
- Format / year / language: PDF / 2022 / Arabic
- Native / scanned: native digital
- Contents: Jeddah branch operating-expense breakdown by category with totals
- Table: yes · Chart/image: no
- Capability tested: Arabic understanding, Arabic table extraction
- Example questions (AR): "ما إجمالي المصروفات التشغيلية لفرع جدة في 2022؟"؛ "أي بند مصروفات كان الأعلى؟"

### DEV-003 — Mixed-language supplier invoice (PDF)

- Proposed filename: `DEV-003_supplier_invoice_2023_MIX.pdf`
- Format / year / language: PDF / 2023 / mixed Arabic/English
- Native / scanned: native digital
- Contents: supplier invoice with Arabic vendor block, English line items, invoice number, VAT, totals
- Table: yes (line items) · Chart/image: no
- Capability tested: mixed-language extraction, exact identifier (invoice number) retrieval via BM25
- Example questions: "What is the total of invoice INV-2023-…?"; "Who is the vendor on that invoice?"

### DEV-004 — Scanned Arabic receipt (PDF)

- Proposed filename: `DEV-004_scanned_receipt_2019_AR.pdf`
- Format / year / language: PDF / 2019 / Arabic
- Native / scanned: scanned (rotated ~3°, stamp overlay)
- Contents: scanned expense receipt with vendor, date, handwritten-style total, stamp
- Table: no · Chart/image: embedded scan noise/stamp
- Capability tested: Arabic OCR on degraded scan
- Example questions: "What amount is shown on the 2019 receipt?"; "Which vendor issued it?"

### DEV-005 — Management commentary (DOCX, mixed)

- Proposed filename: `DEV-005_management_commentary_2023_MIX.docx`
- Format / year / language: DOCX / 2023 / mixed Arabic/English
- Native / scanned: native digital
- Contents: 2–3 pages of management discussion (expense drivers, new Riyadh warehouse, margin explanation)
- Table: no · Chart/image: no
- Capability tested: semantic text RAG, cross-language retrieval, explanation questions
- Example questions: "Why did operating expenses increase in 2023?"; "ماذا قالت الإدارة عن سبب ارتفاع المصروفات؟" (AR question over mixed evidence)

### DEV-006 — Monthly transactions workbook (XLSX)

- Proposed filename: `DEV-006_monthly_transactions_2023_EN.xlsx`
- Format / year / language: XLSX / 2023 / English
- Native / scanned: native digital
- Contents: 12 monthly sheets or one dated table of categorized transactions by branch/department
- Table: yes · Chart/image: no
- Capability tested: spreadsheet parsing, structured extraction → DuckDB, aggregation calculations
- Example questions: "What were total Q4 2023 procurement costs?"; "Which month had the highest revenue?"

### DEV-007 — Transaction extract (CSV)

- Proposed filename: `DEV-007_transactions_extract_2024_MIX.csv`
- Format / year / language: CSV / 2024 / mixed (Arabic descriptions, English headers)
- Native / scanned: native digital
- Contents: flat transaction rows (id, date, category, amount, branch, description)
- Table: yes (flat) · Chart/image: no
- Capability tested: CSV ingestion, exact category/branch filtering, aggregation
- Example questions: "Total Logistics spending in H1 2024?"; "How many transactions exceed 10,000 SAR?"

### DEV-008 — Standalone revenue chart (PNG)

- Proposed filename: `DEV-008_revenue_trend_2015_2024_EN.png`
- Format / year / language: PNG / 2015–2024 / English labels
- Native / scanned: born-digital image
- Contents: 10-year annual revenue line/bar chart with labeled values
- Table: no · Chart/image: chart only
- Capability tested: chart understanding, visual retrieval
- Example questions: "Show me the revenue trend chart."; "In which year was revenue highest?"

### DEV-009 — KPI dashboard (JPG, mixed)

- Proposed filename: `DEV-009_kpi_dashboard_Q4-2023_MIX.jpg`
- Format / year / language: JPG / 2023 / mixed Arabic/English labels
- Native / scanned: born-digital image (dashboard export)
- Contents: quarterly KPI dashboard — revenue vs. budget gauge, branch bars, expense donut
- Table: no · Chart/image: multiple charts + layout
- Capability tested: complex visual layout, mixed-language visual QA, visual citation
- Example questions: "Which branch led Q4 2023 revenue?"; "Was the quarterly budget met?"

### DEV-010 — Scanned English balance sheet (PDF)

- Proposed filename: `DEV-010_balance_sheet_2020_EN_scanned.pdf`
- Format / year / language: PDF / 2020 / English
- Native / scanned: scanned (multi-page, bordered tables)
- Contents: scanned 2020 balance sheet with assets/liabilities/equity tables
- Table: yes (scanned) · Chart/image: no
- Capability tested: English OCR + scanned-table reconstruction
- Example questions: "What were total assets at end of 2020?"; "Compare 2020 vs 2019 equity."

### Coverage check

| Required case        | Covered by |
|----------------------|------------|
| English native PDF   | DEV-001    |
| Arabic               | DEV-002, DEV-004 |
| Mixed Arabic/English | DEV-003, DEV-005, DEV-007, DEV-009 |
| Scan / OCR           | DEV-004, DEV-010 |
| Financial table      | DEV-001, DEV-002, DEV-003, DEV-006, DEV-007, DEV-010 |
| Chart                | DEV-008, DEV-009 |
| DOCX                 | DEV-005    |
| XLSX                 | DEV-006    |
| CSV                  | DEV-007    |
| Standalone image     | DEV-008, DEV-009 |

## 8. Evaluation Ground Truth

For each development document we will later record:

- **document metadata** — doc_id, filename, format, year, language, native/scanned, purpose, branch/department
- **expected extracted text** — reference transcription (for OCR/parsing accuracy scoring)
- **expected financial fields** — metric/value/currency/period records traceable to canonical truth rows
- **expected tables** — cell-level reference for table reconstruction scoring
- **expected chart meaning** — chart type, series, labeled values, and the one-sentence trend reading a correct answer must match
- **expected answerable questions** — the question IDs this document supports
- **correct answers** — values read from canonical truth, with tolerance rules for rounding
- **correct source locations** — filename + page/sheet/region for citation scoring

### Ground-truth provenance rule

Ground truth must NOT be created by blindly accepting an LLM output as truth. Acceptable origins, in priority order:

1. **Canonical source data** — numeric answers derived from truth tables (deterministic recomputation).
2. **Known generated content** — text/tables/charts whose content was fixed at generation time (render scripts + fixtures are the reference).
3. **Human verification** — a person confirms OCR transcripts, chart readings, and cross-language answer equivalence; the reviewer and date are recorded per item.

LLM assistance may draft transcripts or candidate QA pairs, but every item ships only after rule 1–3 confirmation. Unverified items are marked `unverified` and excluded from reported scores.

## 9. Example Evaluation Questions

Initial list of ~20 QUESTION TYPES (not the final benchmark). Do not invent actual financial answers yet — answers will be read from canonical truth once data exists.

| # | Type | Category | Example shape (no invented values) |
|---|------|----------|-------------------------------------|
| 1 | Exact single-value lookup | exact lookup | "What was [metric] in [year]?" |
| 2 | Year-over-year delta | aggregation | "By how much did [metric] change from [year A] to [year B]?" |
| 3 | Multi-year sum/average | aggregation | "What was total/average [metric] over [period]?" |
| 4 | Ratio/margin computation | aggregation | "What was [margin ratio] in [year]?" |
| 5 | Branch comparison | comparison | "Which branch had higher [metric] in [year], and by how much?" |
| 6 | Period comparison | comparison | "Compare [metric] in H1 vs H2 of [year]." |
| 7 | Cross-document reconciliation | multi-document | "Do the income statement and branch reports agree on [year] revenue?" |
| 8 | Explanation from commentary | explanation | "Why did [metric] change in [year], according to management?" |
| 9 | Policy lookup | explanation | "What is the company's policy on [topic]?" |
| 10 | Arabic question, Arabic evidence | AR→AR | (Arabic) question answerable from an Arabic document |
| 11 | Arabic question, English evidence | AR→EN | (Arabic) question answerable only from an English statement |
| 12 | English question, Arabic evidence | EN→AR | (English) question answerable only from an Arabic invoice/report |
| 13 | Mixed-language question | cross-language | question mixing Arabic and English terms |
| 14 | Exact identifier lookup | identifier | "Find invoice [invoice-number] and report its total/vendor." |
| 15 | Table-cell question | table-based | question whose answer is one cell of a statement table |
| 16 | Table aggregation question | table-based | question requiring summing/filtering table rows |
| 17 | Chart reading question | chart-based | "Which year/branch is highest in [chart]?" |
| 18 | Chart + text combination | chart-based | trend from a chart explained with commentary evidence |
| 19 | Insufficient-evidence question | abstention | question about a year/metric absent from the corpus — system must decline |
| 20 | Forecasting request | forecasting | "Forecast [metric] for next year/quarter with method and uncertainty." |

The final benchmark will instantiate each type with concrete questions bound to DEV documents, truth values, source locations, and (for type 19) the expected abstention behavior.

## 10. Dataset Versioning

| Version | Content | Purpose |
|---------|---------|---------|
| `dataset_v0.1` | 10 development documents (DEV-001…DEV-010) + truth rows + ground truth | pipeline proving ground; all Phase 1–9 development and eval |
| `dataset_v0.2` | larger intermediate set (target ~40–60 files) | scaling rehearsal: batch ingestion, retries, checkpointing |
| `dataset_v1.0` | final ~150-document benchmark/demo dataset | portfolio benchmark, demo, final evaluation |

### Why versioning matters

- **Reproducibility:** results (extraction scores, retrieval metrics, forecasts) are meaningless unless the exact data version behind them is recorded.
- **No silent drift:** regenerating a document creates a new versioned artifact; DEV IDs stay stable but content hashes change, so regressions are detectable.
- **Gated scaling:** v0.2 exists only after v0.1 quality gates pass (per DEVELOPMENT_RULES.md Rule 11); v1.0 only after the pipeline proves itself at intermediate scale.
- **Evaluation integrity:** benchmark questions are versioned against the dataset version they were validated on; re-validation is required when the version changes.

 >> Version manifests (`manifest.json` per version: file list, hashes, truth-version pointer) will be specified at generation time, not in this design.

## 11. Data Leakage and Evaluation Safety

Evaluation questions and answers must not leak into the retrievable corpus or into any component that generates answers:

- **No answer keys in the corpus:** ground-truth files (answers, QA pairs, truth tables) live OUTSIDE the indexed document set and are never ingested, chunked, or embedded.
- **No generation-time contamination:** rendered documents must not contain hidden answer text (e.g., evaluation answers embedded in metadata, alt text, or filenames beyond the documented naming scheme).
- **Dev/eval separation:** a subset of questions (the held-out eval set) is reserved exclusively for scoring — never used during prompt tuning, chunking experiments, or reranker selection. Development-time probing uses only the dev question subset.
- **Truth-table isolation:** canonical truth is a build/eval fixture, not a retrieval source; the running system may only access values through indexed documents and DuckDB records produced by the pipeline.
- **Forecasting holdouts:** the 2024 holdout window is excluded from model fitting during evaluation; backtest protocol and metrics are fixed before models see holdout data.
- **Goal:** realistic evaluation — the system must retrieve, compute, and reason its way to answers, never recall them.

## 12. Open Decisions

| Decision | Status | Options | Recommendation | Needs architect approval? |
|----------|--------|---------|----------------|---------------------------|
| Exact source-data strategy | Approved (HYBRID) | A. open-source / B. combined public / C. synthetic / D. hybrid | D (own controlled synthetic truth; public data reference-only) | Approved |
| Company industry/scenario | Approved | retail & distribution / logistics / hospitality / other | Noor Retail & Distribution Co. (Saudi food retail + wholesale; Riyadh/Jeddah/Dammam; Sales, Procurement, Warehouse & Logistics, Administration) | Approved |
| Currency | Approved (SAR only, V1) | SAR / USD / EGP / multi | SAR single-currency (pegged, avoids FX noise) | Approved |
| 10-year range | Approved (2015-01-01–2024-12-31) | 2015–2024 / 2014–2023 / other | 2015–2024 (complete years, incl. 2020 shock + 2024 holdout) | Approved |
| Final ~150-document distribution | Approved as working target | counts in §6 | format/language/purpose tables in §6; may adjust after DEV-set experiments | Approved (working target) |
| Synthetic vs. open-source balance | Approved (synthetic-first hybrid) | pure public / pure synthetic / hybrid | hybrid synthetic-first (§5); external data never authoritative truth | Approved |
| DEV-001…010 file/role list | Proposed | list in §7 | approve roles; filenames adjustable at generation time | Yes |
| Ground-truth tolerance rules | Open | exact / rounding bands per metric | define at schema finalization (Phase 0); see [EVALUATION_PLAN.md](EVALUATION_PLAN.md) §3 | Yes |
| Forecasting holdout protocol | Proposed | last-12-months holdout on 2024 | approve; metrics fixed before modeling; see [EVALUATION_PLAN.md](EVALUATION_PLAN.md) §10 | Yes |

Approvals recorded above reflect architect decisions; remaining "Proposed"/"Open" rows still require review. Evidence-gathering order: (1) ~~approve scenario/currency/range~~ DONE, (2) investigate source-data references, (3) freeze truth schema ([CANONICAL_DATA_MODEL.md](CANONICAL_DATA_MODEL.md)), (4) generate DEV set.
