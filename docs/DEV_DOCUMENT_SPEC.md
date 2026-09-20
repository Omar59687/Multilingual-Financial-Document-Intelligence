# DEV Document Specification — MizanIQ dataset_v0.1

> Authoritative specification for the 10 representative development documents.
> Related docs: [DATASET_DESIGN.md](DATASET_DESIGN.md) §7 · [CANONICAL_DATA_MODEL.md](CANONICAL_DATA_MODEL.md) §6 · [EVALUATION_PLAN.md](EVALUATION_PLAN.md) §2
>
> Status: ACTIVE for `dataset_v0.1`. DEV IDs are permanent: regeneration keeps
> IDs, changes content hashes (recorded in `data/dev/dataset_v0.1/manifest.json`).

## 0. Global rules

- **One-way flow:** canonical truth (`data/canonical/dataset_v0.1/`) → DEV
  documents + ground truth. Canonical truth is NEVER altered to suit rendering.
- **Deterministic selection:** each document's source rows are fixed by an
  explicit rule below (§1–§10, "Source selection"). Rules re-resolve against
  canonical data at generation time; resolved IDs are recorded in ground truth
  and cross-checked against the pinned IDs in this spec (generation fails
  closed on mismatch).
- **Resolved values below are informational snapshots** of canonical v0.1
  (seed 42). Normative for evaluation: the ground-truth JSON files.
- **DEV-004 format decision:** [DATASET_DESIGN.md](DATASET_DESIGN.md) §7 once
  proposed DEV-004 as a scanned-receipt *PDF*. The architect-approved v0.1 role
  (§Coverage) is "scanned Arabic **image/document**", so DEV-004 is a standalone
  **PNG** (this also preserves the format-coverage requirement of a standalone
  Arabic image). No implementation blocker; roles otherwise unchanged.
- **Arabic rendering:** reportlab/Pillow/matplotlib do not shape Arabic
  natively. All Arabic raster/vector text goes through `arabic_reshaper` +
  `python-bidi` with Arial (`C:\Windows\Fonts\arial.ttf`, full Arabic
  coverage). DOCX needs no reshaping (Word shapes natively; paragraphs flagged
  RTL with a complex-script font). No broken/disconnected glyphs are shipped:
  rendered Arabic is verified shaped/connected in the PDF content stream.
- **Bidi-isolation rule (native PDFs):** digits and Latin identifiers are
  NEVER embedded inside reshaped Arabic paragraphs — they live in standalone
  LTR lines/cells (e.g. a KPI headline line, a `Reference event:` line, EN
  invoice cells with Arabic label-only counterparts). Rationale, verified
  empirically: naive text extractors (pypdf-style) drop digit/Latin runs that
  sit inside reshaped RTL paragraphs, even though the rendered glyphs are
  correct for OCR. Arabic prose uses spelled-out years where a year must
  appear in narrative (e.g. "عام ألفين واثنين وعشرين"). Pure-digit table
  cells extract exactly and are used for tabular values.
- **Charts are Pillow-native** (matplotlib intentionally unused — see
  requirements.txt): deterministic drawing with Arial, no native-extension
  nondeterminism.
- **Numbers:** Western digits in native documents; DEV-004/DEV-010 scanned
  fixtures may mix Arabic-Indic digits per the model §5 normalizer contract
  (ground truth records both forms where used).
- **Currency:** SAR throughout (`SAR` / `ر.س`).

## Coverage matrix (must hold; validated)

| Required case        | Covered by            |
|----------------------|-----------------------|
| English native PDF   | DEV-001               |
| Arabic               | DEV-002, DEV-004      |
| Mixed Arabic/English | DEV-003, DEV-005, DEV-007, DEV-009 |
| Scan / OCR           | DEV-004, DEV-010      |
| Financial table      | DEV-001, DEV-002, DEV-003, DEV-006, DEV-007, DEV-010 |
| Chart                | DEV-008, DEV-009      |
| DOCX                 | DEV-005               |
| XLSX                 | DEV-006               |
| CSV                  | DEV-007               |
| Standalone image     | DEV-004, DEV-008, DEV-009 |

---

## DEV-001 — English native annual income statement (PDF)

- **Filename:** `DEV-001_income_statement_2023_EN.pdf` · **Format:** PDF
  · **Language:** en · **Native/scanned:** native digital, selectable text
- **Source year/period:** FY 2023 (company-wide, all branches) with FY 2022
  comparatives · **Source tables:** `monthly_financials` (36 rows: 2023 × 3
  branches), `business_events` (`EVT-2023-001`)
- **Source selection:** aggregate all `monthly_financials` rows with
  `period LIKE '2023-%'` (and `'2022-%'` for comparatives). Full enumeration,
  no sampling.
- **Resolved FY2023 company totals (SAR):** revenue 97,815,213.44 · cogs
  63,174,407.47 · gross_profit 34,640,805.97 · operating_expenses 21,825,673.10
  · operating_profit 12,815,132.87 · other_income 41,726.49 · finance_costs
  32,467.57 · tax_expense 1,410,683.10 · net_income 11,413,708.69.
  Branch revenue split: RUH 55,963,929.28 · JED 28,983,851.32 · DMM 12,867,432.84.
- **Purpose:** financial-performance report for FY2023.
- **Table:** yes — 9-line P&L with 2023 / 2022 / YoY-change columns.
  **Chart/image:** no.
- **Difficulty:** easy (baseline native extraction case).
- **Capabilities tested:** native text + table extraction, financial-field
  extraction, YoY calculation questions.
- **Explanatory paragraph:** one short paragraph tied to `EVT-2023-001`
  (Jeddah Sep–Oct 2023 refit disruption), EN only.
- **Metadata:** title "Noor Retail & Distribution Co. — Financial Performance
  Report FY2023", 1 page, generator `stmt_en_v1`.
- **Example questions:** "What was net income in 2023?" (type 1); "What was
  gross margin in 2023 vs 2022?" (type 4); "Which branch contributed most to
  2023 revenue?" (type 5).
- **Answer source:** recomputed sums of the 36 canonical rows (ground truth
  `expected_numeric_values`).
- **Citation strategy:** stable 1-page layout — filename + page 1 + table row
  label (ground truth `expected_page_locations`).
- **Edge cases:** below-operating lines are small but non-zero (tests sparse
  lines); YoY column is document-computed from the same truth (validator
  recomputes).

## DEV-002 — Arabic branch expense report (PDF)

- **Filename:** `DEV-002_branch_expenses_2022_AR.pdf` · **Format:** PDF
  · **Language:** ar (Arabic-first) · **Native/scanned:** native digital
- **Source year/period:** FY 2022, branch BR-JED · **Source tables:**
  `monthly_financials` (12 rows: 2022 × BR-JED), `transactions` (372 expense
  rows: 2022 × BR-JED, for the labeled extract analysis)
- **Source selection:** all `monthly_financials` rows `2022-*` + BR-JED
  (primary table); all `transactions` rows with `date LIKE '2022-%'`,
  BR-JED, `transaction_type='expense'` (secondary analysis, explicitly
  labeled as a classified extract, NOT exhaustive).
- **Resolved values (SAR):** 2022 Jeddah operating-expenses total
  **6,472,903.83** (12-month table sums exactly to this; headline rendered as
  an Arabic label line plus a standalone LTR KPI line
  `2022 : 6,472,903.83 SAR` per the bidi-isolation rule). Extract category
  totals: salaries 434,730.16 · logistics 378,092.83 · marketing 336,760.09 ·
  maintenance 335,536.91 · utilities 276,178.11 · rent 275,858.56 (sum
  2,037,156.66 — labeled as partial extract; sample size stated on its own
  LTR line `Sample size: 372 transactions`). The narrative paragraph spells
  the year in words and cites the event on an LTR `Reference event:
  EVT-2022-001` line.
- **Purpose:** Jeddah branch annual operating-expense report.
- **Table:** yes — 12-month opex table + category analysis table.
  **Chart/image:** no.
- **Difficulty:** medium (Arabic table extraction, RTL layout).
- **Capabilities tested:** Arabic understanding, Arabic table extraction,
  table aggregation (AR→AR).
- **Metadata:** Arabic title, 1 page, generator `branch_exp_ar_v1`.
- **Example questions (AR):** "ما إجمالي المصروفات التشغيلية لفرع جدة في
  2022؟" (type 10); "أي شهر سجل أعلى مصروفات؟" (type 16).
- **Answer source:** canonical monthly sums (headline); canonical tx subset
  sums (category analysis).
- **Citation strategy:** filename + page 1 + table title
  (`جدول المصروفات الشهرية` / `تحليل البنود`).
- **Edge cases:** RTL table rendering via reshaper+bidi; Western digits used
  (documented); extract-vs-truth distinction explicit in both doc and ground
  truth to prevent competing-truth scoring errors.

## DEV-003 — Mixed-language supplier invoice (PDF)

- **Filename:** `DEV-003_supplier_invoice_2023_MIX.pdf` · **Format:** PDF
  · **Language:** mixed · **Native/scanned:** native digital
- **Source tables:** `invoices` (1 row), `transactions` (1 linked row)
- **Source selection rule:** the 2023 invoice with the highest `total`.
  **Pinned resolution:** `INV-2023-0106` (2023-06-07, BR-RUH, Al-Safi Foods
  Supplier, subtotal 71,486.83, VAT 10,723.02, total **82,209.85**, linked
  `TX-2023-029559`). Generation fails if the rule resolves otherwise.
- **Purpose:** supplier invoice for exact-identifier (BM25) testing.
- **Table:** yes (invoice line block: subtotal/VAT/total). **Chart/image:** no.
- **Difficulty:** easy natively; the value is identifier exactness.
- **Capabilities tested:** mixed-language extraction, exact invoice-number
  retrieval, invoice arithmetic verification.
- **Content:** Arabic vendor block (المورد: الصافي للأغذية… — label-only
  Arabic cells; IDs/dates/amounts in the LTR cells per the bidi-isolation
  rule), English line items, invoice ID prominent, date, branch, VAT 15% line,
  total, related transaction reference `TX-2023-029559`.
- **Example questions:** "What is the total of invoice INV-2023-0106?"
  (type 14); "Who is the vendor?" (type 14).
- **Answer source:** the single canonical invoice row (ground truth
  `expected_identifiers` + `expected_numeric_values`).
- **Citation strategy:** filename + page 1 + invoice-ID anchor.
- **Edge cases:** mixed-direction layout (RTL vendor block, LTR totals);
  ID must be byte-exact including hyphens.

## DEV-004 — Scanned Arabic receipt (PNG image)

- **Filename:** `DEV-004_scanned_receipt_2019_AR.png` · **Format:** PNG
  · **Language:** ar · **Native/scanned:** scanned-style raster, NOT
  selectable.
- **Source tables:** `transactions` (1 row)
- **Source selection rule:** the largest-`total_amount` BR-JED expense
  transaction of 2019-06. **Pinned resolution:** `TX-2019-013526`
  (2019-06-09, marketing, City Marketing Agency / وكالة المدينة للتسويق,
  amount 24,371.25, VAT 3,655.69, total **28,026.94 SAR**).
- **Purpose:** Arabic OCR-on-degraded-scan case.
- **Table:** no. **Chart/image:** the receipt itself (stamp-like overlay,
  faint ruled background).
- **Difficulty:** hard (degraded Arabic raster).
- **Capabilities tested:** Arabic OCR, amount/date/vendor field reading
  under degradation.
- **Degradation parameters (deterministic, seed 42):** rotation **2.5°**,
  Gaussian blur radius **0.6px**, brightness ×**1.04**, Gaussian noise
  σ=**3.0** (grayscale units), PNG (lossless — compression is NOT a tested
  variable here). Must remain comfortably human-readable.
- **Content:** Arabic-dominant receipt: vendor, date 2019-06-09, reference
  `TX-2019-013526`, amount + VAT + total lines (Western digits), stamp box.
- **Example questions:** "What amount is shown on the 2019 receipt?"
  (type 10/AR→AR); "Which vendor issued it?"
- **Answer source:** the single canonical transaction row.
- **Citation strategy:** whole-image citation (filename, no pages).
- **Edge cases:** rotation crops nothing (canvas padded pre-rotation);
  reshaped Arabic verified present in the *clean* pre-degradation render
  (ground truth `generation_parameters.clean_text`).

## DEV-005 — Management commentary (DOCX, mixed)

- **Filename:** `DEV-005_management_commentary_2023_MIX.docx` · **Format:** DOCX
  · **Language:** mixed · **Native/scanned:** native digital
- **Source tables:** `business_events` (`EVT-2023-001` primary,
  `EVT-2022-001` + `EVT-2019-001` context), `monthly_financials` (2023
  Jeddah Sep–Oct vs 2022; company 2023 revenue 97,815,213.44)
- **Source selection:** fixed event IDs above (curated, not sampled).
- **Purpose:** management discussion explaining 2023 expense/margin drivers
  for explanatory (why-) RAG.
- **Table:** no. **Chart/image:** no. ~2 pages, EN paragraphs + AR paragraphs
  (RTL-flagged, native shaping).
- **Difficulty:** medium (cross-language semantic retrieval).
- **Capabilities tested:** semantic text RAG, AR↔EN cross-language retrieval,
  explanation questions.
- **Content rule:** every number/fact traces to canonical rows or the cited
  event explanations; NO free-form invented facts. Includes ≥1 canonical
  numeric value (Jeddah Sep–Oct revenue dip; company 2023 revenue).
- **Example questions:** "Why did Jeddah revenue dip in late 2023?" (type 8);
  "ماذا قالت الإدارة عن سبب ارتفاع المصروفات؟" (type 11/13).
- **Answer source:** event `explanation_en/ar` + canonical aggregates.
- **Citation strategy:** filename + paragraph index (ground truth
  `expected_page_locations` as `p1..pN`).
- **Edge cases:** AR/EN passages informationally equivalent but NOT literal
  translations (per model §5); event IDs cited inline (`EVT-2023-001`).

## DEV-006 — Monthly transactions workbook (XLSX)

- **Filename:** `DEV-006_monthly_transactions_2023_EN.xlsx` · **Format:** XLSX
  · **Language:** en · **Native/scanned:** native digital
- **Source tables:** `transactions`
- **Source selection rule:** all rows with `date LIKE '2023-%'` (full
  enumeration → **4,020 rows**), chronological + transaction_id order.
- **Purpose:** spreadsheet-parsing + structured-extraction fixture.
- **Table:** yes (single flat sheet `transactions_2023` + machine-readable
  header, frozen panes, autofilter; no formulas).
- **Difficulty:** easy-medium (volume + sheet parsing).
- **Capabilities tested:** XLSX ingestion, exact ID preservation,
  aggregation calculations.
- **Columns:** transaction_id, date, branch_id, department_id,
  transaction_type, category, vendor_customer, description_en, amount,
  tax_amount, total_amount, payment_method, reference_number (values
  byte-identical to canonical CSV cells; amounts as numbers with 2dp format).
- **Example questions:** "What were total Q4 2023 procurement costs?"
  (type 16); "How many 2023 transactions exceed 10,000 SAR?"
- **Answer source:** the 4,020 canonical rows (ground truth lists row count
  + expected aggregates + first/last IDs).
- **Citation strategy:** filename + sheet + transaction_id.
- **Edge cases:** 4,020-row volume; empty `reference_number` cells preserved
  as empty (not "NULL" strings).

## DEV-007 — Transaction extract (CSV)

- **Filename:** `DEV-007_transactions_extract_2024_MIX.csv` · **Format:** CSV
  · **Language:** mixed (English headers, Arabic descriptions alongside EN)
  · **Native/scanned:** native digital
- **Source tables:** `transactions`
- **Source selection rule:** `department_id='DEP-LOG'` AND `date` in 2024-01
  … 2024-06 (H1) → **346 rows**, `total_amount` sum **1,666,716.10 SAR**.
  Disjoint in period, filter, and language profile from DEV-006.
- **Purpose:** native CSV ingestion + exact filtering/aggregation fixture.
- **Table:** yes (flat CSV). **Chart/image:** no.
- **Difficulty:** easy.
- **Capabilities tested:** CSV ingestion, category/branch filtering,
  aggregation correctness.
- **Example questions:** "Total Logistics spending in H1 2024?" (type 16);
  "How many transactions exceed 10,000 SAR?" (type 3/16).
- **Answer source:** the 346 canonical rows.
- **Citation strategy:** filename + transaction_id.
- **Edge cases:** UTF-8 with Arabic text (BOM-free); amounts plain `2dp`
  with no thousands separators.

## DEV-008 — Standalone revenue chart (PNG)

- **Filename:** `DEV-008_revenue_trend_2015_2024_EN.png` · **Format:** PNG
  · **Language:** en · **Native/scanned:** born-digital raster
- **Source tables:** `monthly_financials` (annual company sums 2015–2024)
- **Source selection:** full enumeration of the 10 annual totals:
  2015 51,855,389.81 · 2016 55,430,851.44 · 2017 58,327,888.72 ·
  2018 61,840,268.29 · 2019 72,735,487.00 · 2020 86,350,379.35 ·
  2021 86,405,590.69 · 2022 93,847,380.13 · 2023 97,815,213.44 ·
  2024 105,185,955.08.
- **Purpose:** chart-understanding + visual-retrieval fixture.
- **Table:** no. **Chart:** 10-year annual revenue line chart, labeled values,
  title/axes (Pillow-native drawing, 1600×1000).
- **Difficulty:** medium (visual reading, no selectable numbers).
- **Capabilities tested:** chart understanding, visual retrieval, trend
  questions.
- **Example questions:** "In which year was revenue highest?" (type 17);
  "Show me the revenue trend chart." (visual retrieval).
- **Answer source:** the 10 canonical annual sums (ground truth
  `expected_chart_data` + one-sentence trend: steady growth with a 2019
  level shift from the Dammam opening and a 2020 demand surge).
- **Citation strategy:** whole-image citation.
- **Edge cases:** no data table or sidecar inside the image; value labels
  rounded to millions on-chart while ground truth keeps exact sums.

## DEV-009 — KPI dashboard (JPG, mixed)

- **Filename:** `DEV-009_kpi_dashboard_Q4-2023_MIX.jpg` · **Format:** JPG
  · **Language:** mixed · **Native/scanned:** born-digital raster
  (dashboard export look)
- **Source period:** 2023-Q4 (company-wide) · **Source tables:**
  `monthly_financials`, `budgets`
- **Source selection:** Q4-2023 month rows (Oct–Dec, all branches) + the
  three `2023-Q4` revenue budget rows.
- **Resolved KPIs (SAR):** revenue **23,902,434.34** · gross profit
  **8,482,081.72** · operating expenses **5,346,394.95** · net income
  **2,795,781.39** · revenue budget 23,849,529.72 (RUH 13,446,041.21 + JED
  7,409,550.73 + DMM 2,993,937.78) → variance **+52,904.62 (+0.22%,
  budget met)**. Branch revenue bars: RUH 13,833,788.57 · JED 6,858,452.95 ·
  DMM 3,210,192.82.
- **Purpose:** complex visual-layout + mixed-language visual QA fixture.
- **Table:** no. **Chart/image:** KPI cards + branch bars + budget-vs-actual
  panel (1600×1000, JPEG quality 92, no EXIF).
- **Difficulty:** hard (multi-panel layout, mixed labels).
- **Capabilities tested:** visual layout understanding, mixed-language visual
  QA, visual citation.
- **Example questions:** "Which branch led Q4 2023 revenue?" (type 17);
  "Was the quarterly budget met?" (chart + arithmetic).
- **Answer source:** canonical Q4 aggregates + budget rows.
- **Citation strategy:** whole-image citation (+ panel name in ground truth
  `expected_visual_elements`).
- **Edge cases:** JPEG artifacts are presentation-only; ground truth keeps
  exact 2dp values; mixed AR/EN labels reshaped via the standard pipeline.

## DEV-010 — Scanned English balance sheet (PDF)

- **Filename:** `DEV-010_balance_sheet_2020_EN_scanned.pdf` · **Format:** PDF
  · **Language:** en · **Native/scanned:** scanned — single raster page,
  main content NOT selectable.
- **Source tables:** `annual_balance_sheet` (year 2020)
- **Source selection:** fixed row `year='2020'`. **Resolved (SAR):** cash
  5,087,893.71 · AR 7,574,649.52 · inventory 9,606,420.63 · other current
  1,740,159.33 · P&E 30,950,869.54 · **total assets 54,959,992.73** ·
  AP 8,596,533.78 · debt 17,842,587.42 · other liab 2,575,687.57 ·
  **total liabilities 29,014,808.77** · **equity 25,945,183.96**.
- **Purpose:** hard OCR case: English scanned-table reconstruction.
- **Table:** yes (bordered assets/liabilities/equity tables, scanned).
  **Chart/image:** scan texture only.
- **Difficulty:** hard (meaningfully harder than DEV-001).
- **Capabilities tested:** English OCR + scanned-table reconstruction,
  balance-sheet equation questions.
- **Construction:** clean page composed in Pillow (deterministic drawing) →
  degradation → JPEG-encoded (q85, scan-realistic, deterministic) → embedded
  as the only page image in a PDF with normalized metadata. **Degradation:** rotation **1.8°**, blur **0.5px**, brightness
  ×**0.97** (slightly dark scan), noise σ=**2.5**, plus 2 faint horizontal
  band artifacts (fixed positions).
- **Example questions:** "What were total assets at end of 2020?" (type 15);
  "Compare 2020 vs 2019 equity." (needs truth; 2019 equity in ground truth
  as context, NOT rendered).
- **Answer source:** the 2020 canonical balance-sheet row (equation
  verified: 54,959,992.73 = 29,014,808.77 + 25,945,183.96).
- **Citation strategy:** filename + page 1 + table section
  (`Assets`/`Liabilities`/`Equity`).
- **Edge cases:** PDF must contain no extractable body text (validated with
  pypdf: stripped text length < 50 chars); comparatives column shows 2019
  totals row only where needed (from truth, recorded in ground truth).

---

## Directory layout (generated, git-ignored via `data/`)

```text
data/dev/dataset_v0.1/
  documents/        # the 10 DEV files ONLY — the future ingestion corpus
  ground_truth/     # DEV-001.json … DEV-010.json (evaluation tooling, NOT corpus)
  manifest.json     # file list, sha256, sizes, canonical pointer, seed, params
```

## Ground-truth record schema (one JSON per DEV doc)

`dev_id, filename, format, language, native_scanned, source_tables,
source_record_ids, source_period, expected_metadata, expected_text,
expected_fields, expected_numeric_values, expected_identifiers,
expected_tables, expected_chart_data, expected_visual_elements,
expected_page_locations, expected_question_targets, generation_parameters`
(absent sections are `null`, never prose-only). `expected_numeric_values`
are strings at 2dp recomputed from canonical rows at generation time.

## Traceability chain (this task covers the first two arrows)

```text
canonical record (source_record_ids)
  → DEV document (manifest content_hash + ground-truth JSON)   [THIS TASK]
  → page/element → chunk → retrieval → citation                 [FUTURE]
```

## No-leakage rule

`ground_truth/` and canonical CSVs are evaluation/build fixtures and live
OUTSIDE the ingestible corpus. Future ingestion may read ONLY
`data/dev/dataset_v0.1/documents/`. Ground-truth filenames/IDs must never be
embedded in document bytes beyond the documented visible content (e.g. the
visible invoice ID on DEV-003 is content, not leakage). Manifest records this.

## Reproducibility strategy

- Seeded RNG (seed 42) for all stochastic steps (noise, sampling positions).
- Normalized file metadata for byte-determinism: fixed PDF
  Creator/CreationDate/ModDate; DOCX/XLSX `core.xml`
  created/modified reset to a fixed timestamp; PNG/JPG carry no timestamps.
- Verification: generate twice into temp dirs; compare filenames, file bytes
  (PDF/DOCX/XLSX/CSV/PNG/JPG), and   ground-truth JSON modulo `generated_at`.
  Pillow raster output is byte-deterministic on a fixed library+font stack
  (pinned in requirements); the temp-dir comparison proves it empirically
  rather than assuming it.
