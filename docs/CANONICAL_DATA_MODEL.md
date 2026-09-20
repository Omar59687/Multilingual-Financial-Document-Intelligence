# MizanIQ Canonical Data Model

AI-Powered Finance Document Intelligence System

> Defines the source of truth from which future generated documents and evaluation answers originate.
> Related docs: [PROJECT_SPEC.md](PROJECT_SPEC.md) · [ARCHITECTURE.md](ARCHITECTURE.md) · [ROADMAP.md](ROADMAP.md) · [DEVELOPMENT_RULES.md](DEVELOPMENT_RULES.md) · [DATASET_DESIGN.md](DATASET_DESIGN.md) · [EVALUATION_PLAN.md](EVALUATION_PLAN.md)
>
> Status: DRAFT for architect review. No data generated yet.
> Architect-approved context: Noor Retail & Distribution Co. · Riyadh/Jeddah/Dammam (Dammam opened 2019-01-01; pre-opening periods NULL, never zero) · SAR only (V1) · 2015-01-01–2024-12-31 · HYBRID source strategy (own synthetic truth; public data reference-only) · quarterly budgets (budget truth only) · ~36,000 transactions target · flat 15% VAT (simplified, not historical) · generic `tax_expense` (no Zakat modeling) · 2024 final forecasting holdout.

## Founding Principle

The canonical dataset is the authoritative business reality.

Generated PDFs, DOCX files, spreadsheets, scans, charts, invoices, management notes and other documents are REPRESENTATIONS of that underlying truth.

If a generated document and the canonical truth disagree unexpectedly, the document-generation pipeline is wrong.

Do not solve inconsistencies by silently changing canonical truth.

Consequences:

- Truth tables are written once (by the future generator, from approved assumptions) and then frozen per dataset version.
- Every rendered artifact must be traceable to the truth rows it represents (see [Section 6](#6-document-traceability)).
- Evaluation answers are recomputed from truth, never copied from a rendered document.
- Corrections to numbers happen in the generator + truth together, as a new dataset version — never by editing a PDF's number to "match".

## 1. Core Design Principles

1. **Stable identifiers.** Every branch, department, transaction, invoice, event, and future document carries a permanent ID (`BR-RUH`, `TX-…`, `INV-…`, `EVT-…`, `DEV-…`/`DOC-…`). IDs never change meaning between dataset versions; regeneration keeps IDs, changes content hashes.
2. **Explicit dates.** Every dated record uses a full ISO date (`YYYY-MM-DD`); monthly aggregates use `period` = first day of month. No bare years, no ambiguous quarters (`Q3` always means `YYYY-Qn` with defined month bounds).
3. **Single SAR currency in V1.** All money fields are SAR. No currency column is needed on monthly/transaction rows in V1; the currency is a dataset-level constant recorded in the manifest. (Multi-currency is explicitly out of V1 scope.)
4. **Branch and department traceability.** Every financial fact carries `branch_id` (and `department_id` where meaningful) so branch comparisons, department budgets, and per-branch time series are always answerable.
5. **Financial relationships reconcile.** Invariants in [Section 3](#3-financial-consistency-rules) are enforced at generation time by validation code, not by hope. A generator run that violates an invariant fails closed.
6. **Generated documents trace back to canonical records.** Renderers receive truth-row references as input and record them in the document manifest (`source_record_ids`). A number on a page must always resolve to the row(s) that produced it.
7. **Known business explanations have ground truth.** Every narrative "why" the system may be asked about (expense spike, margin dip, branch closure) corresponds to a `business_events` row with bilingual explanations — the RAG equivalent of canonical numeric truth.
8. **Forecasting series come from canonical history.** Forecast training data is the monthly truth series, never numbers scraped from a chart image. Chart readings are evaluated as visual understanding, not as forecast inputs.
9. **Evaluation answers derive from canonical truth or verified generated content.** Numeric answers: recomputed from truth. Text/visual answers: fixed at generation time, then human-verified. Never LLM-asserted.
10. **Deterministic, reproducible generation.** Once implementation begins, all generators take a fixed random seed (default `seed = 42`, recorded in the dataset manifest) so any dataset version can be rebuilt byte-identically.

## 2. Entity Model

### branches

| Field | Type (proposed) | Notes |
|-------|-----------------|-------|
| `branch_id` | string PK | stable, e.g. `BR-RUH` |
| `branch_name_en` | string | e.g. `Riyadh` |
| `branch_name_ar` | string | e.g. `الرياض` |
| `city` | string | same city as branch (kept for document rendering) |
| `opened_date` | date | Dammam opens later — enables "new branch" trend questions |
| `status` | enum | `open` / `closed` / `planned` |

Expected initial rows:

| branch_id | branch_name_en | branch_name_ar | city | opened_date | status |
|-----------|----------------|----------------|------|-------------|--------|
| BR-RUH | Riyadh | الرياض | Riyadh | 2010-06-01 | open |
| BR-JED | Jeddah | جدة | Jeddah | 2012-04-01 | open |
| BR-DMM | Dammam | الدمام | Dammam | 2019-01-01 (APPROVED) | open |

Riyadh and Jeddah opened before the dataset period, so they have full 120-month histories. Dammam opens 2019-01-01 (approved): it creates a legitimate structural break supporting branch-comparison questions, "why did revenue jump" explanations, and honest forecasting edge cases (short history for one branch).

Approved NULL rule: Dammam periods before 2019-01 are NOT APPLICABLE / NULL — they are simply absent from financial tables. They must NEVER be represented as zero financial activity (zero revenue would be a false business fact and would corrupt forecasting and aggregation evaluation).

### departments

| Field | Type (proposed) | Notes |
|-------|-----------------|-------|
| `department_id` | string PK | e.g. `DEP-SALES` |
| `department_name_en` | string | |
| `department_name_ar` | string | |

Expected rows: Sales (`DEP-SALES` / المبيعات), Procurement (`DEP-PROC` / المشتريات), Warehouse & Logistics (`DEP-LOG` / المستودعات والخدمات اللوجستية), Administration (`DEP-ADM` / الإدارة).

Departments are a flat list in V1 — no hierarchy, no cost centers. (Hierarchy would be ERP creep; see [Part 4 review](#appendix-a--overengineering-review) in EVALUATION_PLAN context.)

### monthly_financials

The most important table: one row per month × branch (120 months × 3 branches = 360 rows, minus pre-opening Dammam months).

| Field | Type (proposed) | Role |
|-------|-----------------|------|
| `period` | date (YYYY-MM-01) | grain: branch-month |
| `branch_id` | FK → branches | |
| `revenue` | numeric(12,2) ≥ 0 | base input |
| `cogs` | numeric(12,2) ≥ 0 | base input (cost of sales) |
| `gross_profit` | numeric(12,2) | **derived**, enforced: `revenue − cogs` |
| `operating_expenses` | numeric(12,2) ≥ 0 | base input (single figure at monthly grain; category detail lives in `transactions`) |
| `operating_profit` | numeric(12,2) | **derived**, enforced: `gross_profit − operating_expenses` (may be negative) |
| `other_income` | numeric(12,2) ≥ 0, default 0 | small, sparse (e.g., scrap sales, rebates) |
| `finance_costs` | numeric(12,2) ≥ 0, default 0 | small, sparse (e.g., overdraft profit charges) |
| `tax_expense` | numeric(12,2) ≥ 0, default 0 | generic synthetic tax line (APPROVED: do NOT model Saudi Zakat rules); computed as a flat rate on positive pre-tax profit |
| `net_income` | numeric(12,2) | **derived**, enforced: `operating_profit + other_income − finance_costs − tax_expense` (may be negative) |

Accounting relationships:

```text
gross_profit     = revenue − cogs
operating_profit = gross_profit − operating_expenses
net_income       = operating_profit + other_income − finance_costs − tax_expense
```

Deliberate V1 simplifications (documented, not accidental):

- **No depreciation line.** Depreciation is folded into `operating_expenses` in V1. A separate fixed-asset register is ERP scope and is not needed for any planned question type.
- **No inventory balance sheet chain.** `cogs` is a base input, not derived from opening/closing stock. Balance-sheet documents (DEV-010, annual balance sheets) will render *stated* asset/equity figures from a small approved annual fixture, not from a full double-entry ledger. Full double-entry is out of scope.
- **Below-operating lines are sparse.** `other_income`, `finance_costs`, `tax_expense` default to 0 and appear only in months where a planned narrative or document needs them. This keeps `net_income ≈ operating_profit` in most months (easy to validate) while still exercising the full formula where it matters. `tax_expense` is a generic synthetic line only — no Zakat rules are modeled (approved).
- **No separate P&L-by-department monthly table.** Department detail at monthly grain comes from `transactions` aggregation; only branch-month P&L is canonical. (Prevents two competing truths for the same number.)

### transactions

One row per operational transaction. Approved target for the complete 2015–2024 canonical set: **approximately 36,000 transactions** (a target, not an absolute requirement — a slightly different deterministic count is acceptable if documented). This represents selected financial/operational transactions suitable for document intelligence, NOT every POS receipt.

| Field | Keep? | Notes |
|-------|-------|-------|
| `transaction_id` | ✅ | string PK, e.g. `TX-2019-000123`, chronological prefix aids range questions |
| `date` | ✅ | full date; drives monthly aggregation |
| `branch_id` | ✅ | FK → branches |
| `department_id` | ✅ | FK → departments |
| `transaction_type` | ✅ | enum: `sale` / `purchase` / `expense` / `adjustment` (minimal set) |
| `category` | ✅ | controlled vocabulary per type (e.g., expense → rent, salaries, utilities, logistics, marketing) |
| `vendor_customer` | ✅ | single counterparty field (supplier OR customer name, bilingual where rendered) |
| `description_en` | ✅ | short canonical description |
| `description_ar` | ✅ | aligned Arabic description (not always literal — see §5) |
| `amount` | ✅ | SAR, signed by convention: sales positive, purchases/expenses recorded positive with type implying direction (documented once, enforced always) |
| `tax_amount` | ✅ | VAT on the transaction per §3 VAT rule |
| `total_amount` | derived | `amount + tax_amount` (stored for rendering convenience, validated, never independently edited) |
| `payment_method` | ✅ | enum: `bank_transfer` / `cash` / `credit` (needed for realistic extracts; small vocabulary) |
| `reference_number` | ✅ | links to `invoices.invoice_id` where applicable, else null |

Excluded as unnecessary: approval workflows, approver names, cost-center hierarchies, multi-line transaction legs (line-item detail lives on the invoice document, not in canonical truth — one invoice ↔ one or more transaction rows via `reference_number`).

### budgets

Designed to serve budget-vs-actual questions with **no duplicated truth**:

| Field | Notes |
|-------|-------|
| `period` | quarter grain, APPROVED: `YYYY-Qn` (e.g. `2023-Q4`), covering 2015-Q1 … 2024-Q4 |
| `branch_id` | FK, nullable (company-wide budgets allowed) |
| `department_id` | FK, nullable |
| `metric` | controlled vocabulary: `revenue`, `cogs`, `operating_expenses`, `net_income` |
| `budget_amount` | the planned figure (the only stored amount) |

`actual_amount` is **NOT stored** — it is derived by aggregating `monthly_financials` (or `transactions`) over the budget period. Variance (`actual − budget`, `variance %`) is always computed, never stored. Rationale: storing both guarantees eventual contradiction; deriving keeps one truth and additionally exercises the SQL-calculation path the architecture requires.

### invoices

Invoices get their **own canonical entity** (not just generated paper), because exact-identifier retrieval (`INV-…` lookup via BM25) needs stable IDs with known totals independent of any single rendering.

| Field | Notes |
|-------|-------|
| `invoice_id` | string PK, e.g. `INV-2023-00417` — stable across regenerations |
| `invoice_date` | date within 2015–2024 |
| `vendor_customer` | counterparty name |
| `branch_id` | receiving/issuing branch |
| `subtotal` | SAR, pre-VAT |
| `vat` | SAR, APPROVED simplified flat 15% of subtotal (V1; intentionally not historical) |
| `total` | enforced: `subtotal + vat` |
| `status` | enum: `paid` / `pending` / `overdue` (as-of a fixed dataset "snapshot date" of 2024-12-31, so status is deterministic) |
| `related_transaction_id` | FK → transactions, nullable |

Volume: the ~30 invoice/receipt documents in the final corpus render from a subset of these rows; the table itself may hold more rows than rendered documents (unrendered rows are simply never asked about — but must still satisfy invariants).

### annual_balance_sheet

Small canonical entity for annual company-level balance-sheet truth. Grain: **one row per calendar year, company level — exactly 10 rows (2015–2024).** This is a stated-figures fixture, NOT a full accounting ledger (no double-entry, no cash-flow reconciliation in V1).

| Field | Notes |
|-------|-------|
| `year` | calendar year PK (2015–2024) |
| `cash` | SAR ≥ 0 |
| `accounts_receivable` | SAR ≥ 0 |
| `inventory` | SAR ≥ 0 |
| `other_current_assets` | SAR ≥ 0 |
| `property_and_equipment` | SAR ≥ 0 (net; grows with warehouse/Dammam capex years) |
| `total_assets` | **derived**, enforced: `cash + accounts_receivable + inventory + other_current_assets + property_and_equipment` |
| `accounts_payable` | SAR ≥ 0 |
| `debt` | SAR ≥ 0 |
| `other_liabilities` | SAR ≥ 0 |
| `total_liabilities` | **derived**, enforced: `accounts_payable + debt + other_liabilities` |
| `equity` | **derived plug**, enforced: `total_assets − total_liabilities` |

Required invariant: **`total_assets = total_liabilities + equity`** (exact to 0.01).

Generation approach: asset/liability components are scaled from that year's company revenue (from `monthly_financials` aggregates) with documented ratios plus capex step events (2017 warehouse expansion, 2019 Dammam fit-out); `equity` is the plug that enforces the invariant. Values must evolve plausibly with company growth and be internally consistent year over year (no unexplained jumps except documented capex years).

Purpose: authoritative truth for balance-sheet documents, extraction tests, calculation questions, and DEV-010.

### business_events

CRITICAL for explanatory RAG. This table is the ground truth behind every "why" question.

| Field | Notes |
|-------|-------|
| `event_id` | string PK, e.g. `EVT-2020-001` |
| `start_date` / `end_date` | inclusive bounds; single-day events have equal dates |
| `event_type` | controlled vocabulary (seed list below; extend only with approval) |
| `affected_branch` | FK → branches, nullable (= company-wide) |
| `affected_metric` | controlled vocabulary: `revenue`, `cogs`, `operating_expenses`, `logistics_cost`, `net_income`, … |
| `title_en` / `title_ar` | short aligned titles |
| `explanation_en` / `explanation_ar` | 2–5 sentence causal explanations; semantically aligned, not literal translations (§5) |
| `expected_effect` | machine-checkable direction + rough magnitude, e.g. `logistics_cost up ~20–30% during event window` (used to validate that generated numbers actually reflect the story) |

Seed event-type vocabulary (categories only — do NOT populate dozens of events yet):

- `branch_opening` · `warehouse_expansion` · `supply_cost_increase` · `temporary_closure` · `promotional_campaign` · `logistics_disruption` · `demand_shock`

Architectural role:

```text
business_events row (truth: cause + effect + wording)
    → shapes generator parameters (numbers move as the story says)
    → rendered into commentary/DOCX/KPI documents (evidence the retriever finds)
    → gold "why" answer = event explanation (what the grader checks)
```

This closes the loop that makes "Why did logistics expenses increase in Q3?" an objectively gradable question: the correct answer is the event's explanation, the supporting numbers come from `monthly_financials`, and both originate from the same approved fixture. A small curated set (propose ~8–12 events for the full 10 years, including the 2020 demand shock and the 2019 Dammam opening) is sufficient — one well-specified event beats ten vague ones.

## 3. Financial Consistency Rules

### Invariants (enforced by generation-time validation; violations fail the run)

```text
R1: gross_profit     = revenue − cogs                                        (exact to 0.01)
R2: operating_profit = gross_profit − operating_expenses                     (exact to 0.01)
R3: net_income       = operating_profit + other_income − finance_costs − tax_expense (exact to 0.01)
R4: invoice total    = subtotal + vat                                        (exact to 0.01)
R5: transaction total_amount = amount + tax_amount                            (exact to 0.01)
R6: branch-month sums = company-month sums (sum over branches = company total, exact to 0.01)
R7: month sums = year sums (sum of 12 branch-months = annual statement figure, exact to 0.01)
R8: budget variance  = actual − budget; variance_pct = variance / budget (budget ≠ 0)
R9: balance sheet: total_assets = cash + accounts_receivable + inventory + other_current_assets + property_and_equipment (exact to 0.01)
R10: balance sheet: total_liabilities = accounts_payable + debt + other_liabilities (exact to 0.01)
R11: balance sheet: total_assets = total_liabilities + equity (exact to 0.01)
```

### Sign rules

- Must never be negative: `revenue`, `cogs`, `operating_expenses`, `other_income`, `finance_costs`, `tax_expense`, `subtotal`, `vat`, `amount` (magnitude convention), `budget_amount` (revenues/costs abs; sign carried by metric semantics), all balance-sheet components.
- MAY legitimately be negative: `gross_profit` (distress months), `operating_profit`, `net_income`, `variance`, `variance_pct`.
- Transactions use **magnitude + type** convention: `amount ≥ 0` always; direction comes from `transaction_type`. Documented once, enforced by check constraint.

### Rounding and precision

- SAR precision: **2 decimal places** (halala) for all money fields.
- Generation rule: compute in full precision, round-half-up to 2dp at row write.
- Aggregation rule: **sum the rounded monthlies, then round** — annual figures are sums of stored monthly rows, never independently generated. (R7 enforced on rounded values, tolerance 0.01 × row count.)
- Evaluation tolerance (proposed, TBD per EVALUATION_PLAN.md): exact match after 2dp normalization for statement figures; ±0.5% relative tolerance only for OCR-read values where scan noise is the tested variable — never for native-extraction scoring.

### VAT: Option B APPROVED — simplified controlled 15% assumption (V1)

Saudi VAT actually changed (5% from 2018, 15% from July 2020). The approved V1 rule is **a single flat 15% on all rendered invoices and transaction tax fields, regardless of date**, recorded in the manifest as a known simplification. This is intentionally NOT a historically accurate Saudi tax model.

Why:

- V1 evaluates identifier lookup, total extraction, and arithmetic (`subtotal + vat = total`) — all testable with one rate.
- Historical rates would require date-dependent generator logic, date-dependent validators, and date-aware gold answers; a single off-by-cutoff bug would poison extraction scores with tax-history noise instead of measuring extraction quality.
- The real 5%/15% split is a *harder extraction case* worth adding later (e.g., `dataset_v0.2+` advanced set), not a V1 requirement.

Do NOT implement historically accurate VAT until the architect explicitly schedules it. If introduced later, it must come with its own dated rate table fixture and re-validation of all gold invoice totals.

## 4. Time-Series Design

Future monthly series (Jan 2015 – Dec 2024, **120 months**) should compose, per branch × metric:

```text
value(t) = baseline × growth(t) × seasonality(t) + noise(t) + events(t) + breaks(t)
```

- **Long-term growth:** gentle compound annual growth (~5–9% revenue, branch-specific), so multi-year trend questions have a true direction.
- **Seasonality:** annual cycle with food-retail shape — Ramadan uplift (lunar calendar drift is a feature: the peak month moves year to year, defeating naive month-dummy memorization), summer beverage peak, post-Ramadan dip. Quarterly budget grain must not alias the monthly cycle.
- **Random noise:** multiplicative lognormal-ish noise (CV ~3–8% by metric; opex noisier than revenue) — enough to require real smoothing, never enough to erase the signal.
- **Branch differences:** Riyadh largest, Jeddah mid, Dammam smallest with post-2019 ramp; distinct seasonal amplitudes per branch so branch-identity is recoverable from the series.
- **Category differences:** rent sleek/stepwise (lease steps), salaries smooth growth, logistics spiky (fuel + disruption events), marketing campaign-pulsed.
- **Exceptional events:** the ~8–12 `business_events` rows inject dated shocks/dips (2020 demand shock deepest) with magnitudes recorded in `expected_effect` so generation can be audited against intent.
- **Structural changes:** Dammam 2019 opening (level shift in company totals), one pre/post-2020 template rebrand (document-level only, not a numeric break).

Anti-triviality rule: **the series must NOT become artificially easy to forecast.** No perfectly smooth deterministic curves, no exact repeating seasonal templates, no noise-free validation windows. Concretely: holdout skill must separate models — seasonal-naive must beat plain naive, and fitted models must beat seasonal-naive by a *modest* margin, not by orders of magnitude. If a naive baseline achieves near-zero error on trial data, the generator's noise/seasonality parameters are wrong and must be re-tuned before any model comparison is trusted.

Balance target: **signal + seasonality + noise + known shocks** — a competent analyst (or ETS/SARIMA) should capture the shape with honest residual error; the LLM explains the validated forecast, never invents it.

**2024 is the approved default final forecasting holdout** (12 months). Walk-forward origins inside 2015–2023 serve model selection; 2024 is touched once for final scoring. The monthly series itself is generated as canonical truth in `dataset_v0.1` (this section specifies its required behavior).

## 5. Bilingual Ground Truth

### Alignment rule

Arabic and English explanations of the same event/record are **semantically aligned but NOT always literal translations.** Real bilingual business documents express the same event differently: an Arabic memo may cite the cause first and name the branch informally (فرع الدمام), while the English commentary leads with the metric impact and uses the formal ID (BR-DMM). Generation fixtures must encode this variation deliberately — same facts, different surface forms — so cross-language retrieval is tested on meaning, not on parallel-sentence matching.

Each `business_events` row therefore carries independent `explanation_en` / `explanation_ar` texts that agree on: event, dates, branch, metric, direction. They may differ in: sentence order, emphasis, formality, numeric expression (digits vs. words), and which secondary details are mentioned.

### Fair cross-direction evaluation

The four directions (AR→AR, EN→EN, AR→EN, EN→AR) are evaluated as separate categories with separately reported scores (see EVALUATION_PLAN.md §7), using question pairs that are *informationally equivalent* (same answer, same source rows) rather than string translations of each other. A system must not pass AR→EN by matching cognates while failing EN→AR on the same fact.

### Normalization issues (generator and evaluator must handle identically)

- **Digits:** Arabic-Indic (٠١٢٣٤٥٦٧٨٩) vs. Western (0–9) — both appear in corpus (native docs: Western; scanned Arabic docs: mixed). Canonical truth stores Western digits; a single shared normalizer (specified at implementation) maps both before numeric comparison.
- **Punctuation:** Arabic comma/semicolon/question mark (، ؛ ؟) vs. Latin equivalents — tokenizers and chunkers must not split entities on either.
- **Company/branch names:** `Noor Retail & Distribution Co.` ↔ `شركة نور للتجزئة والتوزيع` (exact approved renderings fixed in a vocabulary fixture; no ad-hoc transliteration at generation time).
- **Currency:** `SAR` / `ر.س` / `ريال` — all map to the dataset-level SAR constant.
- **Dates:** `2023-03-15` / `15/03/2023` / `١٥ مارس ٢٠٢٣` — all resolve to ISO dates; month names in both languages fixed in the vocabulary fixture.

## 6. Document Traceability

### Manifest (conceptual; specified at generation time)

```text
document_manifest
    document_id        (DOC-… stable; DEV-… for the 10 dev docs)
    filename
    document_type      (income_statement, invoice, commentary, …)
    year / period
    language           (en / ar / mixed)
    source_record_ids  (truth rows rendered here: month-keys, invoice_ids, event_ids)
    generation_template (template id + version, e.g. stmt_en_v1)
    dataset_version    (dataset_v0.1 / v0.2 / v1.0)
    content_hash       (sha256 of the artifact)
```

### The traceability chain

```text
canonical truth
        ↓  (renderer input: source_record_ids)
generated document
        ↓  (ingestion: page/element ids)
page / element
        ↓  (chunking: chunk ids + offsets)
retrieved chunk
        ↓  (generation: citation)
final citation
```

Every arrow must be a recorded mapping, not a guess: manifest (truth→document), ingestion log (document→page/element), chunk index (element→chunk with `document_id`, `page`, offsets), evidence package (chunk→answer). Citations resolve backward along this chain to truth rows.

### Why this matters for evaluation

Traceability lets us score the three stages **separately** instead of one blurred "answer looks good":

- **Extraction** scored truth-row ↔ page/element (did we read the number?).
- **Retrieval** scored question ↔ chunk/page (did we find the right evidence?).
- **Citations** scored answer-claim ↔ chunk ↔ truth row (does the cited source actually support the claim?).

Without the chain, a correct-sounding answer with a wrong citation is indistinguishable from a grounded one — and per the evaluation contract, that must count as an error.
