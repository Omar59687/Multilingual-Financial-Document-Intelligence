# Structured Financial Extraction — Design (MizanIQ Phase 3 Foundation)

> Phase 3 foundation design record. Documents the ACTUAL integrated design in
> `src/extraction/` (schemas, periods, provenance, validation, declarative
> DuckDB target), not an aspirational design.
> Related: `docs/ROADMAP.md` (Phase 3) · `docs/ARCHITECTURE.md` (structured DB)
> · `docs/DEVELOPMENT_RULES.md` (Rules 8/10) · `docs/CANONICAL_DATA_MODEL.md`
> · `docs/EVALUATION_PLAN.md` (§§3/9) · `docs/PHASE_2_OCR_VISION_EXIT.md` (§7)
> · `docs/INGESTION_DESIGN.md` · `docs/OCR_VISION_DESIGN.md`

## 1. Phase 3 objective

Define Pydantic-validated structured financial records
(`metric / period / value / currency / source document / source page`), validate
all AI-generated structured data, and specify — but not yet implement — the
DuckDB target that will hold approved records for deterministic SQL
calculations (growth/difference, sums/averages, margins, budget variance,
branch comparisons per Evaluation Plan §9). This foundation milestone covers
schemas, periods, provenance/IDs, validation, and the declarative target
shape. Runtime persistence (connect/init/insert, live DuckDB) is explicitly
deferred to the next bounded task.

## 2. Input contracts from Phase 1 / Phase 2

**Phase 1 native ingestion** (`src/ingestion/model.py`): every source arrives
as a `Document` (stable `document_id`: frozen `DEV-###` prefix else
`DOC-<12hex>` of sha256; `filename`, `file_type` pdf/docx/xlsx/csv/png/jpg,
`language_hint` ar/en/mixed/unknown, page/sheet counts, `status`
native_ok/partial_native/requires_ocr/requires_visual/failed/unsupported,
`requires_ocr`/`requires_visual` flags, `elements`, `latency_ms`) plus flat
`Element`s (`element_id` deterministic, 8-type closed set, page/sheet/row/
column/table coordinates with format-dependent row bases, verbatim `text` or
native `value`). Phase 3 consumes exactly these IDs/coordinates for
provenance and never re-parses source bytes.

**Phase 2 OCR/vision** (routing policy frozen in `PHASE_2_OCR_VISION_EXIT.md`
§6): native → light OCR → PaddleOCR-VL-1.6 (Arabic/layout/tables) → Qwen3-VL-4B
(charts/KPI visuals), escalate only when required. Provenance policy (§7,
`src/ocrbench/canonical.py`): every numeric carries `{display_value,
normalized_value, source_label, precision}` with `precision` in
`exact-visible | display-rounded | unknown`; `23.90M SAR` → `23900000`
(`display-rounded`), never inflated to hidden exact; unparseable → `None`
(never zero); unknown labels stay in `unmapped`. Phase 3 inherits this
vocabulary verbatim.

## 3. Structured record types

- `FinancialRecord` (`src/extraction/schemas.py`, Pydantic v2, frozen,
  `extra=forbid`): `record_id` (FIN-*, §9), `metric` (closed enum, §4),
  `period` (canonical string, §6), `fiscal_year` 2015–2024 (cross-checked
  against `period` when parseable), `value` (2dp SAR `Decimal`, §5),
  `currency` (SAR only, §7), `branch_id` / `department_id` (optional
  dimensions, §8), `provenance` (required `Provenance`, coerced from dict),
  `document_id` / `page` / `source_label` / `display_value` / `precision`
  (mirrored with `Provenance`, equality-enforced).
- `Provenance` (`src/extraction/provenance.py`, frozen/forbid): §9.
- `ExtractionResult`: per-document envelope (`document_id`, `records`,
  `warnings`, `ok`, finite `latency_ms ≥ 0`).

## 4. Metric taxonomy

Closed `Metric` enum (schemas.py): P&L `revenue`, `cogs`, `gross_profit`,
`operating_expenses`, `operating_profit`, `other_income`, `finance_costs`,
`tax_expense`, `net_income`; budget `budget_total`, `budget`, `budget_amount`,
`actual`, `variance`, `variance_pct`; invoice/transaction `subtotal`, `vat`,
`total`, `invoice_subtotal/vat/total`, `amount`, `tax_amount`, `total_amount`.
Phase 2 `branch_revenue_BR-*` keys map to `metric=revenue` + `branch_id`
(branch is a dimension, not a metric). Unknown labels are never invented:
schemas rejects them; validation passes them through with `W_LABEL`
(`strict=True` escalates to `E_LABEL`). Balance-sheet component metrics
(cash, receivables, …) are intentionally deferred — DEV-010/annual-sheet
grain is specified in the DuckDB target (§12) but no Pydantic metric exists
until the bounded balance-sheet task. The enum is P&L/invoice/budget
complete, not DEV-overfit.

## 5. Numeric normalization policy

Project policy: **ROUND_HALF_UP** for normalization of source-visible
financial numbers (matches the dataset generator, dataset validators,
`config.py` money policy, and all of `src/extraction`). `src/ocrbench/
canonical.py` was unified from implicit HALF_EVEN to explicit HALF_UP after
verifying all 52 display tokens in saved Phase 2 benchmark results are
exact-2dp or integer-scaled (zero halfway cases — no saved output changes;
see scope-correction report). Scorer internals (`normalize.py`,
`metrics.py` display helpers) were deliberately left untouched (frozen
scorer boundary; byte-identical on all real data).

Rules (never manufacture precision):

- exact-visible numbers remain exact: `value` must carry exactly 2dp
  (exponent -2, fail-closed — `"100"`, `"100.1"`, `"100.123"`, int `100`
  are rejected, never padded/rounded into records).
- `exact-visible` additionally requires: parseable display, no K/M/B suffix,
  display exactly 2dp, and `value == display` (exponent comparison, not
  numeric `==`, so `"100"` cannot masquerade as `100.00`).
- Suffix-expanded `display-rounded` values preserve visible precision:
  `23.90M SAR` normalizes to `23900000.00` (magnitude expansion, then 2dp);
  a `display-rounded` record whose `value` disagrees with its normalized
  display is rejected (`rounded_value_mismatch` / `E_PRECISION`) — hidden
  digits cannot be laundered through the rounded class.
- Unparseable/foreign-currency displays yield no exact claim (foreign tokens
  `USD/$/AED/…` are `foreign-currency` errors under SAR-only V1); missing
  numerics are `None`/absent — **no invalid/unparsed numeric ever becomes
  zero** (strict missing-value behavior, §10).
- Halfway boundary (`1.005` → `1.01` HALF_UP vs `1.00` HALF_EVEN) is pinned
  by `tests/extraction/test_rounding_policy.py` so the conflict cannot return.

## 6. Date / financial-period normalization

`src/extraction/periods.py` (stdlib, pure): four canonical grains, all
range-checked 2015-01-01…2024-12-31 (mirrors `config.START/END_YEAR`):

- exact date `YYYY-MM-DD` (transaction/invoice grain),
- month `YYYY-MM-01` (monthly aggregates; `YYYY-MM-01` strings are month
  grain, never day-1 facts),
- quarter `YYYY-Qn` (quarterly budgets; lowercase `q` accepted, canonical
  uppercased; bounds Q1 Jan–Mar … Q4 Oct–Dec, leap-aware month ends),
- year `YYYY` (annual/company grain).

`parse_period` returns `{kind, iso_start, iso_end, fiscal_year, canonical}`;
`month/quarter/year_bounds` give inclusive ISO bounds; `normalize_date_display`
handles `YYYY-MM-DD`, `YYYY/MM/DD`, `DD/MM/YYYY` (day-first) plus
Arabic-Indic digits (month names out of scope by design). Dammam `BR-DMM`
pre-2019 periods are not-applicable/NULL — validation rejects them
(`E_BRANCH:dammam_pre_opening`), never zero-fills. **No ambiguous period is
converted into an invented exact date**: bare quarters (`Q3`), slash dates
fed to `parse_period`, impossible dates, and out-of-range years raise
`ValueError`/`E_PERIOD`. There is deliberately no `unknown`/document-level
or year-range period kind — an undated fact fails closed; multi-year spans
are query windows over dated records, never a single invented period.

## 7. Currency handling

SAR only (V1, dataset-level constant per canonical model). `normalize_
currency` accepts `sar` (case-insensitive, trailing-period tolerant) and
`ر.س` / `ريال`; everything else raises. A missing currency token on a
display does **not** mean unknown — the dataset SAR constant applies.
`currency: unknown` does not exist at foundation level: it is allowed
nowhere, because no legitimate V1 route produces a non-SAR fact. Displays
carrying foreign tokens are rejected for exact claims (schemas) and never
valued (validation treats them as unparseable → `None`, never zero).

## 8. Dimension handling

Optional dimensions: `branch_id` (`BR-RUH`/`BR-JED`/`BR-DMM` or `None` =
company-wide), `department_id` (`DEP-*` or `None`), plus sheet/row/column/
table context in provenance where the format provides it. Unknown branch IDs
raise (never silent `None`); exactly-one-grain budget rules and
department-opex-only slices are validated at the dataset layer and
re-checked at extraction validation. `None` means company-wide/source-wide,
never "unknown".

## 9. Provenance model

Provenance authority: **`Provenance` is the single traceability model**
(`FinancialRecord.provenance` coerces dicts through it; top-level
`document_id/page/source_label/display_value/precision` must equal the nested
model). Fields: always `document_id` (`DEV-###`|`DOC-<12hex>`), `source_
label`, `display_value`, `precision`, `normalized_value` (`None` never zero);
"where applicable" (Rule 10) `filename`, `file_type`, `page` (≥1),
`sheet/row/column/table_index`, `element_id`, `branch_id`, `language`,
`doc_type` — document-level provenance with no page is legitimate for
image-wide visuals (PNG/JPG) and is **not** rejected. Stable
source-location identity: coordinates are preserved in native base and never
re-based across formats. Deterministic IDs: **FIN-*** canonical only
(`FIN-<12 lowercase hex>` = sha256 over canonical JSON of the normalized
core-6 `{document_id, page, element_id, metric, period, value}` with sorted
keys, compact separators, UTF-8). Evidence fields are excluded from the hash,
so re-extraction with richer evidence yields the same ID; values quantize to
2dp and periods/metrics normalize before hashing (`100`/`100.00`,
`2023-q1`/`2023-Q1`, `Revenue`/`revenue` converge). `REC-*` was retired —
no second namespace exists.

## 10. Validation rules

`src/extraction/validation.py` (stdlib+decimal, plain-dict in/out,
fail-collecting with stable `E_*` prefixes, fixed stage order
provenance→branch→period→money→sign→precision→reconciliation→label):

- `E_SIGN`: revenue/cogs/opex/other_income/finance/tax/subtotal/vat/amount/
  tax_amount/total_amount/total/budget(_amount)/actual never negative;
  gross/operating/net/variance may be.
- `E_MONEY`: exactly 2dp, `|x| < 1e12`, finite (NaN/Inf → error, never raise).
- `E_PRECISION`: exact claims on rounded/unparseable/missing displays
  rejected; suffixed rounded/value mismatch rejected.
- `E_PERIOD`/`E_BRANCH`: grammar + calendar + 2015–2024 range + Dammam gate.
- `E_RECONCILIATION`: R1/R2/R3 (±0.01), R4 invoice, **R5 transaction**
  (`amount+tax_amount=total_amount`), budget variance + `variance_pct`
  (never divide by zero; pct must be `None` when budget is 0).
- `E_PROVENANCE`: required traceability keys, precision vocab, page ≥ 1.
- Labels: passthrough `W_LABEL`, strict-mode `E_LABEL`. No silent zero
  anywhere; no `bool` numerics; deterministic ordering throughout.

## 11. Duplicate / conflict handling

Identity = `FIN-*` content address. Re-extraction of the same normalized
fact yields the same ID (idempotent; future persistence dedups on PK).
Conflicting aliases in one input (`period` vs `period_canonical`, `value` vs
`value_str` vs `normalized_value`) raise `ValueError` instead of picking a
winner. Top-level vs nested provenance disagreements fail record validation.
Budget `actual` is derived, never stored, so stored truth cannot contradict
itself (R8). `validate_batch` disambiguates colliding `record_id`s as
`{id}#{index}` with input order preserved.

## 12. Proposed DuckDB target shape

Declarative only (`src/extraction/store.py`: `SCHEMA_SQL`, `TABLE_NAME`,
`VIEW_NAME`, `COLUMNS`, `ddl_statements`, documented SQL-string patterns —
no connect/init/insert, no `duckdb` dependency):

```sql
CREATE TABLE IF NOT EXISTS financial_records (
    record_id VARCHAR PRIMARY KEY,          -- FIN-* content address
    metric VARCHAR NOT NULL,                -- §4 taxonomy (validated upstream)
    period VARCHAR NOT NULL,                -- §6 canonical string
    fiscal_year INTEGER NOT NULL CHECK (fiscal_year BETWEEN 2015 AND 2024),
    "value" DECIMAL(18, 2) NOT NULL,        -- quoted: keyword-fragile
    currency VARCHAR NOT NULL DEFAULT 'SAR' CHECK (currency = 'SAR'),
    branch_id VARCHAR CHECK (branch_id IS NULL OR branch_id IN ('BR-RUH','BR-JED','BR-DMM')),
    department_id VARCHAR,                  -- DEP-* or NULL
    document_id VARCHAR NOT NULL,           -- DEV-### | DOC-<12hex>
    page INTEGER CHECK (page IS NULL OR page >= 1),
    source_label VARCHAR NOT NULL,
    display_value VARCHAR NOT NULL,
    "precision" VARCHAR NOT NULL CHECK ("precision" IN ('exact-visible','display-rounded','unknown')),
    created_batch VARCHAR
);
CREATE INDEX ... (metric, period); (branch_id, period); (document_id);
CREATE OR REPLACE VIEW v_monthly_company_totals AS
  SELECT metric, period, ROUND(SUM("value"),2) AS company_total, COUNT(*) AS branch_count
  FROM financial_records WHERE branch_id IS NOT NULL GROUP BY metric, period;
```

R6: company totals sum branch rows (company-source `NULL`-branch rows
excluded, never double-counted; read company grain via `branch_id IS NULL`
directly, never from the view). R7: month sums roll into years. R8: no budget
table — `actual` derived, variance computed. Dammam pre-2019 absence enforced
at validation, not by trigger. `ON CONFLICT(record_id) DO NOTHING` is the
specified (not yet implemented) dedup spelling.

## 13. Error handling

Fail-closed everywhere: invalid periods raise / `E_PERIOD`; bad branches
`E_BRANCH`; non-2dp/non-finite money `E_MONEY` (never rounded, never zeroed);
sign violations `E_SIGN`; precision inflation `E_PRECISION`; reconciliation
drift `E_RECONCILIATION`; bad `tol` raises `ValueError`; conflicting ID
aliases raise; provenance/record disagreements fail validation. Validators
return error lists (never raise on content); parsers raise `ValueError` with
exact-shape messages. No exception message embeds ground truth.

## 14. Evaluation strategy

Per Evaluation Plan §§3/9, scored against truth-recomputed gold (never
self-graded from DuckDB): field-level precision/recall/F1 (a field counts
only if metric+value+period+source all match); numeric accuracy at 2dp
(±0.01 representation tolerance, ±0.5% only for OCR-noise slices); exact
match for identifiers; every numeric miss listed expected-vs-extracted;
calculation suite (growth/diff, sums/averages, margins, budget variance,
branch comparisons) at near-zero logical tolerance with SQL-logic review
(right number via wrong SQL still fails); multilingual slices reported
separately; quality gate G1 (extraction on DEV) blocks scaling. Foundation
tests pin determinism, provenance safety, and the rounding boundary.

## 15. Phase 3 implementation plan

1. ~~Foundation (this milestone)~~ — schemas, periods, provenance/FIN-*,
   validation, declarative target, design record. DONE.
2. **Next bounded task** — balance-sheet metric extension + persistence
   implementation: add cash/receivables/inventory/other-assets/PPE/payables/
   debt/other-liabilities/total-assets/total-liabilities/equity metrics with
   R9–R11 checks, then implement `connect`/`init_db`/`insert_records`
   (dedup `ON CONFLICT`, Decimal passthrough, ordered writes) with live-DB
   tests behind a `duckdb` extra. See report §L.
3. Then: extraction runners per route (native/Paddle/Qwen field adapters
   emitting `FinancialRecord`s), DEV-set G1 measurement, calculation-query
   library over the live target, forecasting handoff (2024 holdout untouched).
