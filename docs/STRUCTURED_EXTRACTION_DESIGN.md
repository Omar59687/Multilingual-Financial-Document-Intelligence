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
(`strict=True` escalates to `E_LABEL`). Balance-sheet metrics (Phase 3A):
`cash`, `accounts_receivable`, `inventory`, `other_current_assets`,
`property_plant_equipment`, `total_assets`, `accounts_payable`, `debt`,
`other_liabilities`, `total_liabilities`, `equity` — covering the canonical
`annual_balance_sheet` grain (DEV-010). Name note: the canonical CSV column
is `property_and_equipment`; the canonical metric is
`property_plant_equipment` (mapping at the label-normalization boundary,
`validation.normalize_bs_label`, never enum duplication). Deliberately NOT
added: `current_assets`, `current_liabilities`, `retained_earnings` (no such
columns in the canonical model — no canonical need); display-text variants
(`Cash`, `TOTAL-ASSETS`, …) are aliases in the normalization layer, never
enum members. The enum is P&L/invoice/budget/balance-sheet complete, not
DEV-overfit.

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
- Reconciliation evaluations report an output grade distinct from input
  precision: `exact-visible` (evaluated at tight ±0.01) vs `approximate`
  (evaluated under the widened rounded band, §10) vs `unknown`. `approximate`
  never claims exact equality — it records tolerance-consistent agreement.

## 6. Date / financial-period normalization

`src/extraction/periods.py` (stdlib, pure): four canonical grains, all
range-checked 2015-01-01…2024-12-31 (mirrors `config.START/END_YEAR`):

- exact date `YYYY-MM-DD` (transaction/invoice grain),
- month `YYYY-MM-01` (monthly aggregates; `YYYY-MM-01` strings are month
  grain, never day-1 facts),
- quarter `YYYY-Qn` (quarterly budgets; lowercase `q` accepted, canonical
  uppercased; bounds Q1 Jan–Mar … Q4 Oct–Dec, leap-aware month ends),
- year `YYYY` (annual/company grain),
- fiscal-year label `FY2024` / `FY 2024` (case-insensitive, Phase 3A):
  a display variant of the year grain for balance-sheet point-in-time
  contexts — canonicalizes to bare `YYYY` (never `YYYY-12-31`; no
  month/day invented), same 2015–2024 range check, Dammam rule compares
  the bare year.

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
image-wide visuals (PNG/JPG) and is **not** rejected. Optional
`extraction_route` (`native` | `light-ocr` | `paddle` | `qwen-visual`,
default `None`, Phase 3A) records the frozen Phase 2 routing-policy row
that produced the fact (native parser → `native`; Tesseract/light OCR →
`light-ocr`; PaddleOCR-VL-1.6 → `paddle`; Qwen3-VL-4B visual →
`qwen-visual`); explicit-kwarg only, never auto-detected, excluded from the
ID hash like all evidence fields. Stable
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
  gross/operating/net/variance may be. Balance-sheet (Phase 3A): all
  components nonneg **except `equity`** (the accounting plug, may be
  negative — mirrors the canonical validators).
- `E_MONEY`: exactly 2dp, `|x| < 1e12`, finite (NaN/Inf → error, never raise).
- `E_PRECISION`: exact claims on rounded/unparseable/missing displays
  rejected; suffixed rounded/value mismatch rejected.
- `E_PERIOD`/`E_BRANCH`: grammar (incl. `FY2024` year grain) + calendar +
  2015–2024 range + Dammam gate (FY compares bare year).
- `E_RECONCILIATION`: R1/R2/R3 (±0.01), R4 invoice, **R5 transaction**
  (`amount+tax_amount=total_amount`), budget variance + `variance_pct`
  (never divide by zero; pct must be `None` when budget is 0).
- Balance-sheet reconciliation (Phase 3A, precision-aware): R9
  (`cash+AR+inventory+other+PPE = total_assets`), R10
  (`AP+debt+other = total_liabilities`), R11
  (`total_assets = total_liabilities+equity`) via
  `check_bs_r9/r10/r11(fields, precisions)` returning
  `{rule, status, reason, metrics, precision}` with status PASS / FAIL /
  **NOT_EVALUATED** (incomplete inputs — absent keys — are not evaluated,
  never failures; present-but-unparseable is FAIL). Tolerance: all
  `exact-visible` → ±0.01; any rounded/unknown → widened
  `max(0.01, 0.005·peak)` band (widens, never fabricates) with output
  grade `approximate`/`unknown` (weakest wins; missing precision counts as
  unknown). Per-field precisions arrive via the `field_precisions` dict
  convention on wide record dicts (malformed ignored → unknown band).
  FAIL surfaces as `E_RECONCILIATION:R9|R10|R11`; PASS/NOT_EVALUATED silent.
- Label normalization: `normalize_bs_label` (casefold/collapse/parens-strip
  + 3-entry static alias table incl. the underscored GT column-name form;
  `None` when unmapped — never invented).
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
table — `actual` derived, variance computed. Balance-sheet metrics (§4) flow
through the same table with no DDL change (`metric` is an open VARCHAR at
the DB layer; the closed taxonomy lives upstream). Dammam pre-2019 absence enforced
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

1. ~~Foundation~~ — schemas, periods, provenance/FIN-*, validation,
   declarative target, design record. DONE.
2. ~~Phase 3A — balance-sheet coverage~~ — 11 metrics, R9/R10/R11 with
   precision-aware NOT_EVALUATED semantics, FY year grain, optional
   `extraction_route`, DEV-010 audit (all labels mapped, none unmapped). DONE.
3. **Next bounded task — DuckDB persistence implementation**: `connect` /
   `init_db` / `insert_records` (dedup `ON CONFLICT`, Decimal passthrough,
   ordered writes) with live-DB tests behind a `duckdb` extra. No taxonomy
   or rule changes.
4. Then: extraction runners per route (native/Paddle/Qwen field adapters
   emitting `FinancialRecord`s), DEV-set G1 measurement, calculation-query
   library over the live target, forecasting handoff (2024 holdout untouched).

## 16. Native-only runner contract (bounded slice, human-approved 2026-10-03)

NATIVE-ONLY scope: deterministic field adapter over native tabular
ingestion elements (`CSV_ROW` mappings + XLSX `SHEET_CELL` row-groups with
header row 1 as column names) emitting validated `FinancialRecord`s inside
`ExtractionResult`. Paddle/Qwen/G1/query-library/forecasting/taxonomy/
persistence changes are explicitly OUT. PDF/DOCX free-text and table-label
inference is deferred (those elements yield zero records + warnings).

1. Input: ingestion `Document` dataclass instance or plain dict from
   `Document.to_dict()` (requires `document_id` + `elements` list; any
   finite iterable with deterministic order is accepted, one-shot
   iterators are consumed). Anything else → `TypeError`.
   `IngestedDocument` unwrapping is caller duty.
2. Output: `ExtractionResult` (`document_id`, `records`, `warnings`, `ok`,
   finite `latency_ms ≥ 0`).
3. Supported elements: `CSV_ROW` (mapping in `value`) + XLSX `SHEET_CELL`
   row-groups (group by `(sheet, row)`, header = row-1 cell texts, required
   explicitly — a sheet without row 1 is skipped with a warning, never
   inferred from the minimum populated row; sheets emit in first-seen
   input order, rows/columns ascending). `TEXT`/`HEADING`/`TABLE`/
   `TABLE_ROW`/`TABLE_CELL`/`IMAGE_REF` and DOCX/PDF tables are unsupported
   in this slice (warnings, zero records).
4. Label → `Metric`: header key `strip().casefold()` in the `Metric` value
   set (identity; covers `amount`/`tax_amount`/`total_amount`,
   `subtotal`/`vat`/`total`), fallback `validation.normalize_bs_label`
   (covers the `property_and_equipment` alias). Unknown non-structural
   labels → one deduplicated `W_LABEL` warning per document header
   (structural columns such as IDs/dates/dimensions stay silent); records
   are never invented.
5. Values: verbatim display (`text` when non-empty else `str(value)` for
   native numerics); `FinancialRecord` 2dp coercion fail-closed (whole
   amounts such as `100.0` skip with warning, never padded); bool/None/
   empty skip, never zero; `Decimal` bound directly, never float.
6. Period: per-row `date` > `invoice_date` > `period` (first present) via
   `normalize_date_display` then `parse_period` (canonical + fiscal_year).
   Missing/unparseable → skip row with warning; no document-year inference
   (undated facts fail closed, §6); Dammam pre-2019 enforced at validation.
7. Provenance: `provenance.build_provenance(document, element,
   source_label=header as-stored, display_value as-stored,
   normalized_value, precision, branch/department passthrough)`; per-cell
   element for XLSX, row element for CSV; native row base preserved.
   As-stored means the ingestion-stored string (parsers strip outer
   whitespace; frozen Pydantic config strips outer on assignment) with
   inner spacing preserved end to end.
8. `extraction_route`: always `"native"` (explicit, excluded from FIN-ID).
9. Validation: build `FinancialRecord`, then
   `validation.validate_record(dict, strict=False)`; any errors → skip +
   warnings (incl. `warnings_for_record` `W_LABEL`). Invalid output never
   emitted. One-metric-per-record grain means cross-field R1–R11 rarely
   trigger here; reconciliation stays query-time.
10. Persistence: excluded. The runner never imports `duckdb`/`store`; the
    caller persists approved records via `store.insert_records`.
11. Malformed: wrong input type → `TypeError`; per-row failures → warnings
    + skip and continue (mirrors ingestion domain-failure style).
12. Ordering: input element order; sheets in first-seen input order with
    rows/columns ascending; CSV header order. FIN-IDs deterministic via
    `record_id_for`.
13. Tests: `tests/extraction/test_runner_native.py` uses synthetic fixtures
    only (no gold numerics); CSV + XLSX groups, BS alias, skips, ordering,
    envelope, `TypeError` guards, AST no-`duckdb` boundary.
14. Exclusions: Paddle/Qwen/G1/thresholds/query-library/forecasting,
    PDF/DOCX label inference, document-year inference, non-SAR currency,
    month-name dates, environment repair.

## 17. Paddle scanned-grid adapter contract (bounded slice)

Scope: deterministic field adapter over PaddleOCR-VL normalized result
shapes (`{document_id, tables: [grids], ...}` as produced by
`ocrbench/paddle_adapter.adapt_output` + `build_result`: lists of
row-lists of strings) emitting validated `FinancialRecord`s inside
`ExtractionResult` with `extraction_route="paddle"`. Covers scanned
grids whose rows are `(label, value)` pairs (DEV-010 balance sheet,
DEV-004 Arabic receipt). No inference, no weights, no GT reads.

1. Input: single result mapping with `document_id` + `tables`
   (list of grids); anything else → `TypeError`. Multi-result looping is
   caller duty.
2. Required explicit `document_period: str` (validated via
   `parse_period`, slash forms via `normalize_date_display`); missing/
   wrong-type → `TypeError`; unparseable → `ValueError` (caller contract
   violation, fail-fast — never inferred from filenames, GT, or text).
3. Optional explicit `branch_id` (default `None` = company-wide grain),
   `page`, `filename`, `file_type`, `language` — values only, never
   auto-detected; invalid `page` type → `TypeError`; invalid branch
   literal → per-record skip + warning (fail-closed, never silent None).
4. Rows: 2-cell → pair attempt; 1-cell non-empty → silent structural
   skip (section headers, stamps — correctly unmapped); 3+-cell →
   warning + skip; empty rows/cells → silent skip.
5. Label → `Metric`: identity casefold, fallback
   `validation.normalize_bs_label`, fallback paddle Arabic table
   (`المبلغ`→`amount`, `الاجمالي`→`total_amount`,
   `الضريبة`-prefixed→`tax_amount`, rate-agnostic). Unknown
   non-structural labels → deduplicated `W_LABEL` per document; records
   never invented. (Alias contents read from benchmark grid labels,
   names only — the §3A audit precedent; no numeric leakage.)
6. Values: value-cell verbatim display (currency tokens tolerated by
   the existing normalizer); 2dp fail-closed; bool/None/empty skip,
   never zero; `Decimal` directly, never float.
7. Provenance: `build_provenance` over a plain document dict with
   `element=None`, `table_index` = grid index, `row` = row index,
   `column` = 1 (value cell — real grid coordinates, never invented
   element IDs; `element_id` stays `None`). Identical pairs in one
   document therefore share a FIN-ID and dedup naturally on persist.
8. Validation, ordering (tables then rows), SAR-only, `ok`/latency
   envelope, `TypeError`-vs-warning split: identical to §16 items
   9/11/12 (record construction via `FinancialRecord`, gate via
   `validate_record(strict=False)`, deterministic input order).
9. Persistence excluded (caller uses `store.insert_records`); no
   `duckdb`/`store`/paddle imports anywhere in the module.
10. Tests: synthetic fixtures only; pairs, structural skips, Arabic
    aliases, unknown warnings, bad values, period/branch guards,
    determinism, AST boundary. Tesseract/`light-ocr` adapters remain
    OUT per the §15 route list.

## 18. Qwen visual-pairs adapter contract (bounded slice)

Scope: deterministic field adapter over Qwen visual pairs shapes (flat
`{label: raw-string}` maps as produced by
`ocrbench/qwen_adapter.extract_json_fields`: the `{"pairs":
[{"label","value"}]}` contract plus legacy flat maps; order preserved,
Unicode preserved) emitting validated `FinancialRecord`s inside
`ExtractionResult` with `extraction_route="qwen-visual"`. Covers KPI /
chart readings (suffixed `display-rounded` values supported where the
display fully determines the normalized figure, e.g. `23.90M`); verdict
or association text that maps to no metric warns and yields nothing.

1. Input: single result mapping with `document_id` + `fields` (flat
   string map; the `fields` key must be present as a mapping — absent or
   non-mapping `fields` is a malformed result shape → `TypeError`); an
   EMPTY map is an empty visual reading → zero records + warning.
   Anything else → `TypeError`. `tables` in Qwen results
   are association-scoring artifacts duplicating the fields verdict —
   a single informational warning per document, never parsed as
   readings. Multi-result looping is caller duty.
2. Required explicit `document_period`, optional explicit `branch_id`
   / `page` / `filename` / `file_type` / `language`: same fail-closed
   semantics as §17 items 2–3 (never inferred).
3. Pairs in map order; empty `fields` → zero records + warning
   (empty visual reading, never an error). Non-string values:
   verbatim `str()`; `None`/empty/bool → skip silently (absent, never
   zero). Nested structures are out (the pairs contract keeps values
   flat strings; non-string scalars stringify like the adapter).
   Foreign-currency displays (`$`/`USD`/`AED`/… substrings, mirroring
   the frozen schemas token set) are rejected before construction
   with a skip warning — an SAR record is never sourced from a
   foreign-denominated display (the shared normalizer strips those
   tokens silently for non-exact claims, so the adapter is the
   fail-closed boundary).
4. Label → `Metric`: identity casefold, fallback
   `validation.normalize_bs_label`. No Arabic receipt aliases (different
   source semantics — receipt Arabic stays Paddle-local). Unknown
   labels → deduplicated `W_LABEL`; never invented.
5. Values: verbatim display; precision via the existing normalizer;
   `display-rounded` suffixed readings emit only when the value is the
   exact magnitude expansion (coarse `24M`-style displays fail the 2dp
   gate and skip — hidden precision is never fabricated).
6. Provenance: `element=None`, no grid coordinates (pairs carry none —
   never invented); otherwise §17 item 7 verbatim (as-stored strings,
   FIN-ID content-address dedup documented).
7. Validation, ordering (map order), SAR-only, `ok`/latency envelope,
   `TypeError`-vs-warning split: §16 item 9/11/12 pattern (record via
   `FinancialRecord`, gate via `validate_record(strict=False)`).
8. Persistence excluded; no `duckdb`/`store`/transformers/`ocrbench`
   imports anywhere in the module.
9. Tests: synthetic fixtures only (identity, BS alias, suffixed
   rounded, unknown warnings, skips, guards, empty-fields, tables
   note, determinism, AST boundary).

## 19. Native statement-tables adapter + shared adapter helpers (bounded slice)

Scope: (a) deterministic adapter over native `TABLE_CELL` grids
(PDF lattice tables and DOCX tables: grouped by
page/table/row, first cell = label candidate, remaining cells =
`(column, value)` pairs) emitting validated `FinancialRecord`s with
`extraction_route="native"`; (b) extraction of the demonstrated-shared
thin helpers into `src/extraction/adapter_common.py` (third consumer —
the deferred-unification point recorded in §§16–18) with the
fleet-wide foreign-currency guard; native/paddle/qwen re-point to the
shared helpers (behavior preserved except the intended new skips).

1. Input: ingestion `Document`/dict (same duck-typed shape as §16.1)
   plus required explicit `column_periods: Mapping[int, str]` (value
   column index → period string; keys must be non-bool non-negative
   ints and values non-empty parseable period strings, validated
   upfront with fail-fast `TypeError`/`ValueError` — caller config like
   `document_period`, never inferred). Optional `label_column: int = 0`, explicit
   `branch_id` (default `None`), no department source (always None).
   Non-mapping `column_periods` → `TypeError`; empty mapping → zero
   records + warning (nothing mapped — never an error).
2. Rows: label cell empty → silent skip (spacers); label unmapped →
   deduplicated `W_LABEL` (header rows self-skip this way — no header
   detection invented); value columns absent from `column_periods` →
   silent structural skip (comparative `%` columns); per-cell value
   failures → skip + warning (sibling columns still emit).
3. Label → `Metric`: identity casefold, fallback
   `validation.normalize_bs_label`, fallback statements-local
   3-entry table (`cost of sales`→`cogs`, V1-frozen `vat 15%`→`vat`,
   `total due`→`total`; the generic
   casefold/collapse/parens-strip rule already covers the other
   statement labels — alias contents from GT label names, names
   only). Unknown → `W_LABEL`; never invented. Arabic statement
   labels and commentary prose remain deferred (no frozen rules).
4. Values/provenance/validation/ordering/SAR/`ok`/latency: §16
   pattern — verbatim display, shared foreign guard, 2dp fail-closed,
   real cell elements (`element_id` preserved → distinct FIN-IDs per
   row), per-cell page/table/row/column, `FinancialRecord` +
   `validate_record(strict=False)` gate, input group order with
   ascending columns.
5. Shared module `adapter_common.py` (stdlib + pydantic only, no
   `duckdb`/`store`/route imports): `METRIC_VALUES`,
   `map_identity_or_bs`, `derive_period`, `display_for`,
   `has_foreign_currency` (+ its token table mirrored from schemas).
   Native/paddle/qwen delegate their identical privates to it
   (wrappers kept where call sites exist); the foreign guard is wired
   into all record paths (the scheduled fleet fix — the only intended
   behavior change to the DONE slices, pinned by new tests).
6. Persistence excluded; import boundaries per §§16–18 hold for all
   four modules (common included).
7. Tests: synthetic fixtures only (statement grid with header
   self-skip + comparative columns + COGS alias + Change%-style skip,
   guards, determinism, AST boundary incl. common); additive
   foreign-guard tests in the native/paddle suites (no existing test
   altered).

## 20. Calculation-query library contract (bounded slice)

Scope: executable SQL builders for the Evaluation Plan §9 evaluated
operations over the live DuckDB target (`src/extraction/queries.py`),
with live-DB logic tests on synthetic validated records plus one
runner→store→query integration test. Provides the reviewable SQL-logic
evidence §9 demands (right number via wrong SQL still fails — tests
use decoy rows that wrong-SQL variants would miscount).

1. Reuse, don't duplicate: company totals and per-branch long-form
   comparison delegate to the frozen `store.sql_company_total` /
   `store.sql_branch_comparison` (read-only import, never modified).
   New builders only for uncovered §9 ops: YoY growth/difference
   (absolute + %, NULLIF zero prior), period averages (company scope
   averages monthly company totals — never a plain row average),
   margins/ratios (×100, NULLIF zero denominator), executable budget
   variance (actual aggregated over the window, budget bound client-side
   as an exact 2dp literal, variance + NULLIF pct — R8: never stored),
   and branch rank+gap (`RANK()` + leader gap via window functions).
2. Builders are pure string functions (like `store.sql_*` — never
   execute in-module); `TypeError` on wrong types (store convention),
   `ValueError` on non-2dp/non-finite budget literals. Literal
   escaping replicates the 6-line store escapers (attributed; single
   second consumer, no shared change to the DONE module).
3. R6/R7 semantics throughout: branch aggregates filter
   `branch_id IS NOT NULL` (company-source NULL rows never
   double-count — pinned by a decoy-row test); 2dp via
   `ROUND(..., 2)`; deterministic `ORDER BY`.
4. Tests (live `duckdb`, `importorskip` pattern): synthetic
   branch-month fixtures incl. NULL-branch decoy, zero-prior,
   zero-budget, ranking ties; each builder's result asserted AND its
   SQL-logic property (decoy exclusion, NULLIF nulls, ordering).
   Integration: synthetic CSV → `run_native` → `insert_records` →
   `sql_company_total` → expected sum.
5. Persistence/validation/taxonomy untouched; no `duckdb` top-level
   import in the module (builders are strings; tests import duckdb).
6. Tests use synthetic values only; no GT/canonical reads.

## 22. G1/G4 measurement harness contract (bounded slice)

Scope: evaluation-side measurement only (`src/evaluation/g1.py` gold
derivation + scoring, `src/evaluation/calc.py` truth-populated
calc-eval support, `scripts/measure_g1.py` + `scripts/eval_calc.py`
thin CLIs, `tests/eval/test_scoring.py` scorer math). First baselines:
`docs/G1_G4_BASELINE.md` + `docs/g1_baseline.json` +
`docs/g4_calc_baseline.json` (miss lists capped at 50/doc in JSON;
counts complete).

1. Gold derives from canonical CSVs re-read at runtime (origin-1),
   scoped by rendered content per DEV fixture (origin-2); no GT
   values copied. Field key `(doc, metric, period, branch-or-None,
   value-2dp)` as a MULTISET (identical tuples from distinct rows
   are distinct facts). Branch exceeds the §3 minimum deliberately.
2. Scoring: occurrence counts decide (TP += min, extras FP,
   shortfalls FN); miss listing classifies (invented / duplicate /
   numeric-mismatch with counterparts / missed); ±0.5% band ONLY for
   scanned slices (reclassifies mismatches to TP, tracked
   separately); out-of-scope grains excluded from denominators and
   registered with rationale (EVAL §2).
3. Extraction dispatch is harness-side per-doc configuration (GT
   metadata + spec shapes; adapters stay generic and GT-blind).
   Calc-eval mints eval records with distinct source element-ids
   (the frozen FIN hash excludes branch — same-valued branch facts
   would otherwise share IDs and dedup; extraction routes always
   carry distinct native elements, minting mirrors that).
4. No thresholds decided in code or scripts (TBD → architect gate);
   no training, no 2024 holdout use (calc cases ≤2023; handoff
   default excludes 2024; DEV-007 is measured, never trained on).
5. Scripts tolerate the known pandas/NumPy env breakage via an
   up-front silent import block (same fallback path, documented —
   no env repair).

## 21. Forecasting-handoff contract (bounded slice)

Scope: deterministic monthly time-series preparation from validated
DuckDB records (`src/extraction/series.py`) — the Phase 3 → Phase 8
handoff. No models, no metrics, no training here (all Phase 8).

1. One function: `fetch_monthly_series(connection, metric, *,
   branch_id=None, end_period="2023-12-01")` → ascending
   `[(period, Decimal)]` at month grain. Company scope (`None`)
   sums branch rows only (R6 — NULL-branch company rows excluded);
   a branch scope reads that branch. Periods filter `period LIKE
   '____-__-01'` (month grain) and `period <= end_period`
   (lexicographic ISO); `end_period` must parse as month grain via
   `parse_period` (kind `"month"`), else `ValueError`. Default end
   excludes the 2024 forecasting holdout (`DESIGN §15`, Evaluation
   Plan §10); callers opt into later windows explicitly.
2. Aggregation months with no rows are absent (never zero-filled —
   the Dammam pre-opening NULL rule flows through: no rows, no
   points). 2dp exact `Decimal` values (`ROUND(..., 2)`); ascending
   period order; deterministic.
3. `TypeError` on wrong types (store convention). No `duckdb`
   top-level import (callers/tests own the live dependency).
4. Tests (live `duckdb`, `importorskip`): synthetic branch-month rows
   incl. NULL-branch decoy exclusion, 2024 default exclusion +
   explicit inclusion, absent-month absence, Decimal exactness,
   ordering, guards. Synthetic values only.
