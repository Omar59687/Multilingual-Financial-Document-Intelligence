# G1 / G4 Baseline — MizanIQ Phase 3 (measured 2026-10-03)

> First baselines for Evaluation Plan gates G1 (extraction on DEV) and
> G4 (calculation exactness). Procedure-frozen, thresholds TBD →
> pending architect approval (human gate). Machine data:
> `docs/g1_baseline.json`, `docs/g4_calc_baseline.json` (miss lists
> capped at 50/doc; counts are complete).
> Method: `scripts/measure_g1.py` (gold via `src/evaluation/g1.py`)
> and `scripts/eval_calc.py` (support via `src/evaluation/calc.py`).
> Related: `docs/EVALUATION_PLAN.md` §§3/9/12 · `docs/STRUCTURED_EXTRACTION_DESIGN.md` §§14–22.

## 1. G1 procedure (frozen for this baseline)

- Gold: truth-recomputed from canonical CSVs (origin-1), scoped by
  rendered content per DEV fixture (origin-2); no GT values copied.
  Multiset occurrence matching (identical tuples from distinct rows
  are distinct facts); scan-noise band matches maximum-cardinality
  one-to-one per partial key (Kuhn augmenting-path; distance only
  breaks ties deterministically). "Source" is the document
  (document-level traceability shared across ingestion/gold/
  extraction; finer page/element alignment is not deterministically
  available across grains — residual same-valued swaps documented;
  architect to confirm at the gate).
- Field key `(document_id, metric, period, branch-or-None, value-2dp)`,
  multiset occurrence matching (identical tuples from distinct source
  rows are distinct facts). Branch exceeds the §3 minimum on purpose:
  right-number-wrong-branch fails.
- Value-exact at 2dp; the restricted ±0.5% band applies ONLY to the
  scanned slices DEV-004/DEV-010 (EVAL §3); a within-band value
  counts TP, otherwise exact-only.
- Out-of-scope grains (prose, visual-rounded, budget, derived,
  non-rendered context) are excluded from denominators and listed in
  §4 (EVAL §2: unverified items excluded from scores).
- Precision overall is computed over extracted records; recall over
  in-scope expected fields. DEV-005/DEV-008 carry zero in-scope
  fields and contribute nothing (shown as n/a, not 0).

## 2. G1 results

| doc | lang | route | expected | extracted | TP | P | R | F1 |
|-----|------|-------|----------|-----------|----|---|---|----|
| DEV-001 | en | statements (native PDF) | 21 | 18 | 18 | 1.000 | 0.857 | 0.923 |
| DEV-002 | ar | statements, empty map (Arabic deferred) | 12 | 0 | 0 | n/a¹ | 0.000 | 0.000 |
| DEV-003 | mixed | statements (native PDF) | 3 | 3 | 3 | 1.000 | 1.000 | 1.000 |
| DEV-004 | ar | paddle (scanned) | 3 | 3 | 3 | 1.000 | 1.000 | 1.000 |
| DEV-005 | mixed | none (prose, no route) | 0 | 0 | — | n/a | n/a | n/a |
| DEV-006 | en | native (XLSX) | 12060 | 10807 | 10807 | 1.000 | 0.896 | 0.945 |
| DEV-007 | mixed | native (CSV) | 1038 | 1038 | 1038 | 1.000 | 1.000 | 1.000 |
| DEV-008 | en | qwen (empty reading) | 0 | 0 | — | n/a | n/a | n/a |
| DEV-009 | mixed | qwen (verdict only; KPI visuals M-rounded, §4) | 0 | 0 | — | n/a | n/a | n/a |
| DEV-010 | en | paddle (scanned) | 11 | 11 | 11 | 1.000 | 1.000 | 1.000 |

¹ P undefined with zero extractions (0/0); shown n/a rather than 0.

- Overall (micro): P = 1.000, R = 0.904, F1 = 0.949 (TP 11880, FP 0, FN 1268).
- Per-language slices: en P 1.000 / R 0.896; ar P 1.000 / R 0.200;
  mixed P 1.000 / R 1.000. Minimum slice recall: ar 0.200.
- FP = 0 on every document: no invented field anywhere — the
  fail-closed design (validation gating, 2dp strictness, W_LABEL
  pass-through) holds empirically, not just by code review.

## 3. Recall-gap ledger (all 1268 FN, by cause)

| cause | count | docs | route status |
|-------|-------|------|--------------|
| Whole-amount native floats (`30.0` = 1dp verbatim, never padded) | 1253 | DEV-006 | fail-closed by frozen MoneyValue rule; honest skips with warnings |
| Text-rendered branch split (no table cells) | 3 | DEV-001 | no extraction rule exists for KPI text lines |
| Arabic statement labels (no frozen mapping) | 12 | DEV-002 | deferred Arabic workstream |

No numeric-mismatch FN exists anywhere: every extracted value that
had an expected counterpart matched exactly (numeric_accuracy 1.0 on
all scored docs). All recall loss is coverage loss, never wrong numbers.

## 4. Out-of-scope registry (excluded from denominators, listed)

- DEV-001 `derived-yoy`: YoY Change% is R-computed, not a field.
- DEV-002 `derived-annual-total`: headline annual opex is derived.
- DEV-002 `derived-categories` + `non-rendered-txns`: category sums are
  G4 aggregation territory; 372 txn rows feed aggregates only
  (explicit partial-extract fixture design).
- DEV-005 `prose-numbers`: commentary figures, no prose route.
- DEV-008 `visual-annual-10y`: M-rounded chart labels (approximate-only).
- DEV-009 `visual-kpi-rounded` + `visual-branch-bars-rounded`:
  M-abbreviated card/bar displays (approximate-only; exact canonical
  sums are not visible per the frozen visual-benchmark evidence —
  exact-match scoring here would demand hidden digits in reverse).
- DEV-009 `budget-grain`: budget_total/variance (no budget tables exist).
- DEV-010 `context-2019`: unevaluated comparative context (spec).

Human-verification backlog (EVAL origin-3): DEV-005 prose periods,
DEV-008/009 visual readings, DEV-002 category-aggregate association.
None blocks the in-scope gate; all recorded.

## 5. G4 calculation evidence (measured, procedure-fixed)

Truth-populated :memory: target (all canonical monthly rows minted as
eval records, 2024 included in the TARGET but every case stays
≤2023 — the 2024 holdout is never queried, trained on, or tuned
against), expectations independently recomputed in Python (Decimal
path, not SQL) over canonical CSVs:

| case | result |
|------|--------|
| company-total-revenue-2023 (12 monthly totals) | PASS |
| branch-comparison-revenue-2023-06 | PASS |
| yoy-revenue-2023-06-vs-2022-06 | PASS |
| margin-gross-2023-06 | PASS |
| budget-variance-revenue-2023-Q4 | PASS |
| branch-rank-gap-revenue-2023-06 | PASS |
| monthly-average-revenue-2023 | PASS |

7/7 PASS at near-zero logical error (Decimal cells exact; float cells
within ±0.01 representation tolerance). Combined with the P4
SQL-logic tests (decoy rows, NULLIF nulls, ordering), this is the G4
evidence; G4 has no TBD number to approve.

## 6. Holdout statement

- 2024 data: DEV-007 (2024-H1) extraction is MEASURED (no training
  involved); forecasting handoff excludes 2024 by default
  (`DEFAULT_END_PERIOD`); calc-eval cases use ≤2023 only. No model,
  threshold, or mapping was tuned on 2024.
- Threshold-setting must not overfit DEV-007: thresholds below are
  structural (precision-first + per-slice minima), not
  DEV-007-fitted values.

## 7. Threshold options for the architect (no decision made here)

- **Option A (uniform bar):** G1 PASS iff overall P ≥ 0.99 AND overall
  R ≥ 0.90 AND every language slice R ≥ 0.85. Current: P ✓, R ✓
  (0.904, thin margin), ar-slice ✗ (0.200) → FAIL → blocks scaling
  until the Arabic route exists.
- **Option B (grain-gated, recommended):** G1 PASS on in-scope grains
  iff P ≥ 0.99 AND R ≥ 0.90 (current: 1.000 / 0.904 ✓), WITH
  mandatory pre-registered follow-ups before G6 scaling rehearsal:
  Arabic statement labels, visual readings, prose/measurement
  policy — each with its own mini-gate. Matches the G6 structure
  (v0.2 rehearsal meets adapted G1–G4) and Rule 11 without halting
  all downstream work on the two deferred routes.
- **Option C (lower bar):** overall R ≥ 0.80 with no slice minimum.
  Risks repeating the multilingual-failure pattern EVAL §7 forbids
  (a blended score hiding ar 0.200); not recommended.

Recommendation: **B**. Rationale: precision is perfect (the
safety-critical property — never a wrong number), recall loss is
entirely characterized coverage loss on two explicitly deferred
routes, and G6 already requires re-measurement before scaling.

## 8. Exact decisions required (human gate)

1. Approve G1 thresholds by choosing **A**, **B** (with any adjusted
   numbers), or **C** — or direct a different bar with rationale. On
   approval, Phase 3 status resolves per the completion rule; on
   rejection/direction, the named follow-ups become the next bounded
   packages. G4 needs no numeric approval (reported 7/7 PASS).
2. Confirm the §1 "source = document" reading for G1 field matching
   (or direct finer page/element alignment as a follow-up package
   with its own gold derivation — note this would require inventing
   cross-grain alignment rules).
3. Prerequisite to (1): approve the DEV-009 scope decision — the 7
   M-abbreviated KPI/branch visuals are excluded from G1-field-exact
   as approximate-only (frozen visual-benchmark evidence: exact sums
   are not visible; scoring them exact would demand hidden digits in
   reverse), with approximate visual evaluation staying in the visual
   benchmark. Quantitative effect disclosed: exclusion moves overall
   recall 0.903 → 0.904 (7 FN removed from 1275). Alternative if
   rejected: score them against renderer-derived visible normalized
   values with display-rounded semantics as its own mini-package.

## 9. Architect gate resolution (2026-10-04, recorded outcome)

1. DEV-009 scope: **APPROVED** — the 7 values stay excluded from G1
   field-exact as approximate visual values; the exclusion remains
   explicitly documented (§4 registry + §8) and is preserved as a
   known visual/approximate-value limitation and follow-up. These
   cases do not disappear from future evaluation coverage.
2. Source granularity: **APPROVED as source=document** for the Phase 3
   G1 gate (recorded granularity; finer alignment may be a later
   evaluation improvement, no package created for it).
3. Threshold policy: **OPTION B APPROVED** — grain-level gate
   P ≥ 0.99 AND R ≥ 0.90. Measured P = 1.000, R = 0.904, F1 = 0.949:
   **G1 gate PASSES**. The Arabic slice (0.200) and visual/
   approximate limitations are explicitly NOT production-quality and
   are NOT averaged away: mandatory follow-ups before G6, recorded
   in §4 and below.

Final gate status: **G1 PASS** (Option B) · **G4 PASS** (7/7, no
numeric approval required). Mandatory follow-ups before G6 scaling
rehearsal: Arabic statement-label route, visual-reading route
(approximate-value semantics), prose/measurement policy — each with
its own pre-registered mini-gate at that time.
