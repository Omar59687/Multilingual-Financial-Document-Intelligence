# Phase 2 OCR/Vision Exit Review — MizanIQ

Branch: `phase-2/ocr-vision`. Frozen truth, fixtures, and scorers unchanged
throughout. All figures below are measured from saved repository artifacts,
not model brochures.

## 1. Phase objective

Prove which extraction route to use for each document class — native text,
scanned text, complex Arabic/layout/tables, and chart/KPI visual semantics —
and freeze a routing policy later pipeline phases can rely on. Design:
`docs/OCR_VISION_DESIGN.md`.

## 2. Frozen benchmark corpus

`dataset_v0.1`: 10 DEV documents + ground truth (`data/dev/dataset_v0.1/`,
manifest-SHA verified), benchmark workspace `data/benchmark/dataset_v0.1/`
(6 primary/secondary inputs incl. DEV-008 chart, DEV-009 KPI dashboard).
No frozen file was modified in Phase 2 (SHA checks in
`tests/ocr_bench/test_qwen_pairs.py::test_17`).

## 3. Models / extraction routes evaluated

- Native parsers (Phase 1, preserved as preferred route).
- Tesseract/light OCR (Kaggle CPU baseline, `eng` fallback — Arabic
  traineddata absent).
- PaddleOCR-VL-1.6 (`PaddleOCRVL(pipeline_version="v1.6")`, PaddlePaddle
  3.2.1 / PaddleOCR 3.7.0, Kaggle Tesla T4).
- Qwen3-VL-4B-Instruct visual route (explicit selection, fallback OFF;
  Kaggle Tesla T4, `max_new_tokens=1024`, `{"pairs": [...]}` contract).
- Execution automation: Kaggle CLI submitter (`scripts/kaggle_submit_visual.py`,
  `src/ocrbench/kaggle_runner.py`); Modal runners retained for future paid
  GPU use (currently billing-blocked on this account).

## 4. Measured results

Tesseract / light OCR (`notebooks/README.md` §reference run):
DEV-004 — identifier text 1.0, numeric text exact 1.0, anchor recall 0.4,
structured 0, ~603 ms. DEV-010 — numeric text exact 1.0, anchor recall 1.0,
table label text recall 1.0, table association 0, structured 0, ~856 ms.
Fast recognition baseline; weak on Arabic and structure.

PaddleOCR-VL-1.6 (`data/benchmark/results/paddle_results.json`):
DEV-004 — identifier text 1.0, numeric text exact 1.0, anchor recall 1.0,
semantic/canonical fields 0, ~8.6 s. DEV-010 — numeric text 1.0, anchors 1.0,
table label recall 1.0, table association 1.0, semantic/canonical fields 0,
~11.3 s. Materially better Arabic/layout/table recovery than Tesseract at
~10× latency; canonical semantic fields remain 0 by design (recognition +
structure only).

Qwen3-VL-4B confirmatory run
(`data/benchmark/results/kaggle_qwen_qwen3-vl-4b_20261003T064217Z_omarabdallah12-mizaniq-qwen-visual/`,
run log `final_status: COMPLETE`, scorer exit 0, zero schema violations):
DEV-008 — chart label recall 0.8333, chart numeric exact 1.0, chart
association 1.0, trend semantic accuracy 0.2, ~121.3 s. DEV-009 — KPI label
recall 1.0, KPI numeric exact 0.9, KPI association 1.0, budget status 1.0,
~86.3 s.
Key finding: the earlier association=0 was a pipeline-contract artifact
(literal `"LABEL"` placeholder keys collapsing + a 512-token truncation),
not a recognition failure — the corrected pair schema scores 1.0 on both
documents from the models' own explicit pairs.

## 5. Failure modes discovered

Infrastructure (not model quality): Kaggle `/kaggle/input` is read-only —
all writes must target `/kaggle/working` (first Qwen run failed with
`Errno 30` after successful generation); Qwen placeholder `{"LABEL": …}`
duplicate-key collapse; 512-token cap truncating structured output;
correct prose scoring 0 structured association under a wrong adapter
contract; Windows cp1252 breaking Unicode scorer output (fixed via
`-X utf8` + `PYTHONUTF8=1`); Tesseract Kaggle env lacking Arabic
traineddata; Modal A10G/T4 GPU allocation requiring a payment method.
Model-quality notes: Paddle chart-semantic execution impractical on the
tested T4 setup (>20 min stall on DEV-008 under the 300 s bound); Qwen
trend wording uses non-keyword equivalents (`upward`/`steadily` vs expected
`growth`/`steady`) — semantic-equivalence gap, scorer unchanged by design.

## 6. Final routing policy

| # | Document condition | Route |
|---|---|---|
| 1 | Native-readable PDF/DOCX/XLSX/CSV | Native parser; OCR/vision skipped unless native fails quality checks |
| 2 | Simple scanned text, OCR-only | Tesseract/light OCR first |
| 3 | Complex Arabic / mixed-language / layout-heavy / scanned table | PaddleOCR-VL-1.6 |
| 4 | Charts / dashboards / KPI visual semantics | Qwen3-VL-4B visual route |
| 5 | Visible rounded finance values | Canonical normalization + provenance (`src/ocrbench/canonical.py`) |
| 6 | Exact hidden precision | NEVER inferred from rounded display |
| 7 | Escalation | native → light OCR → Paddle layout OCR → Qwen semantic visual; escalate only when required |

Paddle is NOT the chart-semantic default (T4-impractical). Qwen is NOT the
default for every document (high latency: ~86–121 s vs <1 s Tesseract).

## 7. Canonical normalization policy

`src/ocrbench/canonical.py` maps explicit visible pairs to canonical keys
via static design-authored aliases (no runtime truth reads) and normalizes
display numerics with provenance `{display_value, normalized_value,
source_label, precision}`. `23.90M SAR` → `23900000`, precision
`display-rounded` — NEVER inflated to a hidden exact figure. Precision
classes: `exact-visible` (fully specified, e.g. `+52,904.62`),
`display-rounded` (suffixed/rounded), unparseable → `None` (never zero).
Unknown labels stay in `unmapped`, never dropped or invented. Consequence:
rounded dashboard values legitimately earn association/display credit but
cannot earn canonical exact credit — exactness requires exact-visible
precision.

## 8. Latency observations

Tesseract ~0.6–0.9 s; PaddleOCR-VL ~8.6–11.3 s per document (init/download
recorded separately); Qwen3-VL-4B ~86–121 s per visual document on Kaggle
T4. Latency is a routing criterion: expensive routes run only on escalation.

## 9. Known limitations

- Qwen trend-keyword recall 0.2 (vocabulary equivalence, scorer frozen).
- KPI numeric exact 0.9 (missing `2023` display token only).
- 5 strict hallucinated DEV-009 labels audited as visible-text artifacts
  (colon-suffixed labels, echoed status lines), not fabrications.
- Qwen 8B primary unrun on available hardware (T4 OOM risk); 4B selected
  explicitly with fallback OFF.
- Modal GPU path retained but billing-blocked; Kaggle free quota is the
  current execution route.

## 10. Phase 2 exit criteria

- native routing preserved — PASS (Phase 1 suite green, no ingestion changes)
- light OCR baseline measured — PASS (§4 Tesseract)
- complex OCR/layout route measured — PASS (§4 Paddle)
- visual semantic route measured — PASS (§4 Qwen confirmatory run)
- Arabic behavior measured — PASS (DEV-004 AR + mixed docs)
- table association measured — PASS (DEV-010 1.0 Paddle)
- chart association measured — PASS (DEV-008 1.0 Qwen)
- KPI association measured — PASS (DEV-009 1.0 Qwen)
- latency recorded — PASS (§8, per-document `latency_ms`)
- hallucination metrics recorded — PASS (label/numeric counts per run)
- provenance policy defined — PASS (§7)
- canonical normalization implemented — PASS (`canonical.py` + 11 tests)
- deterministic tests pass — PASS (203/203, §11)
- no frozen truth modifications — PASS (SHA-pinned test)
- routing policy frozen — PASS (§6 + decision log below)

## 11. Final phase status

**PHASE 2 — COMPLETE.** Every required exit criterion is met with saved
evidence. Decision log: P2-D01 native extraction remains preferred
(fastest exact route); P2-D02 Tesseract retained as lightweight baseline
(§4 sub-second recognition); P2-D03 Paddle retained for Arabic/layout/table
scans (DEV-010 association 1.0); P2-D04 Paddle not a chart-semantic default
(T4-impractical); P2-D05 Qwen3-VL-4B for charts/KPI visuals (association 1.0
both docs); P2-D06 explicit pair schema for visual structured output
(placeholder collapse post-mortem); P2-D07 canonical normalization preserves
visible precision; P2-D08 rounded displays never gain fabricated precision;
P2-D09 no single combined OCR score (layered metrics only); P2-D10
escalation-based routing, never one-model-for-all.
