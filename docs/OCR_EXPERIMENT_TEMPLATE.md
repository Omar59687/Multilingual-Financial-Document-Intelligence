# OCR Experiment Report Template (reusable)

> Copy this file per experiment run into `data/benchmark/results/` notes
> (git-ignored) or the experiment log. One file per (model × run).
> Do not declare a winner here — that happens in a comparison review
> across reports, one variable at a time (EVALUATION_PLAN §13).

## Identity

- Experiment ID:
- Date:
- Dataset version: dataset_v0.1
- Candidate class: A (traditional OCR) / B (PaddleOCR-VL) / C (Qwen-VL)
- Exact model/version:
- Package + version (e.g. `paddleocr==…`, `transformers==…`):
- Hardware/device:
- Results file: `data/benchmark/results/<file>.json`

## Scope

- Document IDs tested:
- Prompt/task type per document (ocr / table / chart / kpi):
- Prompt source: `src/ocrbench/prompts.py` (unedited? note any change):

## Measurements

- Latency per document (ms):
- Identifier accuracy (exact):
- Financial numeric accuracy (EXACT @2dp — primary):
- Financial numeric ±0.5% (diagnostic ONLY):
- Text anchor recall:
- CER / WER (vs anchors diagnostic):
- Table label recall:
- Table association accuracy:
- Chart/KPI recognition (A):
- Chart/KPI semantic (B):
- Schema violations:

## Observed failures

- Missed identifiers (expected vs got):
- Missed numerics (expected vs got — list EVERY miss):
- Layout/association errors:
- Language-specific issues (AR vs EN):

## Resource notes

- GPU/VRAM used:
- Install/runtime issues or blockers:
- Reproducibility notes (seeds, versions, flags):

## Decision

One of: **ACCEPT FOR NEXT ROUND** / **REJECT** / **NEEDS MORE DATA**

Justification (must cite measurements above, not impressions):

---

## Model-selection principle (frozen for Phase 2)

A model must not win for prettier prose. For scanned finance documents,
priority order is fixed:

1. exact identifiers
2. exact numeric values (2dp exact is the score; ±0.5% is diagnostic)
3. correct label/value association
4. Arabic/English text recovery
5. useful layout recovery
6. latency/resource requirements

For charts/KPI visuals, semantic understanding gains weight — but numeric
faithfulness still gates acceptance. The simplest model clearing the
gates wins; thresholds TBD after baselines (EVALUATION_PLAN §12).
