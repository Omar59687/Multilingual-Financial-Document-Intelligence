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

Report recognition (Layer A) and structuring (Layer B) SEPARATELY —
never merge or average them. A correct number in prose earns text
credit but ZERO structured credit without the right field/row.

Layer A — recognition ("did the model see it?"):

- Identifier TEXT accuracy (identifier_text_accuracy):
- Financial numeric TEXT exact accuracy (numeric_text_exact_accuracy):
- Text anchor recall:
- Table LABEL-TEXT recall (table_label_text_recall):
- CER / WER (vs anchors — diagnostic ONLY, not full-doc accuracy):

Layer B — structuring ("did the model attach it correctly?"):

- Identifier STRUCTURED accuracy (identifier_structured_accuracy):
- Financial numeric STRUCTURED exact accuracy
  (numeric_structured_exact_accuracy — primary field score):
- Financial numeric ±0.5% (diagnostic ONLY, never the primary score):
- Table association accuracy (table_association_accuracy):
- Chart/KPI recognition (A):
- Chart/KPI semantic (B):
- Schema violations:

Fair-comparison rule: traditional OCR is judged FIRST on Layer A;
layout/document-understanding models on BOTH layers; VLMs may add
semantic interpretation. No structured credit for prose-only mentions.

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

---

## Appendix A — First real Phase 2 benchmark (measured reference)

Reference values only — do not edit. Recorded here so future runs compare
against the same floor. Recognition (Layer A) and structuring (Layer B)
stay separate; never merge them into one score. Qwen was not run.

Environment: PaddlePaddle 3.2.1, PaddleOCR 3.7.0, CUDA, Tesla T4;
pipeline `PaddleOCRVL(pipeline_version="v1.6")`. Canonical adapter source:
`page.json["res"]["parsing_res_list"]`; table HTML → row grids;
`fields = {}` for v1.6 (no invented semantic fields).

Tesseract (kaggle-CPU, 4.1.1, eng fallback):

- DEV-004: identifier text accuracy = 1.0, numeric text exact = 1.0,
  anchor recall = 0.4, structured fields = 0, latency ~603 ms.
- DEV-010: numeric text exact = 1.0, anchor recall = 1.0, table label
  text recall = 1.0, table association = 0, latency ~856 ms.

PaddleOCR-VL-1.6 (kaggle-GPU T4):

- DEV-004: identifier text accuracy = 1.0, numeric text exact = 1.0,
  anchor recall = 1.0, structured semantic fields = 0,
  latency = 8621.42 ms.
- DEV-010: numeric text exact = 1.0, anchor recall = 1.0, table label
  text recall = 1.0, table association accuracy = 1.0, structured
  semantic fields = 0, latency = 11274.39 ms.

Interpretation: Tesseract remains the fast raw-OCR baseline; Paddle
materially improves Arabic coverage and structured table recovery at
roughly an order of magnitude higher latency; semantic canonical field
extraction is still a separate downstream concern.
