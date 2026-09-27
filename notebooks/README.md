# Kaggle GPU Lab — MizanIQ Phase-2 OCR/Vision Experiments

Temporary free-GPU laboratory for early candidate comparison. The notebook
here is **independent from the production pipeline**: it reads benchmark
PNGs, runs ONE candidate model, and exports schema-conformant JSON that the
local scorer (`src/ocrbench/`) grades. No weights, secrets, or production
coupling ever enter this repository.

## What to upload to Kaggle

1. This notebook (`notebooks/ocr_vision_benchmark.ipynb`), **or** recreate
   its cells in a new Kaggle notebook.
2. The benchmark inputs as a Kaggle Dataset (or notebook file upload):
   `data/benchmark/dataset_v0.1/inputs/` (6 PNG/JPG files, < 5 MB total)
   plus `manifest.json` and `ocr_benchmark_truth.json` for self-scoring.
3. Nothing else. Never upload `data/dev/` originals, canonical CSVs, or
   any credentials.

## Running the first experiment (exact steps)

1. Create/open a Kaggle notebook and upload the export-bundle files
   (`inputs/`, `manifest.json`, `ocr_benchmark_truth.json`,
   `ocr_vision_benchmark.ipynb`) — easiest: upload the bundle ZIP from
   `data/benchmark/export/` as a Kaggle Dataset and attach it.
2. Enable Internet (≡ Settings → Internet → ON) only because the
   PaddleOCR-VL first run downloads official model files; Tesseract
   alone needs no Internet beyond package installs.
3. Select **CPU** accelerator for the Tesseract baseline run.
4. Run notebook sections 0–3 (environment → load → Tesseract) and then
   section 6 (export). Tesseract results export independently.
5. For the Paddle experiment: switch to (or create) a **GPU T4** session
   with the same files attached.
6. Run the environment cell (section 0) and confirm the GPU report.
7. Run the Setup cell: it reports the CUDA environment first, then installs
   in SEPARATE steps — first the PaddlePaddle GPU build (>=3.2.1,
   CUDA-matched per the official install table linked in the cell —
   never guess a wheel), then `paddleocr[doc-parser]` (the doc-parser
   extra is required for the v1.6 `PaddleOCRVL` pipeline). Keep package
   installation and `import` cells separate: install → (kernel restart
   may be required after installation; if `import paddleocr` fails or
   reports CPU-only right after install, restart the kernel and re-run
   the import cell) → import and verify versions. If `import paddle`
   is CPU-only on the GPU session, install the CUDA-matched
   `paddlepaddle-gpu` wheel per the official install table (linked in
   the cell) — never guess a wheel.
8. Run section 4 (Experiment B): DEV-004 and DEV-010 run by default;
   leave `RUN_PADDLE_OPTIONAL = False` for the first pass (DEV-008/009
   come later). Model init/download time is recorded separately from
   per-document inference latency by the notebook itself.
9. Run section 6 to write `kaggle_results.json` + `experiment_meta.json`.
10. Download both files to local `data/benchmark/results/` (git-ignored),
    e.g. `data/benchmark/results/paddle-vl-1.6-<date>.json`.
11. Score locally with `python scripts/score_ocr_results.py
    data/benchmark/results/<file>.json` and file one report per run
    using `docs/OCR_EXPERIMENT_TEMPLATE.md`.

## Rules

- One variable per experiment (EVALUATION_PLAN §13): same inputs, same
  truth, only the model changes.
- Record `device` honestly (`kaggle-T4`, `kaggle-CPU`, …) and per-document
  `latency_ms` — latency is a selection criterion, not an afterthought.
- Results with a failed schema (`src/ocrbench/schema.py`) are not scored;
  validate locally with `python -c` before reporting numbers.
- Do NOT commit model binaries, datasets, or `kaggle.json` credentials.

## Suggested order

1. Class A (Tesseract, CPU) on DEV-004 + DEV-010 → fixes the floor.
2. Class B (PaddleOCR-VL) on the same inputs → AR/EN + layout delta.
3. Class C (Qwen-VL) only where B demonstrably fails (DEV-008 trend,
   DEV-009 layout/verdict). Class D (ColQwen) is a Phase-6 retrieval
   concern, not an OCR experiment.

## First real benchmark (measured — reference, do not edit)

Kaggle lab, `PaddleOCRVL(pipeline_version="v1.6")`
(PaddlePaddle 3.2.1, PaddleOCR 3.7.0, Tesla T4). Canonical adapter source:
`page.json["res"]["parsing_res_list"]`; table HTML → row grids;
`fields = {}` (no invented semantic fields). Qwen not run.

- Tesseract DEV-004: identifier text 1.0, numeric text exact 1.0, anchor
  recall 0.4, structured fields 0, latency ~603 ms.
- Tesseract DEV-010: numeric text exact 1.0, anchor recall 1.0, table
  label text recall 1.0, table association 0, latency ~856 ms.
- PaddleOCR-VL-1.6 DEV-004: identifier text 1.0, numeric text exact 1.0,
  anchor recall 1.0, structured semantic fields 0, latency 8621.42 ms.
- PaddleOCR-VL-1.6 DEV-010: numeric text exact 1.0, anchor recall 1.0,
  table label text recall 1.0, table association 1.0, structured semantic
  fields 0, latency 11274.39 ms.

Interpretation: Tesseract stays the fast raw-OCR baseline; Paddle
materially improves Arabic coverage and structured table recovery at
roughly an order of magnitude higher latency; semantic field extraction
is a separate downstream concern. Full detail:
`docs/OCR_VISION_DESIGN.md` §10–§11.

## Provisional routing (evidence-based candidate, not hardcoded)

1. Native parser first; stay native when sufficient.
2. Scanned/simple text: lightweight OCR may suffice.
3. Arabic-heavy, layout-heavy, table-heavy, or complex scanned:
   PaddleOCR-VL is the preferred current candidate.
4. Charts/KPI visual reasoning remain a separate pending experiment.
