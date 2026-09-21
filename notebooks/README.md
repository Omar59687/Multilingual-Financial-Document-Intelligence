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
7. Run the Setup cell: it reports the CUDA environment first, then runs
   `%pip install -q paddleocr`. If `import paddle` is CPU-only on the GPU
   session, install the CUDA-matched `paddlepaddle-gpu` wheel per the
   official install table (linked in the cell) — never guess a wheel.
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
