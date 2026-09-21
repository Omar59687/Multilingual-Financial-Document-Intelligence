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

## Running one experiment

1. Enable GPU (⌘ Settings → Accelerator → GPU T4) only for model classes
   B/C/D. Class A (Tesseract baseline) runs on CPU.
2. Install exactly one candidate in the notebook session, e.g.
   `%pip install -q paddlepaddle-gpu paddleocr` (class B) or load a
   Qwen-VL checkpoint from Hugging Face (class C). Pin versions in the
   notebook and record them in `model_version`.
3. Fill in `run_candidate(image_path)` — return `(text, fields, tables,
   visual_description)`. Keep `fields` keys aligned with the derived truth
   (`ocr_benchmark_truth.json`) so the local scorer matches them directly.
4. Run all cells. Download `kaggle_results.json` and place it under local
   `data/benchmark/results/<model>-<date>.json` (git-ignored) for scoring.

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
