# Kaggle automation (current free execution path)

Runs the Phase-2 Qwen visual benchmark (DEV-008/DEV-009) on Kaggle's free
GPU without manual notebook-UI operation. One explicit command = one
Kaggle submission. Modal code is retained for future paid GPU access;
Kaggle is the current path because Modal A10G/T4 allocation requires a
payment method on this account.

## Setup

1. Install the Kaggle CLI: `pip install kaggle`.
2. Authenticate (supported mechanisms only, never hardcoded):
   - `KAGGLE_USERNAME` + `KAGGLE_KEY` environment variables, or
   - `~/.kaggle/kaggle.json` containing `{"username": "...", "key": "..."}`
     (permissions `600` on Linux/macOS).
3. Attach access to the benchmark dataset
   `omarabdallah12/mizaniq-ocr-benchmark-v0-1` (override with `--dataset`
   if your copy lives under a different owner).

`kaggle.json` and generated staging (`infra/kaggle/build/`) are
git-ignored and never committed.

## Dry run (validates, submits nothing)

```bash
python scripts/kaggle_submit_visual.py --dry-run
```

Checks the source notebook, generates the execution notebook + metadata
into `infra/kaggle/build/run-<stamp>/`, prints the plan (model, docs,
dataset, kernel slug, switches, auth status), and exits 0 without
contacting Kaggle.

## Submitting Qwen 4B (explicit selection)

```bash
python scripts/kaggle_submit_visual.py --model qwen3-vl-4b --docs DEV-008 DEV-009 --score
```

## Submitting Qwen 8B (primary target)

```bash
python scripts/kaggle_submit_visual.py --model qwen3-vl-8b --docs DEV-008 DEV-009 --score
```

8B is never silently remapped to 4B: the exact requested id
(`Qwen/Qwen3-VL-8B-Instruct` vs `Qwen/Qwen3-VL-4B-Instruct`) is baked
into the generated notebook with `ALLOW_QWEN_FALLBACK = False`. An 8B
failure is recorded as a blocker result; no automatic 4B rerun happens.

## Attached dataset

`kernel-metadata.json` attaches the dataset by slug
(`dataset_sources`), and the generated notebook resolves the workspace
robustly by searching for `manifest.json` under `/kaggle/input` (no
fragile hardcoded nested paths). Ground truth stays evaluation-side;
only image bytes are inferred on.

## GPU behavior

- `enable_gpu=true`, `enable_internet=true` (first Qwen download from
  Hugging Face needs internet), accelerator `NvidiaTeslaT4`.
- No specific GPU model is guaranteed; at runtime the notebook records
  detected GPU name, VRAM, CUDA availability, torch/transformers
  versions, and the selected Qwen model in `experiment_meta.json`.
- A run is never called successful merely because a GPU exists.
- Gated models needing a Hugging Face token are reported as blockers;
  tokens are never hardcoded.

## Output directories

Downloads land in a collision-safe run directory, never overwriting
prior runs:

```text
data/benchmark/results/kaggle_qwen_<model>_<timestamp>/
  qwen_visual_results.json   (from kaggle_results.json)
  qwen_visual_meta.json      (from experiment_meta.json)
  qwen_raw_DEV-008.json
  qwen_raw_DEV-009.json
  kaggle_run_log.json        (slug, status, model, dataset, elapsed, files)
```

## Scoring

With `--score`, the runner invokes the existing
`scripts/score_ocr_results.py` on the downloaded results and prints its
output. Scoring logic is not duplicated. Scoring failure preserves the
downloads and is reported separately.

## Polling / timeouts

- `--poll-interval 20 --max-wait-minutes 45` (defaults). The loop
  sleeps between checks (no busy-loop).
- Terminal states: `COMPLETE` → download; `ERROR` → report + kernel URL;
  local window expiry → `RUNNER_TIMEOUT` (the Kaggle job itself is NOT
  claimed killed; inspect it at the printed kernel URL).
- No automatic retries: quota is precious.

## Troubleshooting

- Missing auth → clear error naming both supported mechanisms.
- `RUNNER_TIMEOUT` → check the kernel URL; outputs may appear later and
  can be fetched manually.
- `ERROR` status → kernel URL + `kaggle_run_log.json` narrow it down.
- Gated-model auth errors → blocker, not a code bug.
- Schema-invalid downloads → reported by the contract check; files are
  kept for inspection.

## How this differs from Modal

- Modal (`infra/modal/`): serverless GPUs, kept for future paid use;
  currently billing-blocked for A10G and T4 on this account.
- Kaggle (this directory): free quota, slower/queued, notebook-kernel
  model — automated here so no manual UI clicking is needed.
