# MizanIQ Modal runners

## 1. Prerequisite: Modal setup

```bash
modal setup
```

Authenticate once and verify with e.g. `modal app list`.

## 2. Smoke test (CPU-only, no models)

From the repository root:

```bash
modal run infra/modal/smoke_test.py
```

Expected: a line from the remote worker plus the returned dict:

```text
[MizanIQ][Modal][remote] Hello from the Modal remote worker!
[MizanIQ][Modal][local] Remote call succeeded. Result:
{'status': 'ok', 'project': 'MizanIQ', 'message': 'Modal remote execution works', 'python_version': '3.x.x ...'}
```

`python_version` is the remote worker's interpreter.

## 3. PaddleOCR-VL-1.6 visual command (DEV-008 + DEV-009)

ONE command from the repository root (no Kaggle notebook needed):

```bash
modal run infra/modal/paddle_runner.py
```

Default behavior: runs DEV-008 then DEV-009 sequentially (one document
at a time), adapts each raw result locally via
`src/ocrbench/paddle_adapter.py`, and saves everything under
`data/benchmark/results/`.

Only the requested image bytes are sent to Modal. Ground truth stays
evaluation-side and is never uploaded to the inference worker.

## 4. Expected output

Concise per-document summary lines, e.g.:

```text
DEV-008 | OK | 12345 ms
DEV-009 | OK | 9876 ms
```

A document exceeding the hard timeout is reported, not left hanging:

```text
DEV-008 | TIMEOUT | >300000 ms
```

## 5. Output files

Created automatically (directory is made if missing):

- `data/benchmark/results/modal_paddle_visual_results.json` — one
  schema-compatible result per document (`model`, `model_version`,
  `device`, `document_id`, `latency_ms`, `text`, `fields`, `tables`,
  `visual_description`, `warnings`, plus `status`, `gpu_type`,
  `paddle_version`, `paddleocr_version`, `pipeline_version`,
  `chart_recognition`, `init_latency_ms`).
- `data/benchmark/results/modal_paddle_visual_meta.json` — run
  metadata (app, GPU, timeout, pins, volume, documents, timestamps).
- `data/benchmark/results/modal_paddle_raw_DEV-008.json`
- `data/benchmark/results/modal_paddle_raw_DEV-009.json` — preserved
  raw Paddle payloads (deterministic names, no hand correction).

## 6. A10G resource choice

- Exactly one `A10G` GPU per inference call (`gpu="A10G"`).
- No multi-GPU runs, no A100/H100 in this task.
- No always-on containers: Modal scales to zero after completion.

## 7. 300-second timeout

- The remote function uses Modal's real timeout: `timeout=300`.
- `retries=0`: a slow/hung document is NOT silently retried forever.
- On timeout the local entrypoint records a schema-valid `TIMEOUT`
  result (`status="TIMEOUT"`, `latency_ms=300000`) and continues/prints
  `DEV-008 | TIMEOUT | >300000 ms` instead of blocking indefinitely
  (the Kaggle T4 stall of 20+ minutes must not recur).

## 8. Model caching

- Image installs `paddlepaddle-gpu==3.2.1` (Paddle cu126 index) plus
  `paddleocr[doc-parser]==3.7.0`, matching the successful Kaggle stack.
  No Qwen/Transformers and no extra Torch CUDA stack (isolated to avoid
  the Paddle-vs-Torch CUDA/NCCL conflicts).
- Weights persist on Modal Volume `mizaniq-paddle-cache` mounted at
  `/cache`, with `PADDLE_PDX_CACHE_HOME=/cache/paddlex`, so
  PaddleOCR-VL weights are reused across calls instead of redownloaded
  every run. No credentials are stored in code.

## 9. Troubleshooting

- `modal setup` first; `modal app list` should succeed.
- Missing input file: the runner reads
  `data/benchmark/dataset_v0.1/inputs/DEV-008.png` /
  `inputs/DEV-009.jpg` first, falling back to
  `data/dev/dataset_v0.1/documents/...`. Rebuild the workspace if absent.
- `TIMEOUT`: expected for a >300s document; check the saved TIMEOUT
  result and Modal dashboard logs, then retry or investigate the image.
- `FAILED`: see the saved result's `warnings` and the raw file for the
  remote error string.
- GPU queue delays: Modal may queue if no A10G is free; the 300s timeout
  still bounds each call.
- Never edit frozen truth/DEV fixtures to fix an inference miss; record
  it in results/meta instead.

## 10. Qwen

Qwen is intentionally NOT implemented yet. This runner covers
PaddleOCR-VL-1.6 only (`PaddleOCRVL(pipeline_version="v1.6",
use_chart_recognition=True)`).
