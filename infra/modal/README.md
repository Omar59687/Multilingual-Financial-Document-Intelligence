# MizanIQ Modal smoke test

Minimal check proving this repository can execute code on a Modal remote worker.

Scope is intentionally tiny:

- CPU-only, no GPU requested.
- No PaddleOCR / Torch / Transformers / model dependencies.
- No secrets.
- Isolated under `infra/modal/`; does not touch dataset truth, benchmark files, OCR code, or Phase 2 metrics.

## Prerequisite

```bash
modal setup
```

Authenticate once and verify with e.g. `modal app list`.

## Smoke-test command

From the repository root:

```bash
modal run infra/modal/smoke_test.py
```

## Expected output

A line printed from the remote worker, followed by the returned dict printed locally, e.g.:

```text
[MizanIQ][Modal][remote] Hello from the Modal remote worker!
[MizanIQ][Modal][local] Remote call succeeded. Result:
{'status': 'ok', 'project': 'MizanIQ', 'message': 'Modal remote execution works', 'python_version': '3.x.x ...'}
```

`python_version` reflects the remote worker's interpreter, not your local one.

## Next steps

GPU runners (PaddleOCR / Qwen workloads) will be added only after this test passes.
