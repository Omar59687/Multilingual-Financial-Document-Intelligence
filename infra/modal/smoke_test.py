"""Minimal Modal smoke test for MizanIQ.

Proves this repository can execute code on a Modal remote worker.

- CPU-only: no GPU requested.
- No heavy dependencies (no Paddle, Torch, Transformers).
- No secrets.
- Isolated under infra/modal/; does not touch dataset truth,
  benchmark files, OCR code, or Phase 2 metrics.

Run:
    modal run infra/modal/smoke_test.py
"""

import modal

app = modal.App(name="mizaniq-smoke-test")


@app.function()
def smoke() -> dict:
    """Run on Modal (not locally) and return a small structured result."""
    import sys

    print("[MizanIQ][Modal][remote] Hello from the Modal remote worker!")

    return {
        "status": "ok",
        "project": "MizanIQ",
        "message": "Modal remote execution works",
        "python_version": sys.version,
    }


@app.local_entrypoint()
def main() -> None:
    """Local entrypoint: call the remote function and print its result."""
    result = smoke.remote()

    print("[MizanIQ][Modal][local] Remote call succeeded. Result:")
    print(result)
