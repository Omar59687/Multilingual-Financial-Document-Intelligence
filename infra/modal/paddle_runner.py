"""Modal runner for PaddleOCR-VL-1.6 visual benchmark (DEV-008/DEV-009).

Replaces manual Kaggle notebook interaction with ONE local command::

    modal run infra/modal/paddle_runner.py

Design notes (locked for this task):

- Modal app ``mizaniq-paddle-visual``, single A10G GPU, one document at a
  time, scale-to-zero (no always-on containers, no A100/H100).
- Real Modal timeout ``timeout=300`` on the remote function (not a fake
  elapsed-time check). Timeouts surface as TIMEOUT results locally.
- ``retries=0``: never silently retry forever.
- Isolated Paddle image: ``paddlepaddle-gpu==3.2.1`` (cu126 index) +
  ``paddleocr[doc-parser]==3.7.0``. No Qwen/Transformers, no Torch CUDA
  stack mixed in (avoids the Kaggle CUDA/NCCL conflicts).
- Pipeline ``PaddleOCRVL(pipeline_version="v1.6",
  use_chart_recognition=True)`` — visual benchmark, chart parsing on.
- Model weights cached on a Modal Volume via ``PADDLE_PDX_CACHE_HOME`` so
  weights are reused across calls, never redownloaded every run.
- Input strategy: local entrypoint reads ONLY the requested workspace
  image bytes (default DEV-008/DEV-009) and sends ``(document_id,
  image_bytes, filename)`` to Modal. No canonical dataset upload, no
  ground-truth upload — truth stays evaluation-side.
- Remote returns raw ``page.json`` payloads + versions/latency; the local
  side normalizes via the real ``src/ocrbench/paddle_adapter.py`` v1.6
  path (``page.json["res"]["parsing_res_list"]``) into the existing
  result schema. No duplicated benchmark logic, no invented fields.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import modal

# --------------------------------------------------------------------------
# Locked configuration
# --------------------------------------------------------------------------

APP_NAME = "mizaniq-paddle-visual"
GPU_TYPE = "A10G"  # single A10G only; no A100/H100 in this task
TIMEOUT_S = 300  # hard Modal timeout per remote inference call
TIMEOUT_MS = TIMEOUT_S * 1000

MODEL_CANDIDATE = "paddleocr-vl"
MODEL_ID = "PaddleOCR-VL-1.6"
PIPELINE_VERSION = "v1.6"
CHART_RECOGNITION = True  # visual benchmark: explicitly enabled
DEVICE_LABEL = "modal-A10G"

# Pinned to the successful Kaggle experiment (PaddleOCR 3.7.0 /
# PaddlePaddle 3.2.1). Task allows >=3.2.1; we lock exact for repro.
PADDLE_PIN = "paddlepaddle-gpu==3.2.1"
PADDLE_INDEX_URL = "https://www.paddlepaddle.org.cn/packages/stable/cu126/"
PADDLEOCR_PIN = "paddleocr[doc-parser]==3.7.0"

VOLUME_NAME = "mizaniq-paddle-cache"
CACHE_MOUNT_PATH = "/cache"
PADDLE_CACHE_DIR = "/cache/paddlex"  # PADDLE_PDX_CACHE_HOME value

DEFAULT_DOCS = ["DEV-008", "DEV-009"]
SUPPORTED_DOCS = frozenset(DEFAULT_DOCS)

# Workspace candidates (relative to repo root, tried in order). Primary is
# the frozen benchmark workspace; fallback is the original DEV documents.
INPUT_CANDIDATES = {
    "DEV-008": [
        Path("data/benchmark/dataset_v0.1/inputs/DEV-008.png"),
        Path("data/dev/dataset_v0.1/documents/DEV-008_revenue_trend_2015_2024_EN.png"),
    ],
    "DEV-009": [
        Path("data/benchmark/dataset_v0.1/inputs/DEV-009.jpg"),
        Path("data/dev/dataset_v0.1/documents/DEV-009_kpi_dashboard_Q4-2023_MIX.jpg"),
    ],
}

RESULTS_FILENAME = "modal_paddle_visual_results.json"
META_FILENAME = "modal_paddle_visual_meta.json"
RAW_FILENAME_TEMPLATE = "modal_paddle_raw_{doc_id}.json"

# Keys allowed in the inference input sent to Modal. Anything resembling
# ground truth must never appear here (checked by tests).
INFERENCE_INPUT_KEYS = frozenset(
    {
        "document_id",
        "filename",
        "image_bytes",
        "size_bytes",
        "sha256",
        "workspace_file",
        "mime",
    }
)


def repo_root() -> Path:
    """Repository root (two levels above this file)."""
    return Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# Pure local helpers (no GPU, fully unit-testable; Modal boundary mocked)
# --------------------------------------------------------------------------

def select_documents(requested=None) -> list:
    """Resolve the document list. Defaults to DEV-008/DEV-009.

    Accepts None (defaults), a comma-separated string, or a list/tuple.
    Raises ValueError for anything outside SUPPORTED_DOCS.
    """
    if requested is None:
        return list(DEFAULT_DOCS)
    if isinstance(requested, str):
        items = [p.strip() for p in requested.split(",") if p.strip()]
        if not items:
            return list(DEFAULT_DOCS)
    else:
        items = list(requested)
        if not items:
            return list(DEFAULT_DOCS)
    for doc_id in items:
        if doc_id not in SUPPORTED_DOCS:
            raise ValueError(
                f"unsupported document {doc_id!r}; "
                f"supported: {sorted(SUPPORTED_DOCS)}"
            )
    return items


def resolve_input_path(document_id: str, root=None) -> Path:
    """Resolve the local workspace image for a supported document."""
    if document_id not in SUPPORTED_DOCS:
        raise ValueError(f"unsupported document {document_id!r}")
    base = Path(root) if root is not None else repo_root()
    for candidate in INPUT_CANDIDATES[document_id]:
        full = base / candidate
        if full.is_file():
            return full
    raise FileNotFoundError(
        f"no local input found for {document_id}; tried: "
        + ", ".join(str(c) for c in INPUT_CANDIDATES[document_id])
    )


def _mime_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    return "application/octet-stream"


def load_inference_input(document_id: str, root=None) -> dict:
    """Read ONLY the requested image bytes for remote inference.

    Never reads ground-truth files; returns just enough for the worker:
    document id, filename, bytes, size, sha, workspace relpath, mime.
    """
    if document_id not in SUPPORTED_DOCS:
        raise ValueError(f"unsupported document {document_id!r}")
    path = resolve_input_path(document_id, root)
    base = Path(root) if root is not None else repo_root()
    data = path.read_bytes()
    if not data:
        raise ValueError(f"input file is empty: {path}")
    try:
        rel = str(path.resolve().relative_to(Path(base).resolve()))
    except ValueError:
        rel = path.name
    return {
        "document_id": document_id,
        "filename": path.name,
        "image_bytes": data,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "workspace_file": rel,
        "mime": _mime_for(path),
    }


def _lazy_schema():
    sys.path.insert(0, str(repo_root() / "src"))
    from ocrbench import schema

    return schema


def _lazy_adapter():
    sys.path.insert(0, str(repo_root() / "src"))
    from ocrbench import paddle_adapter

    return paddle_adapter


def validate_schema(result: dict) -> list:
    """Return schema violations for a result (empty = valid)."""
    return _lazy_schema().validate_result(result)


def _extras(
    paddle_version="unknown",
    paddleocr_version="unknown",
    init_latency_ms=None,
    status="OK",
) -> dict:
    return {
        "status": status,
        "gpu_type": GPU_TYPE,
        "paddle_version": paddle_version or "unknown",
        "paddleocr_version": paddleocr_version or "unknown",
        "pipeline_version": PIPELINE_VERSION,
        "chart_recognition": CHART_RECOGNITION,
        "init_latency_ms": init_latency_ms,
    }


def build_ok_result(
    document_id: str,
    latency_ms: float,
    text: str = "",
    fields: dict | None = None,
    tables: list | None = None,
    visual_description: str = "",
    warnings: list | None = None,
    paddle_version: str = "unknown",
    paddleocr_version: str = "unknown",
    init_latency_ms=None,
    device: str = DEVICE_LABEL,
) -> dict:
    """Schema-valid OK result with hardware/model metadata."""
    schema = _lazy_schema()
    adapter = _lazy_adapter()
    assert adapter.CANDIDATE == MODEL_CANDIDATE
    assert adapter.MODEL_ID == MODEL_ID
    result = schema.blank_result(
        MODEL_CANDIDATE, MODEL_ID, device, document_id
    )
    result["latency_ms"] = float(latency_ms)
    result["text"] = text if isinstance(text, str) else str(text)
    result["fields"] = dict(fields) if isinstance(fields, dict) else {}
    result["tables"] = list(tables) if isinstance(tables, list) else []
    result["visual_description"] = (
        visual_description if isinstance(visual_description, str) else ""
    )
    result["warnings"] = list(warnings) if warnings else []
    result.update(
        _extras(paddle_version, paddleocr_version, init_latency_ms, "OK")
    )
    return result


def build_timeout_result(
    document_id: str,
    timeout_ms: float = float(TIMEOUT_MS),
    device: str = DEVICE_LABEL,
    note: str | None = None,
) -> dict:
    """Schema-valid TIMEOUT result (Modal timeout fired, no inference)."""
    schema = _lazy_schema()
    result = schema.blank_result(
        MODEL_CANDIDATE, MODEL_ID, device, document_id
    )
    result["latency_ms"] = float(timeout_ms)
    result["warnings"] = [
        f"TIMEOUT: {document_id} exceeded {TIMEOUT_S}s Modal timeout; "
        "reported as TIMEOUT/blocked, not retried."
    ]
    if note:
        result["warnings"].append(str(note))
    result.update(_extras("unknown", "unknown", None, "TIMEOUT"))
    return result


def build_failed_result(
    document_id: str,
    error: str,
    latency_ms: float = 0.0,
    device: str = DEVICE_LABEL,
) -> dict:
    """Schema-valid FAILED result (remote raised before completing)."""
    schema = _lazy_schema()
    result = schema.blank_result(
        MODEL_CANDIDATE, MODEL_ID, device, document_id
    )
    result["latency_ms"] = float(latency_ms)
    result["warnings"] = [f"FAILED: {error}"]
    result.update(_extras("unknown", "unknown", None, "FAILED"))
    return result


def adapt_remote_payload(payload: dict, raw_path=None) -> dict:
    """Convert a remote inference payload into the final schema result.

    Remote payloads carry ``raw_pages`` (list of ``page.json`` dicts);
    normalization reuses the real v1.6 adapter locally. FAILED payloads
    become FAILED results; anything unexpected becomes FAILED (never crash).
    """
    try:
        status = payload.get("status", "FAILED")
        document_id = payload.get("document_id", "unknown")
        latency_ms = float(payload.get("latency_ms", 0.0) or 0.0)
        init_latency = payload.get("init_latency_ms")
        paddle_version = payload.get("paddle_version", "unknown")
        paddleocr_version = payload.get("paddleocr_version", "unknown")
        device = payload.get("device", DEVICE_LABEL)
        if status == "FAILED":
            return build_failed_result(
                document_id,
                str(payload.get("error", "remote FAILED without detail")),
                latency_ms,
                device,
            )
        raw_pages = payload.get("raw_pages", [])
        adapter = _lazy_adapter()
        identity = adapter.identity_for(
            paddleocr_version, paddle_version, device
        )
        notes_extra = list(payload.get("warnings", []))
        notes_extra.append(
            f"modal gpu={GPU_TYPE} pipeline={PIPELINE_VERSION} "
            f"chart_recognition={CHART_RECOGNITION}"
        )
        result = adapter.build_result(
            document_id, raw_pages, latency_ms, identity,
            raw_path=raw_path, notes_extra=notes_extra,
        )
        result.update(
            _extras(
                paddle_version, paddleocr_version, init_latency, "OK"
            )
        )
        result["device"] = device
        return result
    except Exception as exc:  # never crash the entrypoint on adapt errors
        doc = payload.get("document_id", "unknown") \
            if isinstance(payload, dict) else "unknown"
        return build_failed_result(doc, f"adapt failed: {exc}")


# --------------------------------------------------------------------------
# Deterministic local output paths
# --------------------------------------------------------------------------

def results_path(root=None) -> Path:
    base = Path(root) if root is not None else repo_root()
    return base / "data" / "benchmark" / "results" / RESULTS_FILENAME


def meta_path(root=None) -> Path:
    base = Path(root) if root is not None else repo_root()
    return base / "data" / "benchmark" / "results" / META_FILENAME


def raw_filename_for(document_id: str) -> str:
    return RAW_FILENAME_TEMPLATE.format(doc_id=document_id)


def raw_path_for(document_id: str, root=None) -> Path:
    base = Path(root) if root is not None else repo_root()
    return (
        base / "data" / "benchmark" / "results"
        / raw_filename_for(document_id)
    )


def ensure_output_dir(root=None) -> Path:
    outdir = results_path(root).parent
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def save_local_outputs(results, meta, raw_by_doc, root=None) -> dict:
    """Write results/meta/raw JSON files; return the paths used."""
    outdir = ensure_output_dir(root)
    rpath = results_path(root)
    mpath = meta_path(root)
    rpath.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    mpath.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    written_raw = {}
    for doc_id, raw in (raw_by_doc or {}).items():
        rp = raw_path_for(doc_id, root)
        rp.write_text(
            json.dumps(raw, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        written_raw[doc_id] = str(rp)
    return {
        "dir": str(outdir),
        "results": str(rpath),
        "meta": str(mpath),
        "raw": written_raw,
    }


def format_summary_line(document_id: str, status: str, latency_ms) -> str:
    """Concise terminal line, e.g. ``DEV-008 | OK | 12345 ms``."""
    try:
        ms = int(float(latency_ms))
    except (TypeError, ValueError):
        ms = TIMEOUT_MS
    if status == "TIMEOUT":
        return f"{document_id} | TIMEOUT | >{ms} ms"
    return f"{document_id} | {status} | {ms} ms"


def classify_remote_error(exc: BaseException) -> str:
    """Map a .remote() exception to TIMEOUT vs FAILED without GPU."""
    name = type(exc).__name__
    text = f"{name}: {exc}".lower()
    if name in ("FunctionTimeoutError", "TimeoutError") or "timeout" in text:
        return "TIMEOUT"
    # Modal may wrap timeouts as ExecTimeoutError / generic timeouts.
    if "timed out" in text or "deadline" in text:
        return "TIMEOUT"
    return "FAILED"


# --------------------------------------------------------------------------
# Modal app / image / volume / remote function
# --------------------------------------------------------------------------

app = modal.App(name=APP_NAME)

paddle_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(PADDLE_PIN, index_url=PADDLE_INDEX_URL)
    .pip_install(PADDLEOCR_PIN)
    .env({"PADDLE_PDX_CACHE_HOME": PADDLE_CACHE_DIR})
)

paddle_cache_volume = modal.Volume.from_name(
    VOLUME_NAME, create_if_missing=True
)


@app.function(
    image=paddle_image,
    gpu=GPU_TYPE,  # exactly one A10G
    timeout=TIMEOUT_S,  # REAL Modal timeout: kills the call at 300s
    retries=0,  # never silently retry forever
    volumes={CACHE_MOUNT_PATH: paddle_cache_volume},
)
def run_paddle_visual(
    document_id: str, image_bytes: bytes, filename: str
) -> dict:
    """GPU inference worker: PaddleOCR-VL-1.6 on one document's bytes.

    Runs ONLY on Modal. Returns a JSON-safe payload with raw
    ``page.json`` dicts + versions/latency; the caller adapts it locally
    via ``src/ocrbench/paddle_adapter.py``. Failures return a FAILED
    payload (never raise, except Modal timeout which aborts remotely).
    """
    import json as _json
    import os as _os
    import tempfile as _tempfile
    import time as _time

    _os.environ["PADDLE_PDX_CACHE_HOME"] = PADDLE_CACHE_DIR

    started_init = _time.perf_counter()
    try:
        from paddleocr import PaddleOCRVL

        pipeline = PaddleOCRVL(
            pipeline_version="v1.6",
            use_chart_recognition=True,
        )
        init_latency_ms = (_time.perf_counter() - started_init) * 1000.0
    except Exception as exc:
        return {
            "status": "FAILED",
            "document_id": document_id,
            "latency_ms": 0.0,
            "init_latency_ms": None,
            "raw_pages": [],
            "paddle_version": "unknown",
            "paddleocr_version": "unknown",
            "pipeline_version": PIPELINE_VERSION,
            "chart_recognition": CHART_RECOGNITION,
            "gpu_type": GPU_TYPE,
            "device": DEVICE_LABEL,
            "error": f"pipeline init failed: {type(exc).__name__}: {exc}",
            "warnings": [f"init failed: {exc}"],
        }

    try:
        from importlib import metadata as _metadata

        try:
            _paddle_v = _metadata.version("paddlepaddle-gpu")
        except Exception:
            _paddle_v = "unknown"
        try:
            _paddleocr_v = _metadata.version("paddleocr")
        except Exception:
            _paddleocr_v = "unknown"
    except Exception:
        _paddle_v, _paddleocr_v = "unknown", "unknown"

    suffix = Path(filename).suffix.lower() or ".png"
    tmp = _tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(image_bytes)
        tmp.close()
        print(
            f"[MizanIQ][Modal][remote] {document_id}: "
            f"PaddleOCR-VL-1.6 inference starting "
            f"({len(image_bytes)} bytes, chart_recognition=True)"
        )
        started = _time.perf_counter()
        output = pipeline.predict(tmp.name)
        pages = list(output) if output is not None else []
        latency_ms = (_time.perf_counter() - started) * 1000.0

        raw_pages, raw_markdowns, notes = [], [], []
        for idx, page in enumerate(pages):
            try:
                page_json = page.json
                if callable(page_json):
                    try:
                        page_json = page_json()
                    except TypeError:
                        pass
                if isinstance(page_json, dict):
                    # Force JSON-safety for the Modal return trip.
                    page_json = _json.loads(
                        _json.dumps(page_json, ensure_ascii=False, default=str)
                    )
                    raw_pages.append(page_json)
                else:
                    notes.append(f"page {idx}: non-dict page.json skipped")
            except Exception as exc:
                notes.append(f"page {idx}: page.json unreadable: {exc}")
            try:
                md = getattr(page, "markdown", None)
                if callable(md):
                    try:
                        md = md()
                    except TypeError:
                        pass
                if isinstance(md, (dict, str)):
                    raw_markdowns.append(
                        _json.loads(
                            _json.dumps(md, ensure_ascii=False, default=str)
                        )
                    )
            except Exception:
                pass
        print(
            f"[MizanIQ][Modal][remote] {document_id}: done "
            f"pages={len(raw_pages)} latency_ms={latency_ms:.1f}"
        )
        return {
            "status": "OK",
            "document_id": document_id,
            "latency_ms": latency_ms,
            "init_latency_ms": init_latency_ms,
            "raw_pages": raw_pages,
            "raw_markdowns": raw_markdowns,
            "paddle_version": _paddle_v,
            "paddleocr_version": _paddleocr_v,
            "pipeline_version": PIPELINE_VERSION,
            "chart_recognition": CHART_RECOGNITION,
            "gpu_type": GPU_TYPE,
            "device": DEVICE_LABEL,
            "warnings": notes,
        }
    except Exception as exc:
        return {
            "status": "FAILED",
            "document_id": document_id,
            "latency_ms": 0.0,
            "init_latency_ms": init_latency_ms,
            "raw_pages": [],
            "paddle_version": _paddle_v,
            "paddleocr_version": _paddleocr_v,
            "pipeline_version": PIPELINE_VERSION,
            "chart_recognition": CHART_RECOGNITION,
            "gpu_type": GPU_TYPE,
            "device": DEVICE_LABEL,
            "error": f"inference failed: {type(exc).__name__}: {exc}",
            "warnings": [f"inference failed: {exc}"],
        }
    finally:
        try:
            Path(tmp.name).unlink(missing_ok=True)
        except Exception:
            pass


# --------------------------------------------------------------------------
# Local entrypoint: ONE command runs DEV-008 + DEV-009 sequentially
# --------------------------------------------------------------------------

@app.local_entrypoint()
def main() -> None:
    """Run DEV-008 + DEV-009 on Modal sequentially, save locally."""
    docs = select_documents(None)
    results, raw_by_doc = [], {}
    summary = []
    for doc_id in docs:  # one document at a time: cost/resource safety
        payload_input = load_inference_input(doc_id)
        print(
            f"[MizanIQ][Modal][local] {doc_id}: sending "
            f"{payload_input['size_bytes']} bytes "
            f"({payload_input['filename']}) to {APP_NAME} on {GPU_TYPE}..."
        )
        try:
            payload = run_paddle_visual.remote(
                payload_input["document_id"],
                payload_input["image_bytes"],
                payload_input["filename"],
            )
        except Exception as exc:
            kind = classify_remote_error(exc)
            if kind == "TIMEOUT":
                result = build_timeout_result(doc_id, note=str(exc))
                print(f"[MizanIQ][Modal][local] {doc_id}: TIMEOUT ({exc})")
            else:
                result = build_failed_result(doc_id, f"{exc}")
                print(f"[MizanIQ][Modal][local] {doc_id}: FAILED ({exc})")
            results.append(result)
            raw_by_doc[doc_id] = {
                "status": result["status"],
                "document_id": doc_id,
                "error": str(exc),
            }
            summary.append(
                format_summary_line(
                    doc_id, result["status"], result["latency_ms"]
                )
            )
            continue
        raw_path_hint = str(raw_path_for(doc_id))
        result = adapt_remote_payload(payload, raw_path=raw_path_hint)
        results.append(result)
        raw_by_doc[doc_id] = payload
        summary.append(
            format_summary_line(
                doc_id, result.get("status", "?"),
                result.get("latency_ms", 0),
            )
        )
        print(
            f"[MizanIQ][Modal][local] {doc_id}: "
            f"{result.get('status')} latency_ms="
            f"{result.get('latency_ms'):.1f}"
        )

    meta = {
        "app": APP_NAME,
        "gpu_type": GPU_TYPE,
        "gpu_count": 1,
        "timeout_s": TIMEOUT_S,
        "model": MODEL_CANDIDATE,
        "model_version": MODEL_ID,
        "pipeline_version": PIPELINE_VERSION,
        "chart_recognition": CHART_RECOGNITION,
        "paddle_pin": PADDLE_PIN,
        "paddleocr_pin": PADDLEOCR_PIN,
        "volume": VOLUME_NAME,
        "cache_dir": PADDLE_CACHE_DIR,
        "documents": docs,
        "device": DEVICE_LABEL,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "qwen_implemented": False,
    }
    paths = save_local_outputs(results, meta, raw_by_doc)
    print(f"[MizanIQ][Modal][local] wrote {paths['results']}")
    print(f"[MizanIQ][Modal][local] wrote {paths['meta']}")
    for doc_id in docs:
        print(
            f"[MizanIQ][Modal][local] wrote "
            f"{raw_path_for(doc_id)}"
        )
    for line in summary:
        print(line)
