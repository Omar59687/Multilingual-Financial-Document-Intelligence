"""Kaggle automation runner (local side, stdlib only).

Builds a Kaggle GPU execution path for the Phase-2 visual benchmark
(DEV-008/DEV-009, Qwen3-VL) without manual notebook-UI operation:

- explicit model selection (4B or 8B, never silent fallback),
- generated execution notebook (strategy A: source notebook + programmatic
  switch flips; the committed source keeps expensive execution OFF),
- valid ``kernel-metadata.json`` (GPU on, internet on, benchmark dataset
  attached by slug),
- bounded status polling (COMPLETE / ERROR / RUNNER_TIMEOUT),
- deterministic download layout under ``data/benchmark/results/``,
- optional scoring via the existing ``scripts/score_ocr_results.py``.

No Kaggle import here (the ``kaggle`` CLI/API is invoked by the caller via
subprocess and is mockable in tests). No credentials are read, printed,
or stored by this module — see :func:`check_auth`.
"""

import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

CANDIDATE = "qwen3-vl"

# Frozen candidate policy: 8B primary, 4B fallback-only. Automation selects
# EXPLICITLY via CLI alias; 8B is never silently remapped to 4B.
MODEL_ALIASES = {
    "qwen3-vl-4b": "Qwen/Qwen3-VL-4B-Instruct",
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
}
MODEL_IDS = frozenset(MODEL_ALIASES.values())
DEFAULT_MODEL_ALIAS = "qwen3-vl-4b"

DEFAULT_DOCS = ["DEV-008", "DEV-009"]
SUPPORTED_DOCS = frozenset(DEFAULT_DOCS)

# Benchmark dataset on Kaggle (owner may differ per account; overridable).
DEFAULT_DATASET_SLUG = "omarabdallah12/mizaniq-ocr-benchmark-v0-1"

SOURCE_NOTEBOOK = Path("notebooks/ocr_vision_benchmark.ipynb")
RESULTS_ROOT = Path("data/benchmark/results")
SCORER_SCRIPT = Path("scripts/score_ocr_results.py")

# Kernel outputs -> deterministic local names (existing notebook writes
# kaggle_results.json / experiment_meta.json; raw files pass through).
DOWNLOAD_RENAMES = {
    "kaggle_results.json": "qwen_visual_results.json",
    "experiment_meta.json": "qwen_visual_meta.json",
}
RAW_PREFIXES = ("qwen_raw_", "paddle_raw_")

AUTOMATION_CELL_MARKER = "MIZANIQ-KAGGLE-AUTOMATION"

TERMINAL_OK = ("complete", "completed", "success")
TERMINAL_BAD = ("error", "failed", "failure", "cancelled", "canceled")


# --------------------------------------------------------------------------
# Model / document selection
# --------------------------------------------------------------------------

def resolve_model_id(model) -> str:
    """Map a CLI alias (or exact HF id) to the exact model id.

    Raises ValueError for anything unknown. Never remaps 8B -> 4B: the
    returned id is exactly what was requested.
    """
    if model is None:
        return MODEL_ALIASES[DEFAULT_MODEL_ALIAS]
    key = str(model).strip()
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    if key in MODEL_IDS:
        return key
    raise ValueError(
        f"unsupported model {model!r}; use "
        f"{sorted(MODEL_ALIASES)} or an exact id from {sorted(MODEL_IDS)}"
    )


def select_docs(docs=None) -> list:
    """Default to DEV-008 + DEV-009; reject anything else."""
    if docs is None:
        return list(DEFAULT_DOCS)
    if isinstance(docs, str):
        items = [p.strip() for p in docs.split(",") if p.strip()]
        return list(DEFAULT_DOCS) if not items else _check_docs(items)
    items = list(docs)
    return list(DEFAULT_DOCS) if not items else _check_docs(items)


def _check_docs(items) -> list:
    for doc_id in items:
        if doc_id not in SUPPORTED_DOCS:
            raise ValueError(
                f"unsupported document {doc_id!r}; "
                f"supported: {sorted(SUPPORTED_DOCS)}"
            )
    return items


# --------------------------------------------------------------------------
# Authentication (supported mechanisms only, never hardcoded)
# --------------------------------------------------------------------------

# Safe, non-destructive CLI probe: succeeds only when authenticated and
# reveals the account via dataset `ref` ("owner/slug") without touching
# any token. NEVER call `kaggle auth print-access-token` here.
_OAUTH_PROBE_CMD = ("kaggle", "datasets", "list", "--mine",
                    "--page-size", "1", "--format", "json")


def _run_cli_json(cmd, runner=None, timeout_s: float = 60.0):
    """Run a CLI command expecting JSON stdout.

    Returns ``(ok, payload)``. ``runner`` is injectable for tests
    (same ``(cmd, **kwargs) -> CompletedProcess`` protocol as
    :func:`subprocess.run`). Only ``returncode``/``stdout`` are read —
    never any token content.
    """
    run = runner or subprocess.run
    try:
        proc = run(list(cmd), capture_output=True, text=True,
                   timeout=timeout_s)
    except (OSError, subprocess.SubprocessError):
        return False, None
    if getattr(proc, "returncode", 1) != 0:
        return False, None
    try:
        return True, json.loads(getattr(proc, "stdout", "") or "null")
    except ValueError:
        return False, None


def _owner_from_dataset_refs(payload):
    """Extract the account owner from a dataset-list payload's refs."""
    items = []
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        for key in ("datasets", "data", "items", "results"):
            if isinstance(payload.get(key), list):
                items = payload[key]
                break
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("ref"), str) \
                and "/" in item["ref"]:
            owner = item["ref"].split("/")[0].strip()
            if owner:
                return owner
    return None


def check_oauth_cli(runner=None):
    """Validate current Kaggle OAuth CLI authentication (non-destructive).

    Returns ``{"ok", "username", "detail"}``. ``username`` is the account
    owner when a dataset ref reveals it, else None (auth may still be
    valid, e.g. an account with no datasets). Never reads or prints
    OAuth token contents.
    """
    ok, payload = _run_cli_json(_OAUTH_PROBE_CMD, runner)
    if not ok:
        return {"ok": False, "username": None,
                "detail": "kaggle OAuth CLI probe failed "
                          "(datasets list --mine unsuccessful)"}
    owner = _owner_from_dataset_refs(payload)
    if owner:
        return {"ok": True, "username": owner,
                "detail": f"OAuth CLI authenticated as {owner} "
                          "(kaggle datasets list --mine succeeded)"}
    return {"ok": True, "username": None,
            "detail": "OAuth CLI authenticated (datasets list --mine "
                      "succeeded, no owner ref found)"}


def _kaggle_json_path(kaggle_dir=None) -> Path:
    home = Path(kaggle_dir) if kaggle_dir is not None else Path.home()
    cred = home / ".kaggle" / "kaggle.json"
    # Allow tests to pass a dir that IS the .kaggle dir.
    if home.name == ".kaggle" and home / "kaggle.json" != cred:
        cred = home / "kaggle.json"
    return cred


def _kaggle_json_username(kaggle_dir=None):
    try:
        data = json.loads(_kaggle_json_path(kaggle_dir).read_text(
            encoding="utf-8"))
    except (ValueError, OSError):
        return None
    user = data.get("username") if isinstance(data, dict) else None
    return str(user).strip() or None if user else None


def check_auth(env=None, kaggle_dir=None, cli_runner=None) -> dict:
    """Check Kaggle credentials without reading secrets into logs.

    Order: legacy env, legacy kaggle.json, current OAuth CLI probe.
    Returns ``{"ok": bool, "method": str, "detail": str}`` with method
    in ``env`` / ``kaggle_json`` / ``oauth_cli`` / ``missing``. Secret
    VALUES (keys, tokens) are never included in the result.
    """
    env = env if env is not None else os.environ
    if env.get("KAGGLE_USERNAME") and env.get("KAGGLE_KEY"):
        return {
            "ok": True,
            "method": "env",
            "detail": "KAGGLE_USERNAME + KAGGLE_KEY are set",
        }
    cred = _kaggle_json_path(kaggle_dir)
    if cred.is_file():
        try:
            data = json.loads(cred.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {
                "ok": False,
                "method": "missing",
                "detail": f"{cred} is not valid JSON",
            }
        if isinstance(data, dict) and data.get("username") \
                and data.get("key"):
            return {
                "ok": True,
                "method": "kaggle_json",
                "detail": f"credentials found at {cred}",
            }
        return {
            "ok": False,
            "method": "missing",
            "detail": f"{cred} lacks username/key fields",
        }
    oauth = check_oauth_cli(runner=cli_runner)
    if oauth["ok"]:
        return {"ok": True, "method": "oauth_cli",
                "detail": oauth["detail"]}
    return {
        "ok": False,
        "method": "missing",
        "detail": (
            "Kaggle auth missing: run `kaggle auth login`, or set "
            "KAGGLE_USERNAME + KAGGLE_KEY, or place kaggle.json at "
            f"{Path.home() / '.kaggle' / 'kaggle.json'}. "
            "See infra/kaggle/README.md. Never paste secrets into source."
        ),
    }


def resolve_username(explicit=None, env=None, kaggle_dir=None,
                     cli_runner=None) -> str:
    """Resolve the Kaggle account username, or raise ValueError.

    Order: explicit ``--kaggle-user`` > ``KAGGLE_USERNAME`` env > legacy
    kaggle.json username > authenticated OAuth CLI account owner.
    Never returns a placeholder: unresolvable means a clear failure
    before any real submission, so generated metadata never contains
    ``YOURUSERNAME``.
    """
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    env = env if env is not None else os.environ
    if env.get("KAGGLE_USERNAME") and str(env["KAGGLE_USERNAME"]).strip():
        return str(env["KAGGLE_USERNAME"]).strip()
    file_user = _kaggle_json_username(kaggle_dir)
    if file_user:
        return file_user
    oauth = check_oauth_cli(runner=cli_runner)
    if oauth["ok"] and oauth["username"]:
        return oauth["username"]
    raise ValueError(
        "Kaggle username unresolved: pass --kaggle-user, set "
        "KAGGLE_USERNAME, provide kaggle.json, or run "
        "`kaggle auth login` with an account owning datasets."
    )


# --------------------------------------------------------------------------
# Execution notebook generation (strategy A)
# --------------------------------------------------------------------------

_AUTOMATION_CELL_SOURCE = [
    f"# {AUTOMATION_CELL_MARKER} (generated): resolve benchmark workspace\n",
    "import os as _m_os\n",
    "from pathlib import Path as _m_Path\n",
    "def _mizaniq_find_workspace():\n",
    "    for _base in (_m_Path('/kaggle/input'), _m_Path('.')):\n",
    "        if not _base.exists():\n",
    "            continue\n",
    "        for _p in sorted(_base.rglob('manifest.json')):\n",
    "            return _p.parent\n",
    "    return None\n",
    "_MIZANIQ_WS = _mizaniq_find_workspace()\n",
    "if _MIZANIQ_WS is not None:\n",
    "    _m_OS_CWD = _m_os.getcwd()\n",
    "    _m_os.chdir(_MIZANIQ_WS)\n",
    "    print(f'mizaniq workspace: {_MIZANIQ_WS} (was {_m_OS_CWD})')\n",
    "else:\n",
    "    print('mizaniq workspace: manifest.json not found under /kaggle/input')\n",
]


def _flip_qwen_switches(source_text: str, model_id: str, docs: list) -> str:
    """Flip the Qwen execution switches in one notebook cell source."""
    out = source_text
    out = out.replace("RUN_QWEN_VISUAL = False", "RUN_QWEN_VISUAL = True")
    out = re.sub(
        r"QWEN_MODEL_ID\s*=\s*'[^']*'",
        f"QWEN_MODEL_ID = '{model_id}'",
        out,
    )
    out = re.sub(
        r"QWEN_DOCS\s*=\s*\[[^\]]*\]",
        f"QWEN_DOCS = {docs!r}",
        out,
    )
    out = out.replace(
        "ALLOW_QWEN_FALLBACK = True", "ALLOW_QWEN_FALLBACK = False"
    )
    return out


def build_execution_notebook(src_path, dest_path, model_id: str,
                             docs: list) -> dict:
    """Generate the Kaggle execution notebook from the source notebook.

    Returns ``{"switches": {...}}`` describing what changed. The committed
    source notebook is never modified and keeps expensive execution OFF.
    """
    src = Path(src_path)
    nb = json.loads(src.read_text(encoding="utf-8"))
    flipped = False
    for cell in nb.get("cells", []):
        joined = "".join(cell.get("source", []))
        if "RUN_QWEN_VISUAL" in joined and "QWEN_MODEL_ID" in joined:
            cell["source"] = _flip_qwen_switches(joined, model_id,
                                                 list(docs)).splitlines(
                keepends=True)
            # Ensure trailing newline on last line for nbformat hygiene.
            if cell["source"] and not cell["source"][-1].endswith("\n"):
                cell["source"][-1] += "\n"
            flipped = True
    if not flipped:
        raise ValueError(
            "source notebook has no Qwen config cell "
            "(RUN_QWEN_VISUAL/QWEN_MODEL_ID)"
        )
    nb["cells"].insert(0, {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": list(_AUTOMATION_CELL_SOURCE),
    })
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(nb, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    return {
        "switches": {
            "RUN_QWEN_VISUAL": True,
            "QWEN_MODEL_ID": model_id,
            "QWEN_DOCS": list(docs),
            "ALLOW_QWEN_FALLBACK": False,
            "automation_cell": AUTOMATION_CELL_MARKER,
        }
    }


# --------------------------------------------------------------------------
# Kernel metadata
# --------------------------------------------------------------------------

def build_kernel_metadata(kernel_id: str, title: str, code_file: str,
                          dataset_slug: str = DEFAULT_DATASET_SLUG) -> dict:
    """Build valid ``kernel-metadata.json`` content.

    GPU on (Kaggle free T4 path), internet on (first model download from
    Hugging Face), benchmark dataset attached by slug. Never includes
    secrets — validated by tests.
    """
    if not kernel_id or "/" not in kernel_id:
        raise ValueError(
            f"kernel id must look like 'owner/slug', got {kernel_id!r}"
        )
    if not dataset_slug or "/" not in dataset_slug:
        raise ValueError(
            f"dataset slug must look like 'owner/slug', got {dataset_slug!r}"
        )
    return {
        "id": kernel_id,
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [dataset_slug],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def metadata_has_secrets(metadata: dict) -> bool:
    """True if metadata text looks like it contains credentials."""
    blob = json.dumps(metadata).lower()
    return any(token in blob for token in (
        "kaggle_key", "kagglekey", "api_key", "apikey", "hf_token",
        "huggingface_token", "secret", "password",
    ))


# --------------------------------------------------------------------------
# Status polling (bounded; never busy-loops)
# --------------------------------------------------------------------------

def classify_kernel_status(status_text) -> str:
    """Map raw ``kaggle kernels status`` output to a terminal bucket."""
    text = str(status_text or "").lower()
    if any(token in text for token in TERMINAL_OK):
        return "COMPLETE"
    if any(token in text for token in TERMINAL_BAD):
        return "ERROR"
    return "RUNNING"


def poll_kernel_status(status_fn, poll_interval_s: float = 20.0,
                       max_wait_minutes: float = 45.0, sleep_fn=None,
                       clock=None) -> dict:
    """Poll until COMPLETE/ERROR or the local wait window expires.

    ``status_fn`` is a zero-arg callable returning raw status text (mocked
    in tests). Returns ``{"final": ..., "elapsed_s": ..., "checks": ...,
    "last_raw": ...}`` where final is COMPLETE, ERROR, or RUNNER_TIMEOUT
    (local window expired — the Kaggle job itself is NOT claimed killed).
    """
    if poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be > 0")
    if max_wait_minutes <= 0:
        raise ValueError("max_wait_minutes must be > 0")
    sleep = sleep_fn or time.sleep
    now = clock or time.monotonic
    deadline = now() + max_wait_minutes * 60.0
    checks, last_raw = 0, ""
    while True:
        last_raw = status_fn()
        checks += 1
        bucket = classify_kernel_status(last_raw)
        if bucket in ("COMPLETE", "ERROR"):
            return {
                "final": bucket,
                "elapsed_s": None,
                "checks": checks,
                "last_raw": str(last_raw),
            }
        if now() >= deadline:
            return {
                "final": "RUNNER_TIMEOUT",
                "elapsed_s": max_wait_minutes * 60.0,
                "checks": checks,
                "last_raw": str(last_raw),
            }
        sleep(poll_interval_s)


def kernel_url(kernel_id: str) -> str:
    """Public Kaggle URL for a kernel slug (for failure inspection)."""
    return f"https://www.kaggle.com/code/{kernel_id}"


# --------------------------------------------------------------------------
# Output download layout (deterministic, collision-safe)
# --------------------------------------------------------------------------

def run_dir_for(results_root=None, run_id: str = None,
                model_alias: str = None) -> Path:
    """Run directory like ``kaggle_qwen_<alias?>_<run-id>``.

    Collision-safe: appends ``-2``, ``-3``, ... when the directory exists.
    Always stays under ``data/benchmark/results``.
    """
    root = Path(results_root) if results_root is not None else RESULTS_ROOT
    stamp = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_stamp = re.sub(r"[^A-Za-z0-9._-]+", "-", stamp).strip("-") or "run"
    if model_alias:
        safe_alias = re.sub(r"[^A-Za-z0-9._-]+", "-",
                            str(model_alias)).strip("-")
        base = f"kaggle_qwen_{safe_alias}_{safe_stamp}"
    else:
        base = f"kaggle_qwen_{safe_stamp}"
    candidate = root / base
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = root / f"{base}-{suffix}"
    return candidate


def local_name_for(kernel_filename: str) -> str:
    """Deterministic local name for a kernel output file."""
    if kernel_filename in DOWNLOAD_RENAMES:
        return DOWNLOAD_RENAMES[kernel_filename]
    if kernel_filename.startswith(RAW_PREFIXES):
        return kernel_filename
    return kernel_filename


def download_paths_stay_in_results(run_dir, filenames,
                                   results_root=None) -> bool:
    """True iff every resolved download path stays under results root."""
    root = (Path(results_root) if results_root is not None
            else RESULTS_ROOT).resolve()
    base = (Path(run_dir)).resolve() if Path(run_dir).is_absolute() \
        else (Path.cwd() / run_dir).resolve()
    if root not in base.parents and base != root:
        return False
    for name in filenames:
        target = (base / local_name_for(name)).resolve()
        if root not in target.parents and target != root:
            return False
    return True


def validate_downloaded_results(run_dir) -> dict:
    """Check the result contract of a downloaded run directory."""
    run_dir = Path(run_dir)
    results_file = run_dir / DOWNLOAD_RENAMES["kaggle_results.json"]
    report = {"ok": False, "results_file": str(results_file),
              "schema_errors": {}, "files": []}
    if not results_file.is_file():
        report["error"] = f"missing {results_file.name}"
        return report
    try:
        payload = json.loads(results_file.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        report["error"] = f"unparseable {results_file.name}: {exc}"
        return report
    items = payload if isinstance(payload, list) else [payload]
    errors = {}
    for item in items:
        if not isinstance(item, dict):
            errors.setdefault("non-dict", []).append(type(item).__name__)
            continue
        # Local import to keep this module dependency-light for Kaggle reuse.
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                                / "src"))
        from ocrbench import schema as _schema
        errs = _schema.validate_result(item)
        if errs:
            errors[item.get("document_id", "?")] = errs
    report["files"] = sorted(p.name for p in run_dir.iterdir()
                             if p.is_file())
    report["schema_errors"] = errors
    report["ok"] = not errors
    return report


# --------------------------------------------------------------------------
# Scoring integration (existing scorer only, never duplicated)
# --------------------------------------------------------------------------

def run_scorer(results_file, out_file=None, runner=None) -> dict:
    """Run ``scripts/score_ocr_results.py`` on a downloaded result file.

    Returns ``{"ok": bool, "stdout": str, "stderr": str,
    "returncode": int}``. Scoring failure never deletes downloads.
    """
    cmd = ["python", str(SCORER_SCRIPT), str(results_file)]
    if out_file is not None:
        cmd += ["--out", str(out_file)]
    run = runner or subprocess.run
    proc = run(cmd, capture_output=True, text=True)
    return {
        "ok": proc.returncode == 0,
        "stdout": getattr(proc, "stdout", "") or "",
        "stderr": getattr(proc, "stderr", "") or "",
        "returncode": proc.returncode,
    }


def build_run_log(kernel_id: str, model_id: str, docs: list,
                  dataset_slug: str, final_status: str, elapsed_s,
                  output_files: list, scorer: dict = None) -> dict:
    """Debuggable run record (never contains secrets)."""
    return {
        "kernel": kernel_id,
        "kernel_url": kernel_url(kernel_id),
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "final_status": final_status,
        "model_id": model_id,
        "is_fallback_model": model_id == MODEL_ALIASES["qwen3-vl-4b"],
        "documents": list(docs),
        "dataset_slug": dataset_slug,
        "elapsed_s": elapsed_s,
        "output_files": list(output_files),
        "scorer": scorer,
    }


def format_summary(kernel_id: str, final_status: str, docs: list,
                   run_dir) -> str:
    """Concise terminal summary lines."""
    lines = [
        f"kernel: {kernel_id} ({kernel_url(kernel_id)})",
        f"status: {final_status}",
        f"docs: {' '.join(docs)}",
        f"outputs: {run_dir}",
    ]
    return "\n".join(lines)
