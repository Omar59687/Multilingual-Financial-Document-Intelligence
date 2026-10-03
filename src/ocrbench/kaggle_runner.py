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
import sys
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

# Kernel outputs -> deterministic local names. The fixed execution
# notebook writes model-specific qwen_visual_* files (plus kaggle_* /
# experiment_meta.json aliases); the downloader prefers the
# model-specific names and falls back to the aliases.
PRIMARY_RESULTS = "qwen_visual_results.json"
PRIMARY_META = "qwen_visual_meta.json"
DOWNLOAD_RENAMES = {
    "kaggle_results.json": PRIMARY_RESULTS,
    "experiment_meta.json": PRIMARY_META,
}
RESULT_FILENAME_CANDIDATES = (PRIMARY_RESULTS, "kaggle_results.json")
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
# Execution notebook generation (strategy A: dedicated Qwen-only notebook)
# --------------------------------------------------------------------------
#
# Root cause of the first real-run failure: the bootstrap did
# ``os.chdir(dataset_workspace)`` and every write used a bare relative
# filename, so artifacts landed on Kaggle's READ-ONLY /kaggle/input
# (``OSError: [Errno 30] Read-only file system`` for qwen_raw_* and
# kaggle_results.json) even though Qwen generation itself had succeeded.
#
# Fixed contract: BENCHMARK_WORKSPACE (read-only, under /kaggle/input)
# for manifest/truth/images; OUTPUT_DIR (/kaggle/working) for EVERY
# write. No chdir anywhere. Tesseract + Paddle cells are replaced with
# skip stubs so a Qwen-only run never installs/initializes Paddle.

KAGGLE_INPUT_ROOT = "/kaggle/input"
KAGGLE_OUTPUT_DIR = "/kaggle/working"

_AUTOMATION_CELL_SOURCE = [
    f"# {AUTOMATION_CELL_MARKER} (generated): Qwen-only visual run\n",
    "# READ benchmark files from the attached dataset (read-only).\n",
    "# WRITE every generated artifact to /kaggle/working. Never\n",
    "# change the working directory into the dataset.\n",
    "from pathlib import Path as _m_Path\n",
    "BENCHMARK_WORKSPACE = None\n",
    "for _m_base in (_m_Path('/kaggle/input'), _m_Path('.')):\n",
    "    if not _m_base.exists():\n",
    "        continue\n",
    "    for _m_p in sorted(_m_base.rglob('manifest.json')):\n",
    "        BENCHMARK_WORKSPACE = _m_p.parent\n",
    "        break\n",
    "    if BENCHMARK_WORKSPACE is not None:\n",
    "        break\n",
    "print(f'mizaniq benchmark workspace (read-only): {BENCHMARK_WORKSPACE}')\n",
    "OUTPUT_DIR = _m_Path('/kaggle/working')\n",
    "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n",
    "print(f'mizaniq output dir: {OUTPUT_DIR}')\n",
    "# Qwen-only execution controls (the committed notebook has no\n",
    "# RUN_TESSERACT/RUN_PADDLE switches; Tesseract+Paddle cells are\n",
    "# replaced with skip stubs below for automated runs).\n",
    "RUN_TESSERACT = False\n",
    "RUN_PADDLE = False\n",
    "RUN_PADDLE_VISUAL = False\n",
    "RUN_QWEN_VISUAL = True\n",
]


def _stub_cell_source(reason: str) -> list:
    """Replacement source for a cell skipped in Qwen-only runs."""
    return [
        f"# {AUTOMATION_CELL_MARKER}: {reason}\n",
        f"print('{AUTOMATION_CELL_MARKER}: skipped ({reason})')\n",
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


def _qwen_only_cell_transform(source_text: str) -> str:
    """Point every Qwen write at OUTPUT_DIR + classify write failures.

    Write failures after successful generation are labeled
    OUTPUT_WRITE_FAILED so they are never misread as model inference
    errors (MODEL_INFERENCE_FAILED); the original exception text is
    preserved in both cases.
    """
    out = source_text.replace(
        "rp = f'qwen_raw_{doc_id}.json'",
        "rp = str(OUTPUT_DIR / f'qwen_raw_{doc_id}.json')",
    )
    out = out.replace(
        "                with open(rp, 'w', encoding='utf-8') as f:\n"
        "                    json.dump({'generated_text': gen}, f,"
        " ensure_ascii=False)\n",
        "                try:\n"
        "                    with open(rp, 'w', encoding='utf-8') as f:\n"
        "                        json.dump({'generated_text': gen}, f,"
        " ensure_ascii=False)\n"
        "                except OSError as _wexc:\n"
        "                    r['warnings'].append(\n"
        "                        f'OUTPUT_WRITE_FAILED: {_wexc}')\n"
        "                    raise\n",
    )
    out = out.replace(
        "            except Exception as e:\n"
        "                r['warnings'].append(f'inference failed: "
        "{type(e).__name__}: {e}')",
        "            except Exception as e:\n"
        "                _w = [w for w in r['warnings']\n"
        "                      if 'OUTPUT_WRITE_FAILED' in w]\n"
        "                _kind = ('OUTPUT_WRITE_FAILED' if _w\n"
        "                         else 'MODEL_INFERENCE_FAILED')\n"
        "                r['warnings'].append(\n"
        "                    f'{_kind}: {type(e).__name__}: {e}')",
    )
    return out


def _export_cell_transform(source_text: str) -> str:
    """Point export writes at OUTPUT_DIR + keep model-specific aliases."""
    out = source_text.replace(
        "with open('kaggle_results.json', 'w', encoding='utf-8') as f:",
        "with open(OUTPUT_DIR / 'kaggle_results.json', 'w',"
        " encoding='utf-8') as f:",
    )
    out = out.replace(
        "with open('experiment_meta.json', 'w', encoding='utf-8') as f:",
        "with open(OUTPUT_DIR / 'experiment_meta.json', 'w',"
        " encoding='utf-8') as f:",
    )
    out += (
        "\n(OUTPUT_DIR / 'qwen_visual_results.json').write_bytes(\n"
        "    (OUTPUT_DIR / 'kaggle_results.json').read_bytes())\n"
        "(OUTPUT_DIR / 'qwen_visual_meta.json').write_bytes(\n"
        "    (OUTPUT_DIR / 'experiment_meta.json').read_bytes())\n"
        "print('saved qwen_visual_results.json + qwen_visual_meta.json '\n"
        "      '(aliases: kaggle_results.json + experiment_meta.json)')\n"
    )
    return out


def _set_cell_source(cell, source_text: str) -> None:
    lines = source_text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    cell["source"] = lines


def build_execution_notebook(src_path, dest_path, model_id: str,
                             docs: list) -> dict:
    """Generate the Qwen-only Kaggle execution notebook.

    Transformations (committed source untouched, manual defaults kept):
    - prepend bootstrap defining BENCHMARK_WORKSPACE (/kaggle/input
      discovery) and OUTPUT_DIR (/kaggle/working); no chdir;
    - ``WORKSPACE = Path('.')`` -> ``WORKSPACE = BENCHMARK_WORKSPACE``;
    - Paddle install/imports, Tesseract, and Paddle experiment cells ->
      skip stubs (Qwen-only: no Paddle install, no Paddle init);
    - Qwen config cell -> explicit model/docs, fallback OFF;
    - Qwen + export writes -> OUTPUT_DIR, with OUTPUT_WRITE_FAILED vs
      MODEL_INFERENCE_FAILED classification and qwen_visual_* aliases.

    Returns ``{"switches": {...}}`` describing what changed.
    """
    src = Path(src_path)
    nb = json.loads(src.read_text(encoding="utf-8"))
    seen = {"workspace": False, "qwen": False, "export": False}
    stubs = {"tesseract": False, "paddle_install": False,
             "paddle_imports": False, "paddle_experiment": False}
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        joined = "".join(cell.get("source", []))
        if "RUN_QWEN_VISUAL" in joined and "QWEN_MODEL_ID" in joined:
            transformed = _qwen_only_cell_transform(
                _flip_qwen_switches(joined, model_id, list(docs)))
            _set_cell_source(cell, transformed)
            seen["qwen"] = True
        elif "kaggle_results.json" in joined and "experiment_meta.json" \
                in joined and "open(" in joined:
            _set_cell_source(cell, _export_cell_transform(joined))
            seen["export"] = True
        elif "WORKSPACE = Path('.')" in joined:
            _set_cell_source(
                cell,
                joined.replace("WORKSPACE = Path('.')",
                               "WORKSPACE = BENCHMARK_WORKSPACE"))
            seen["workspace"] = True
        elif "TESS_DOCS" in joined:
            cell["source"] = _stub_cell_source(
                "Tesseract skipped (Qwen-only automated run)")
            stubs["tesseract"] = True
        elif "%pip install" in joined and "paddlepaddle-gpu" in joined:
            cell["source"] = _stub_cell_source(
                "Paddle install skipped (Qwen-only automated run)")
            stubs["paddle_install"] = True
        elif "STEP 2/2: imports" in joined:
            cell["source"] = _stub_cell_source(
                "Paddle imports skipped (Qwen-only automated run)")
            stubs["paddle_imports"] = True
        elif "PADDLE_MODEL_ID" in joined:
            cell["source"] = _stub_cell_source(
                "Paddle experiment skipped (Qwen-only automated run)")
            stubs["paddle_experiment"] = True
    missing = [k for k, v in {**seen, **stubs}.items() if not v]
    if missing:
        raise ValueError(
            f"source notebook missing expected cells: {missing}")
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
            "RUN_TESSERACT": False,
            "RUN_PADDLE": False,
            "RUN_PADDLE_VISUAL": False,
            "RUN_QWEN_VISUAL": True,
            "QWEN_MODEL_ID": model_id,
            "QWEN_DOCS": list(docs),
            "ALLOW_QWEN_FALLBACK": False,
            "BENCHMARK_WORKSPACE": KAGGLE_INPUT_ROOT,
            "OUTPUT_DIR": KAGGLE_OUTPUT_DIR,
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
    """Check the result contract of a downloaded run directory.

    Accepts the model-specific ``qwen_visual_results.json`` or the
    legacy ``kaggle_results.json`` alias (scorer integration unchanged).
    """
    run_dir = Path(run_dir)
    results_file = None
    for candidate in RESULT_FILENAME_CANDIDATES:
        if (run_dir / candidate).is_file():
            results_file = run_dir / candidate
            break
    if results_file is None:
        results_file = run_dir / PRIMARY_RESULTS
        report = {"ok": False, "results_file": str(results_file),
                  "schema_errors": {}, "files": []}
        report["error"] = f"missing {PRIMARY_RESULTS} (or alias)"
        return report
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

    The scorer is launched in UTF-8 mode (``-X utf8`` plus ``PYTHONUTF8=1``)
    so Arabic output cannot crash under Windows cp1252 consoles. Score
    content is unchanged. Returns ``{"ok": bool, "stdout": str,
    "stderr": str, "returncode": int}``. Scoring failure never deletes
    downloads.
    """
    cmd = [sys.executable, "-X", "utf8", str(SCORER_SCRIPT),
           str(results_file)]
    if out_file is not None:
        cmd += ["--out", str(out_file)]
    run = runner or subprocess.run
    env = dict(os.environ, PYTHONUTF8="1")
    proc = run(cmd, capture_output=True, text=True, env=env)
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
