"""Submit the Phase-2 Qwen visual benchmark to Kaggle (free GPU path).

ONE explicit command = ONE Kaggle experiment submission. No retries, no
background polling, no auto-submit during implementation — use --dry-run
to validate locally without touching Kaggle.

Examples:
    python scripts/kaggle_submit_visual.py --dry-run
    python scripts/kaggle_submit_visual.py --model qwen3-vl-4b --docs DEV-008 DEV-009 --score
    python scripts/kaggle_submit_visual.py --model qwen3-vl-8b --docs DEV-008 DEV-009
"""

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocrbench import kaggle_runner as K  # noqa: E402

EXEC_NOTEBOOK_NAME = "mizaniq_qwen_visual_exec.ipynb"
ACCELERATOR = "NvidiaTeslaT4"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit the Qwen visual benchmark to Kaggle.")
    parser.add_argument(
        "--model", default=K.DEFAULT_MODEL_ALIAS,
        help=f"one of {sorted(K.MODEL_ALIASES)} "
             "(exact HF ids also accepted; 8B never silently becomes 4B)")
    parser.add_argument(
        "--docs", nargs="*", default=None,
        help="subset of DEV-008 DEV-009 (default: both)")
    parser.add_argument(
        "--dataset", default=K.DEFAULT_DATASET_SLUG,
        help="Kaggle dataset slug owner/name")
    parser.add_argument(
        "--kaggle-user", default=None,
        help="Kaggle username for the kernel slug (default: autodetect "
             "from KAGGLE_USERNAME / kaggle.json / OAuth CLI)")
    parser.add_argument(
        "--kernel-id", default=None,
        help="kernel slug owner/name (default: derived from username)")
    parser.add_argument(
        "--title", default="mizaniq-qwen-visual",
        help="kernel title for new kernels")
    parser.add_argument(
        "--poll-interval", type=float, default=20.0,
        help="seconds between status checks")
    parser.add_argument(
        "--max-wait-minutes", type=float, default=45.0,
        help="local wait window before RUNNER_TIMEOUT")
    parser.add_argument(
        "--score", action="store_true",
        help="run scripts/score_ocr_results.py on the downloaded results")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="validate + show what would be submitted; submit nothing")
    parser.add_argument(
        "--staging-dir", default=None,
        help="staging dir for generated notebook/metadata "
             "(default: infra/kaggle/build/<stamp>)")
    return parser


def _run_cli(cmd: list) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    try:
        model_id = K.resolve_model_id(args.model)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        docs = K.select_docs(args.docs)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    src_nb = ROOT / K.SOURCE_NOTEBOOK
    if not src_nb.is_file():
        print(f"error: source notebook missing: {src_nb}", file=sys.stderr)
        return 2

    auth = K.check_auth()
    try:
        username = K.resolve_username(args.kaggle_user)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    kernel_id = args.kernel_id or f"{username}/mizaniq-qwen-visual"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    staging = (Path(args.staging_dir) if args.staging_dir
               else ROOT / "infra" / "kaggle" / "build" / f"run-{stamp}")
    staging.mkdir(parents=True, exist_ok=True)
    exec_nb = staging / EXEC_NOTEBOOK_NAME

    try:
        plan = K.build_execution_notebook(src_nb, exec_nb, model_id, docs)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        metadata = K.build_kernel_metadata(
            kernel_id, args.title, EXEC_NOTEBOOK_NAME, args.dataset)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    (staging / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"model: {model_id} (requested: {args.model})")
    print(f"docs: {' '.join(docs)}")
    print(f"dataset: {args.dataset}")
    print(f"kernel: {kernel_id} ({K.kernel_url(kernel_id)})")
    print(f"staging: {staging}")
    print(f"switches: {json.dumps(plan['switches'])}")
    print(f"auth: {auth['method']} ({auth['detail']})")

    if args.dry_run:
        print("dry-run: validated local files + metadata; "
              "nothing submitted to Kaggle.")
        return 0

    if not auth["ok"]:
        print(f"error: {auth['detail']}", file=sys.stderr)
        return 2

    push = _run_cli(["kaggle", "kernels", "push", "-p", str(staging),
                     "--accelerator", ACCELERATOR])
    print(push.stdout or "")
    if push.returncode != 0:
        print(push.stderr or "kaggle kernels push failed",
              file=sys.stderr)
        print(K.format_summary(kernel_id, "ERROR", docs, "(no outputs)"))
        return 1

    def _status_fn() -> str:
        proc = _run_cli(["kaggle", "kernels", "status", kernel_id])
        return (proc.stdout or "") + (proc.stderr or "")

    poll = K.poll_kernel_status(
        _status_fn, poll_interval_s=args.poll_interval,
        max_wait_minutes=args.max_wait_minutes)
    print(f"poll: {poll['final']} after {poll['checks']} checks; "
          f"last: {poll['last_raw'][:200]}")
    if poll["final"] != "COMPLETE":
        print(K.format_summary(kernel_id, poll["final"], docs,
                               "(no outputs)"))
        if poll["final"] == "RUNNER_TIMEOUT":
            print("RUNNER_TIMEOUT: local wait window expired; the Kaggle "
                  "job was NOT killed. Inspect it at the kernel URL above.")
        return 1

    run_dir = K.run_dir_for(ROOT / K.RESULTS_ROOT,
                            run_id=f"{stamp}_{kernel_id.replace('/', '-')}",
                            model_alias=args.model)
    run_dir.mkdir(parents=True, exist_ok=False)
    out = _run_cli(["kaggle", "kernels", "output", kernel_id, "-p",
                    str(run_dir)])
    if out.returncode != 0:
        print(out.stderr or "kaggle kernels output failed", file=sys.stderr)
        return 1
    # Deterministic local names: the fixed notebook writes model-specific
    # qwen_visual_* files (plus kaggle_* aliases); accept either.
    for local_name, kernel_name in (
            (K.PRIMARY_RESULTS, "kaggle_results.json"),
            (K.PRIMARY_META, "experiment_meta.json")):
        dest_file, src_file = run_dir / local_name, run_dir / kernel_name
        if not dest_file.exists() and src_file.is_file():
            dest_file.write_bytes(src_file.read_bytes())

    downloaded = sorted(p.name for p in run_dir.iterdir() if p.is_file())
    print(f"downloaded: {', '.join(downloaded) or '(none)'} -> {run_dir}")

    validation = K.validate_downloaded_results(run_dir)
    print(f"contract: {'ok' if validation['ok'] else 'INVALID'} "
          f"{json.dumps(validation.get('schema_errors', {}))[:300]}")

    scorer_record = None
    if args.score:
        results_file = Path(
            K.validate_downloaded_results(run_dir)["results_file"])
        scored = K.run_scorer(results_file)
        scorer_record = {"ok": scored["ok"],
                         "returncode": scored["returncode"]}
        print(scored["stdout"] or "")
        if not scored["ok"]:
            print(f"scoring failed (downloads preserved): "
                  f"{scored['stderr'][:500]}", file=sys.stderr)

    log = K.build_run_log(kernel_id, model_id, docs, args.dataset,
                          poll["final"], poll.get("elapsed_s"), downloaded,
                          scorer_record)
    (run_dir / "kaggle_run_log.json").write_text(
        json.dumps(log, indent=2), encoding="utf-8")
    print(K.format_summary(kernel_id, poll["final"], docs, run_dir))
    if not validation["ok"]:
        return 1
    return 0 if scorer_record is None or scorer_record["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
