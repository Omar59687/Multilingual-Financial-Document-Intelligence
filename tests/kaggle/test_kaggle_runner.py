"""Deterministic local tests for the Kaggle automation runner.

No real Kaggle submissions, no GPU, no network. Kaggle CLI/API
boundaries are mocked (fake status functions / fake subprocess runners).
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ocrbench import kaggle_runner as K  # noqa: E402
from ocrbench import qwen_adapter, schema  # noqa: E402


def _load_submit_script():
    spec = importlib.util.spec_from_file_location(
        "kaggle_submit_visual",
        str(ROOT / "scripts" / "kaggle_submit_visual.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 1. defaults ------------------------------------------------------------------
def test_1_default_docs_are_dev008_009():
    assert K.select_docs(None) == ["DEV-008", "DEV-009"]
    assert K.select_docs([]) == ["DEV-008", "DEV-009"]
    script = _load_submit_script()
    args = script.build_parser().parse_args([])
    assert args.model == "qwen3-vl-4b"
    assert K.select_docs(args.docs) == ["DEV-008", "DEV-009"]


# 2. model mapping ---------------------------------------------------------------
def test_2_model_aliases_map_to_exact_ids():
    assert K.resolve_model_id("qwen3-vl-4b") == \
        "Qwen/Qwen3-VL-4B-Instruct" == qwen_adapter.FALLBACK_MODEL_ID
    assert K.resolve_model_id("qwen3-vl-8b") == \
        "Qwen/Qwen3-VL-8B-Instruct" == qwen_adapter.PRIMARY_MODEL_ID
    assert K.resolve_model_id(None) == "Qwen/Qwen3-VL-4B-Instruct"


# 3. unsupported model --------------------------------------------------------------
def test_3_unsupported_model_rejected():
    with pytest.raises(ValueError):
        K.resolve_model_id("qwen3-vl-70b")
    with pytest.raises(ValueError):
        K.resolve_model_id("")
    with pytest.raises(ValueError):
        K.select_docs(["DEV-004"])


# 4. GPU metadata -----------------------------------------------------------------------
def test_4_metadata_gpu_enabled():
    meta = K.build_kernel_metadata("user/slug", "t", "nb.ipynb",
                                   "owner/mizaniq-ocr-benchmark-v0-1")
    assert meta["enable_gpu"] == "true"
    assert meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["language"] == "python"
    assert meta["kernel_type"] == "notebook"


# 5. dataset slug ------------------------------------------------------------------------------
def test_5_metadata_attaches_dataset_slug():
    meta = K.build_kernel_metadata("user/slug", "t", "nb.ipynb",
                                   "owner/mizaniq-ocr-benchmark-v0-1")
    assert "owner/mizaniq-ocr-benchmark-v0-1" in meta["dataset_sources"]
    assert K.DEFAULT_DATASET_SLUG == \
        "omarabdallah12/mizaniq-ocr-benchmark-v0-1"
    with pytest.raises(ValueError):
        K.build_kernel_metadata("user/slug", "t", "nb.ipynb", "noslash")


# 6. code file ------------------------------------------------------------------------------------------------
def test_6_metadata_points_to_generated_notebook(tmp_path):
    dest = tmp_path / "mizaniq_qwen_visual_exec.ipynb"
    K.build_execution_notebook(ROOT / K.SOURCE_NOTEBOOK, dest,
                               K.resolve_model_id("qwen3-vl-4b"),
                               ["DEV-008", "DEV-009"])
    meta = K.build_kernel_metadata("user/slug", "t", dest.name,
                                   "o/mizaniq-ocr-benchmark-v0-1")
    assert meta["code_file"] == dest.name
    assert dest.is_file()
    # Source notebook keeps expensive execution OFF.
    src = (ROOT / K.SOURCE_NOTEBOOK).read_text(encoding="utf-8")
    assert "RUN_QWEN_VISUAL = False" in src
    gen = dest.read_text(encoding="utf-8")
    assert "RUN_QWEN_VISUAL = True" in gen
    assert K.AUTOMATION_CELL_MARKER in gen


# 7. dry-run ----------------------------------------------------------------------------------------------------------
def test_7_dry_run_submits_nothing(tmp_path, capsys):
    script = _load_submit_script()
    staging = tmp_path / "staging"

    def _boom(cmd, **kwargs):
        raise AssertionError(f"must not call Kaggle: {cmd}")

    script._run_cli = _boom
    rc = script.main(["--dry-run", "--staging-dir", str(staging),
                      "--model", "qwen3-vl-4b"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert (staging / "kernel-metadata.json").is_file()
    assert (staging / script.EXEC_NOTEBOOK_NAME).is_file()


# 8/9/10. polling ------------------------------------------------------------------------------------------------------------
def test_8_poll_handles_complete():
    calls = iter(["running", "complete"])
    res = K.poll_kernel_status(lambda: next(calls), poll_interval_s=20,
                               max_wait_minutes=45, sleep_fn=lambda s: None)
    assert res["final"] == "COMPLETE"
    assert res["checks"] == 2


def test_9_poll_handles_error():
    res = K.poll_kernel_status(lambda: "error: CUDA OOM",
                               poll_interval_s=20, max_wait_minutes=45,
                               sleep_fn=lambda s: None)
    assert res["final"] == "ERROR"


def test_10_poll_handles_runner_timeout():
    ticks = {"t": 0.0}
    res = K.poll_kernel_status(
        lambda: "running", poll_interval_s=20, max_wait_minutes=45,
        sleep_fn=lambda s: ticks.update(t=ticks["t"] + 20),
        clock=lambda: ticks["t"])
    assert res["final"] == "RUNNER_TIMEOUT"
    assert res["checks"] >= 2
    with pytest.raises(ValueError):
        K.poll_kernel_status(lambda: "running", poll_interval_s=0,
                             max_wait_minutes=1)
    assert K.classify_kernel_status("CoMpLeTe") == "COMPLETE"
    assert K.classify_kernel_status("some running job") == "RUNNING"


# 11. run dir -----------------------------------------------------------------------------------------------------------------------
def test_11_run_dir_deterministic_and_collision_safe(tmp_path):
    first = tmp_path / "kaggle_qwen_qwen3-vl-4b_RUN1"
    first.mkdir()
    second = K.run_dir_for(tmp_path, run_id="RUN1",
                           model_alias="qwen3-vl-4b")
    assert second != first
    assert second.name == "kaggle_qwen_qwen3-vl-4b_RUN1-2"
    assert K.run_dir_for(tmp_path, run_id="RUN1",
                         model_alias="qwen3-vl-4b") == second


# 12. paths --------------------------------------------------------------------------------------------------------------------------------
def test_12_download_paths_stay_in_results(tmp_path):
    run_dir = tmp_path / "data" / "benchmark" / "results" / "kaggle_qwen_x"
    run_dir.mkdir(parents=True)
    root = tmp_path / "data" / "benchmark" / "results"
    assert K.download_paths_stay_in_results(
        run_dir, ["kaggle_results.json", "qwen_raw_DEV-008.json"],
        results_root=root)
    assert K.local_name_for("kaggle_results.json") == \
        "qwen_visual_results.json"
    assert K.local_name_for("experiment_meta.json") == \
        "qwen_visual_meta.json"
    assert K.local_name_for("qwen_raw_DEV-008.json") == \
        "qwen_raw_DEV-008.json"


# 13. secrets ----------------------------------------------------------------------------------------------------------------------------------
def test_13_no_secrets_in_metadata():
    meta = K.build_kernel_metadata("user/slug", "t", "nb.ipynb",
                                   "o/mizaniq-ocr-benchmark-v0-1")
    assert not K.metadata_has_secrets(meta)
    template = json.loads(
        (ROOT / "infra" / "kaggle" / "kernel-metadata.json").read_text(
            encoding="utf-8"))
    assert not K.metadata_has_secrets(template)
    assert K.metadata_has_secrets({"x": "my kaggle_key abc"}) is True


# 14. scorer ----------------------------------------------------------------------------------------------------------------------------------------
def test_14_score_invokes_existing_scorer_not_duplicate():
    seen = {}

    class _Proc:
        returncode = 0
        stdout = "scored"
        stderr = ""

    def _fake(cmd, **kwargs):
        seen["cmd"] = cmd
        return _Proc()

    res = K.run_scorer("some/results.json", runner=_fake)
    assert res["ok"] is True
    assert "score_ocr_results.py" in seen["cmd"][1]
    assert "results.json" in seen["cmd"][2]


# 15. auth -------------------------------------------------------------------------------------------------------------------------------
def test_15_missing_auth_clear_error(tmp_path):
    res = K.check_auth(env={}, kaggle_dir=tmp_path)
    assert res["ok"] is False
    assert "KAGGLE_USERNAME" in res["detail"]
    assert "kaggle.json" in res["detail"]
    good = K.check_auth(
        env={"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"},
        kaggle_dir=tmp_path)
    assert good["ok"] is True and good["method"] == "env"
    cred_dir = tmp_path / ".kaggle"
    cred_dir.mkdir()
    (cred_dir / "kaggle.json").write_text(
        json.dumps({"username": "u", "key": "k"}), encoding="utf-8")
    file_auth = K.check_auth(env={}, kaggle_dir=tmp_path)
    assert file_auth["ok"] is True


# 16. no silent 8B->4B ------------------------------------------------------------------------------------------------------------------------------
def test_16_selected_8b_never_silently_becomes_4b(tmp_path):
    assert K.resolve_model_id("qwen3-vl-8b") != K.resolve_model_id(
        "qwen3-vl-4b")
    dest = tmp_path / "exec.ipynb"
    K.build_execution_notebook(ROOT / K.SOURCE_NOTEBOOK, dest,
                               K.resolve_model_id("qwen3-vl-8b"),
                               ["DEV-008", "DEV-009"])
    nb = json.loads(dest.read_text(encoding="utf-8"))
    qwen_cells = ["".join(c.get("source", [])) for c in nb["cells"]
                  if "QWEN_MODEL_ID" in "".join(c.get("source", []))]
    assert len(qwen_cells) == 1
    assert "Qwen/Qwen3-VL-8B-Instruct" in qwen_cells[0]
    assert "ALLOW_QWEN_FALLBACK = False" in qwen_cells[0]
    assert "Qwen/Qwen3-VL-4B-Instruct" not in [
        line for line in qwen_cells[0].splitlines()
        if line.strip().startswith("QWEN_MODEL_ID")]


# 17. contract -----------------------------------------------------------------------------------------------------------------------------------
def test_17_downloaded_result_contract(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert K.validate_downloaded_results(run_dir)["ok"] is False
    valid = [schema.blank_result("qwen3-vl",
                                 qwen_adapter.PRIMARY_MODEL_ID,
                                 "kaggle-T4", "DEV-008")]
    (run_dir / "qwen_visual_results.json").write_text(
        json.dumps(valid), encoding="utf-8")
    report = K.validate_downloaded_results(run_dir)
    assert report["ok"] is True
    assert report["schema_errors"] == {}


# 18. kernel URL -------------------------------------------------------------------------------------------------------------------------------
def test_18_kernel_url_printed_on_failure():
    text = K.format_summary("owner/slug", "ERROR", ["DEV-008"], "/tmp/x")
    assert "https://www.kaggle.com/code/owner/slug" in text
    assert "ERROR" in text
    log = K.build_run_log("owner/slug", qwen_adapter.PRIMARY_MODEL_ID,
                          ["DEV-008"], "o/d", "ERROR", 12.0, [])
    assert log["kernel_url"].endswith("owner/slug")
    assert "KAGGLE_KEY" not in json.dumps(log)
