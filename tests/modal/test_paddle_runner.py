"""Local tests for the Modal PaddleOCR-VL runner (no GPU, no network).

Covers the task gates with the Modal/remote boundary mocked:
 1. defaults to DEV-008/009
 2. unsupported document ID rejected
 3. local input file loading
 4. result schema validation
 5. OK serialization
 6. TIMEOUT serialization/handling
 7. FAILED serialization/handling
 8. deterministic local output paths
 9. raw output path naming
10. no ground-truth leakage into inference input
11. chart_recognition recorded true
12. GPU configuration is single T4 (A10G billing-blocked fallback)
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ocrbench import paddle_adapter, schema  # noqa: E402


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "modal_paddle_runner", str(ROOT / "infra" / "modal" / "paddle_runner.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()


# 1. defaults ---------------------------------------------------------------
def test_1_defaults_to_dev008_009():
    assert RUNNER.select_documents(None) == ["DEV-008", "DEV-009"]
    assert RUNNER.select_documents([]) == ["DEV-008", "DEV-009"]
    assert RUNNER.select_documents("") == ["DEV-008", "DEV-009"]
    assert RUNNER.DEFAULT_DOCS == ["DEV-008", "DEV-009"]


# 2. unsupported rejected ----------------------------------------------------
def test_2_unsupported_document_rejected():
    with pytest.raises(ValueError):
        RUNNER.select_documents(["DEV-004"])
    with pytest.raises(ValueError):
        RUNNER.select_documents("DEV-008,DEV-010")
    with pytest.raises(ValueError):
        RUNNER.resolve_input_path("DEV-004")
    with pytest.raises(ValueError):
        RUNNER.load_inference_input("QWEN-001")


# 3. local input loading ------------------------------------------------------
def test_3_local_input_file_loading():
    for doc_id in ("DEV-008", "DEV-009"):
        payload = RUNNER.load_inference_input(doc_id)
        assert payload["document_id"] == doc_id
        assert isinstance(payload["image_bytes"], bytes)
        assert len(payload["image_bytes"]) > 1000
        assert payload["size_bytes"] == len(payload["image_bytes"])
        assert len(payload["sha256"]) == 64
        assert payload["filename"]
        # Resolves inside the repo to a real workspace image.
        resolved = RUNNER.resolve_input_path(doc_id)
        assert resolved.is_file()
        assert resolved.name == payload["filename"]


# 4. schema validation ----------------------------------------------------------
def test_4_result_schema_validation():
    ok = RUNNER.build_ok_result("DEV-008", 1234.0, text="hello")
    assert RUNNER.validate_schema(ok) == []
    assert schema.validate_result(ok) == []
    broken = dict(ok)
    del broken["fields"]
    assert RUNNER.validate_schema(broken)
    assert schema.validate_result(broken)


# 5. OK serialization ------------------------------------------------------------
def test_5_ok_result_serialization():
    ok = RUNNER.build_ok_result(
        "DEV-008",
        12345.0,
        text="Revenue 2024",
        fields={},
        tables=[[["A", "1"]]],
        visual_description="steady growth",
        warnings=["n1"],
        paddle_version="3.2.1",
        paddleocr_version="3.7.0",
        init_latency_ms=999.0,
    )
    assert ok["status"] == "OK"
    assert ok["model"] == "paddleocr-vl"
    assert ok["model_version"] == "PaddleOCR-VL-1.6"
    assert ok["document_id"] == "DEV-008"
    # JSON round-trip (what gets written to disk).
    clone = json.loads(json.dumps(ok, ensure_ascii=False))
    assert clone == ok
    assert RUNNER.validate_schema(clone) == []


def test_5b_adapt_remote_payload_uses_real_v16_adapter():
    raw_pages = [
        {
            "res": {
                "parsing_res_list": [
                    {
                        "block_label": "text",
                        "block_content": "Annual Revenue 2024",
                        "block_bbox": [0, 0, 1, 1],
                    }
                ]
            }
        }
    ]
    payload = {
        "status": "OK",
        "document_id": "DEV-008",
        "latency_ms": 5000.0,
        "init_latency_ms": 80000.0,
        "raw_pages": raw_pages,
        "paddle_version": "3.2.1",
        "paddleocr_version": "3.7.0",
        "pipeline_version": "v1.6",
        "chart_recognition": True,
        "gpu_type": "T4",
        "device": RUNNER.DEVICE_LABEL,
        "warnings": [],
    }
    result = RUNNER.adapt_remote_payload(payload, raw_path="raw.json")
    assert result["status"] == "OK"
    assert result["gpu_type"] == "T4"
    assert "Annual Revenue 2024" in result["text"]
    assert result["fields"] == {}  # v1.6 never invents semantic fields
    assert RUNNER.validate_schema(result) == []


# 6. TIMEOUT ----------------------------------------------------------------------
def test_6_timeout_serialization_and_handling():
    res = RUNNER.build_timeout_result("DEV-008")
    assert res["status"] == "TIMEOUT"
    assert res["latency_ms"] == float(RUNNER.TIMEOUT_MS) == 300000.0
    assert RUNNER.validate_schema(res) == []
    assert json.loads(json.dumps(res))["status"] == "TIMEOUT"
    line = RUNNER.format_summary_line("DEV-008", "TIMEOUT", 300000)
    assert line == "DEV-008 | TIMEOUT | >300000 ms"
    # Modal timeout maps to TIMEOUT; generic errors map to FAILED.
    import modal.exception

    assert (
        RUNNER.classify_remote_error(
            modal.exception.FunctionTimeoutError("timed out")
        )
        == "TIMEOUT"
    )
    assert RUNNER.classify_remote_error(TimeoutError("x")) == "TIMEOUT"
    assert RUNNER.classify_remote_error(ValueError("boom")) == "FAILED"


# 7. FAILED -------------------------------------------------------------------------
def test_7_failed_serialization_and_handling():
    res = RUNNER.build_failed_result("DEV-009", "CUDA OOM (simulated)")
    assert res["status"] == "FAILED"
    assert RUNNER.validate_schema(res) == []
    assert json.loads(json.dumps(res))["status"] == "FAILED"
    line = RUNNER.format_summary_line("DEV-009", "FAILED", 12.5)
    assert line == "DEV-009 | FAILED | 12 ms"
    # FAILED remote payloads adapt to FAILED results, never crash.
    adapted = RUNNER.adapt_remote_payload(
        {"status": "FAILED", "document_id": "DEV-009",
         "latency_ms": 0.0, "error": "init blew up"}
    )
    assert adapted["status"] == "FAILED"
    assert RUNNER.validate_schema(adapted) == []


# 8. deterministic output paths ----------------------------------------------------------
def test_8_deterministic_local_output_paths():
    assert RUNNER.results_path() == RUNNER.results_path()
    assert RUNNER.meta_path() == RUNNER.meta_path()
    assert str(RUNNER.results_path()).endswith(
        "modal_paddle_visual_results.json"
    )
    assert str(RUNNER.meta_path()).endswith("modal_paddle_visual_meta.json")
    assert RUNNER.results_path().parent == RUNNER.meta_path().parent


def test_8b_save_local_outputs_roundtrip(tmp_path):
    results = [RUNNER.build_ok_result("DEV-008", 100.0, text="t")]
    meta = {"app": RUNNER.APP_NAME}
    raw = {"DEV-008": {"status": "OK", "raw_pages": []}}
    first = RUNNER.save_local_outputs(results, meta, raw, root=tmp_path)
    second = RUNNER.save_local_outputs(results, meta, raw, root=tmp_path)
    assert first == second  # deterministic
    assert Path(first["results"]).is_file()
    assert Path(first["meta"]).is_file()
    assert Path(first["raw"]["DEV-008"]).is_file()
    loaded = json.loads(Path(first["results"]).read_text(encoding="utf-8"))
    assert loaded[0]["document_id"] == "DEV-008"


# 9. raw naming ------------------------------------------------------------------------------
def test_9_raw_output_path_naming():
    assert RUNNER.raw_filename_for("DEV-008") == "modal_paddle_raw_DEV-008.json"
    assert RUNNER.raw_filename_for("DEV-009") == "modal_paddle_raw_DEV-009.json"
    assert str(RUNNER.raw_path_for("DEV-008")).endswith(
        "modal_paddle_raw_DEV-008.json"
    )
    assert RUNNER.raw_path_for("DEV-008").parent == RUNNER.results_path().parent


# 10. no truth leakage --------------------------------------------------------------------------------
def test_10_no_ground_truth_leakage_into_inference_input():
    for doc_id in ("DEV-008", "DEV-009"):
        payload = RUNNER.load_inference_input(doc_id)
        assert set(payload) <= RUNNER.INFERENCE_INPUT_KEYS
        blob = json.dumps(
            {k: v for k, v in payload.items() if k != "image_bytes"},
            ensure_ascii=False,
        ).lower()
        for banned in ("ground_truth", "expected_", "truth.json", "canonical"):
            assert banned not in blob
        # Input file lives under an inputs/documents workspace, never under
        # a ground_truth directory.
        resolved = RUNNER.resolve_input_path(doc_id)
        assert "ground_truth" not in str(resolved)


# 11. chart_recognition -----------------------------------------------------------------------------------
def test_11_chart_recognition_recorded_true():
    assert RUNNER.CHART_RECOGNITION is True
    assert RUNNER.PIPELINE_VERSION == "v1.6"
    ok = RUNNER.build_ok_result("DEV-009", 1.0)
    assert ok["chart_recognition"] is True
    assert ok["pipeline_version"] == "v1.6"
    assert paddle_adapter.PIPELINE_VERSION == "v1.6"
    source = (ROOT / "infra" / "modal" / "paddle_runner.py").read_text(
        encoding="utf-8"
    )
    assert "use_chart_recognition=True" in source


# 12. GPU config ------------------------------------------------------------------------------------------------
def test_12_gpu_configuration_is_single_t4():
    assert RUNNER.GPU_TYPE == "T4"
    assert RUNNER.DEVICE_LABEL == "modal-T4"
    assert RUNNER.TIMEOUT_S == 300
    assert RUNNER.APP_NAME == "mizaniq-paddle-visual"
    source = (ROOT / "infra" / "modal" / "paddle_runner.py").read_text(
        encoding="utf-8"
    )
    assert 'gpu=GPU_TYPE' in source or 'gpu="T4"' in source
    assert "timeout=TIMEOUT_S" in source or "timeout=300" in source
    assert "retries=0" in source
    # T4 fallback: A10G must no longer be requested (may be mentioned in
    # comments/docstring only as the billing-blocked predecessor).
    assert 'gpu="A10G"' not in source
    assert "gpu='A10G'" not in source
    # No active A100/H100 GPU requests (docstring may mention them only as
    # explicitly excluded) and no multi-GPU request syntax.
    assert 'gpu="A100' not in source
    assert "gpu='A100" not in source
    assert 'gpu="H100' not in source
    assert "gpu='H100" not in source
    assert '"A100:' not in source
    assert '"H100:' not in source
