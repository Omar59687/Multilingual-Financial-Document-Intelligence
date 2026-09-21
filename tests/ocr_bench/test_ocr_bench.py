"""Phase-2 benchmark-tooling tests (evaluation-side only).

Covers workspace preparation, DEV-010 rendering, derived truth, Arabic
normalization, scoring primitives, the result schema, and the no-leakage
guards. GPU models are NEVER tested here. Frozen GT is read directly by
these tests for derivation checks (evaluation side — production code in
src/ingestion must never do this; guarded below).
"""

import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ocrbench import (metrics, normalize, paddle_adapter, prompts, schema,  # noqa: E402
                      truth)
from prepare_ocr_benchmark import build_workspace  # noqa: E402

DOCS = ROOT / "data" / "dev" / "dataset_v0.1" / "documents"
GT = ROOT / "data" / "dev" / "dataset_v0.1" / "ground_truth"


def frozen(dev_id):
    return json.loads((GT / f"{dev_id}.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Workspace preparation
# --------------------------------------------------------------------------
def test_workspace_manifest_generation(tmp_path):
    manifest = build_workspace(DOCS, GT, tmp_path / "bench")
    assert len(manifest["entries"]) == 6
    by_id = {e["document_id"]: e for e in manifest["entries"]}
    assert set(by_id) == set(truth.BENCHMARK_DOCS)
    for dev_id, (role, tasks) in truth.BENCHMARK_DOCS.items():
        entry = by_id[dev_id]
        assert entry["role"] == role and entry["tasks"] == tasks
        dest = tmp_path / "bench" / entry["workspace_file"]
        assert dest.is_file()
    copies = [e for e in manifest["entries"] if e["kind"] == "copy"]
    assert {e["document_id"] for e in copies} == {"DEV-004", "DEV-008",
                                                 "DEV-009"}
    for entry in copies:  # byte-identical copies, sha-verified
        assert entry["workspace_sha256"] == entry["source_sha256"]
    render010 = by_id["DEV-010"]["render"]
    assert (render010["width_px"], render010["height_px"]) == (1191, 1684)
    derived = json.loads((tmp_path / "bench" / "ocr_benchmark_truth.json")
                         .read_text(encoding="utf-8"))
    assert set(derived) == set(truth.BENCHMARK_DOCS)


def test_dev010_render_deterministic(tmp_path):
    first = build_workspace(DOCS, GT, tmp_path / "a")
    second = build_workspace(DOCS, GT, tmp_path / "b")
    for manifest, base in ((first, "a"), (second, "b")):
        assert manifest  # built ok
    for dev_id in ("DEV-010", "DEV-002", "DEV-003"):
        fa = next(e for e in first["entries"]
                  if e["document_id"] == dev_id)["workspace_file"]
        fb = next(e for e in second["entries"]
                  if e["document_id"] == dev_id)["workspace_file"]
        assert (tmp_path / "a" / fa).read_bytes() == \
            (tmp_path / "b" / fb).read_bytes()


def test_prepare_does_not_modify_originals(tmp_path):
    before = {p.name: (p.stat().st_mtime_ns, p.stat().st_size)
              for p in DOCS.iterdir() if p.is_file()}
    build_workspace(DOCS, GT, tmp_path / "bench")
    after = {p.name: (p.stat().st_mtime_ns, p.stat().st_size)
             for p in DOCS.iterdir() if p.is_file()}
    assert before == after


# --------------------------------------------------------------------------
# Derived truth
# --------------------------------------------------------------------------
def test_derived_truth_matches_frozen_gt():
    derived = truth.derive_all(GT)
    g4 = frozen("DEV-004")
    assert derived["DEV-004"]["identifiers"] == g4["expected_identifiers"]
    assert derived["DEV-004"]["numeric_values"] == \
        g4["expected_numeric_values"]
    assert derived["DEV-004"]["text_anchors"] == g4["expected_text"]
    assert derived["DEV-004"]["language"] == "ar"
    assert derived["DEV-010"]["unrendered_context"] == \
        ["equity_2019_context"]
    assert len(derived["DEV-008"]["chart"]["series"][0]["values"]) == 10
    assert derived["DEV-009"]["budget_verdict"]["variance"] == \
        frozen("DEV-009")["expected_numeric_values"]["variance"]
    sections = derived["DEV-010"]["table_sections"]
    assert [s["title"] for s in sections] == \
        ["Assets", "Liabilities", "Equity"]
    assert derived["DEV-010"]["labels_verified"] is True


def test_scorable_numerics_excludes_context():
    derived = truth.derive_all(GT)
    scorable = truth.scorable_numerics(derived["DEV-010"])
    assert "equity_2019_context" not in scorable
    assert scorable["total_assets"] == "54959992.73"
    assert set(truth.scorable_numerics(derived["DEV-004"])) == \
        {"amount", "tax_amount", "total_amount"}


# --------------------------------------------------------------------------
# Normalization (Arabic rules)
# --------------------------------------------------------------------------
def test_arabic_normalization_rules():
    assert "ـ" not in normalize.normalize_text("مـصـرف")
    assert normalize.normalize_text("مَصْرِف") == \
        normalize.normalize_text("مصرف")
    assert normalize.normalize_text("أإآٱ") == "اااا"
    folded = normalize.normalize_text("٢٨٠٢٦٫٩٤")
    assert folded == "28026.94", folded
    # NOT folded: teh-marbuta/heh and alef-maqsura/ya stay distinct.
    assert normalize.normalize_text("مدرسة") != normalize.normalize_text("مدرسه")
    assert normalize.normalize_text("على") != normalize.normalize_text("علي")


def test_digit_error_survives_normalization():
    assert normalize.normalize_amount("28,026.94") == Decimal("28026.94")
    assert normalize.normalize_amount("٢٨٬٠٢٦٫٩٤ SAR") == Decimal("28026.94")
    assert normalize.normalize_amount("28026.94") != \
        normalize.normalize_amount("28026.95")
    with pytest.raises(ValueError):
        normalize.normalize_amount("not an amount")


# --------------------------------------------------------------------------
# Scoring primitives
# --------------------------------------------------------------------------
def test_identifier_exactness():
    assert metrics.identifier_match("INV-2023-0106", "INV-2023-0106")
    assert not metrics.identifier_match("INV20230106", "INV-2023-0106")
    assert not metrics.identifier_match("inv-2023-0106", "INV-2023-0106")
    scored = metrics.score_identifiers(
        {"transaction_id": "TX-2019-013526"},
        {"transaction_id": "TX-2019-013526"})
    assert scored["accuracy"] == 1.0 and not scored["misses"]


def test_numeric_scoring_with_miss_listing():
    expected = {"amount": "24371.25", "tax_amount": "3655.69",
                "total_amount": "28026.94"}
    ok = metrics.score_numerics(expected, expected)
    assert ok["accuracy"] == 1.0 and not ok["misses"]
    bad = metrics.score_numerics({**expected, "total_amount": "28026.95"},
                                 expected)
    assert bad["accuracy"] == pytest.approx(2 / 3)
    assert bad["misses"] == [{"field": "total_amount",
                              "expected": "28026.94", "got": "28026.95"}]
    tol = metrics.score_numerics({"total_amount": "28027.00"}, expected,
                                 rel_tol=0.005)
    assert tol["per_field"]["total_amount"] is True  # explicit opt-in only


def test_anchor_recall():
    scored = metrics.score_anchors(
        "TAX INVOICE Total due 82,209.85",
        ["TAX INVOICE", "INV-2023-0106", "Total due 82,209.85"])
    assert scored["recall"] == pytest.approx(2 / 3)
    assert scored["missing"] == ["INV-2023-0106"]


def test_cer_wer_sanity():
    assert metrics.cer("abc", "abc") == 0.0
    assert metrics.cer("abc", "abd") == pytest.approx(1 / 3)
    assert metrics.cer("abc", "") == 1.0
    assert metrics.wer("net income total", "net income total") == 0.0
    assert metrics.wer("a b c", "a x c") == pytest.approx(1 / 3)


def test_table_association_scoring():
    expected = [("Cash", "5087893.71"),
                ("Total assets", "54959992.73")]
    # Right digits on the wrong row must NOT count as correct.
    pred = [("Cash", "54959992.73"), ("Total assets", "54959992.73")]
    scored = metrics.score_table(pred, expected)
    assert scored["label_recall"] == 1.0
    assert scored["association_accuracy"] == pytest.approx(1 / 2)
    assert scored["missing_labels"] == []
    missing = metrics.score_table([("Cash", "5087893.71")], expected)
    assert missing["association_accuracy"] == pytest.approx(1 / 2)
    assert missing["missing_labels"] == ["Total assets"]


def test_chart_and_kpi_semantic_scoring():
    derived = truth.derive_all(GT)
    chart = metrics.score_chart_understanding(
        {"peak_year": "2024"},
        "Steady growth with a 2019 level shift; 2024 highest.",
        derived["DEV-008"]["chart"])
    assert chart["expected_peak_year"] == "2024"
    assert chart["peak_year_ok"] is True
    assert chart["trend_recall"] >= 0.5
    kpi = metrics.score_kpi_understanding(
        {"revenue": "23902434.34", "gross_profit": "8482081.72",
         "operating_expenses": "5346394.95", "net_income": "2795781.39",
         "budget_total": "23849529.72", "variance": "52904.62",
         "branch_revenue_BR-RUH": "13833788.57",
         "branch_revenue_BR-JED": "6858452.95",
         "branch_revenue_BR-DMM": "3210192.82",
         "top_branch": "branch_revenue_BR-RUH",
         "budget_verdict": "Budget met"},
        "Budget met with four KPI cards on top.",
        {"numeric_values": derived["DEV-009"]["numeric_values"],
         "budget_verdict": derived["DEV-009"]["budget_verdict"]})
    assert kpi["kpi_values"]["accuracy"] == 1.0
    assert kpi["top_branch_ok"] is True
    assert kpi["verdict_ok"] is True
    assert kpi["variance_ok"] is True


# --------------------------------------------------------------------------
# Result schema
# --------------------------------------------------------------------------
def test_result_schema_validation():
    valid = schema.blank_result("tesseract", "5.4.1", "cpu", "DEV-004")
    assert schema.validate_result(valid) == []
    broken = dict(valid)
    del broken["fields"]
    assert any("fields" in e for e in schema.validate_result(broken))
    bad_type = dict(valid, latency_ms="fast")
    assert schema.validate_result(bad_type)
    negative = dict(valid, latency_ms=-1.0)
    assert any("latency_ms" in e for e in schema.validate_result(negative))
    assert schema.validate_result("not a dict")


# --------------------------------------------------------------------------
# Guards: no GT/bench leakage into production; notebook sanity
# --------------------------------------------------------------------------
def test_no_benchmark_truth_import_in_production():
    import re
    # Production must never IMPORT benchmark tooling/truth. (Merely naming
    # the "ground_truth" directory is the Phase-1 skip-list protection in
    # service.py SKIP_DIRS — that exclusion is required, not leakage.)
    banned_import = re.compile(
        r"^\s*(import|from)\s+(ocrbench|prepare_ocr_benchmark)\b",
        re.MULTILINE)
    for module in ("model", "router", "parsers", "service"):
        source = (ROOT / "src" / "ingestion" / f"{module}.py").read_text(
            encoding="utf-8")
        assert not banned_import.search(source), \
            f"{module}.py imports benchmark tooling"


def test_notebook_is_minimal_and_valid():
    nb_path = ROOT / "notebooks" / "ocr_vision_benchmark.ipynb"
    assert nb_path.stat().st_size < 30 * 1024, "notebook must stay minimal"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    sources = "\n".join("".join(c.get("source", []))
                        for c in nb["cells"])
    for token in ("_get_paddle_pipeline", "kaggle_results.json",
                   "latency_ms", "PADDLE_MODEL_ID", "DEVICE"):
        assert token in sources, f"notebook missing {token}"
    assert all(not c.get("outputs") for c in nb["cells"]
               if c["cell_type"] == "code"), "outputs must be cleared"
    assert "safetensors" not in sources and ".bin" not in sources


# --------------------------------------------------------------------------
# First-experiment additions: verified labels, grid adapter, scoring
# script, export bundle, notebook sections, prompts
# --------------------------------------------------------------------------
def test_dev010_verified_pairs_match_frozen_values():
    derived = truth.derive_all(GT)
    assert derived["DEV-010"]["labels_verified"] is True
    pairs = truth.expected_table_pairs(derived["DEV-010"])
    assert len(pairs) == 11
    numerics = frozen("DEV-010")["expected_numeric_values"]
    for label, value in pairs:
        field = dict(truth.DEV010_VERIFIED_LABELS)[label]
        assert value == numerics[field], label
    with pytest.raises(ValueError):
        truth.expected_table_pairs(derived["DEV-002"])  # unverified


def test_pairs_from_table_grid():
    grid = [[["Cash", "5,087,893.71"],
             ["TOTAL ASSETS", "54,959,992.73"],
             ["Debt", "no amount here"]]]
    pairs = metrics.pairs_from_table_grid(
        grid, ["Cash", "TOTAL ASSETS", "Debt"])
    assert ("Cash", "5,087,893.71") in pairs
    assert ("TOTAL ASSETS", "54,959,992.73") in pairs
    assert all(label != "Debt" for label, _ in pairs)
    scored = metrics.score_table(
        pairs, [("Cash", "5087893.71"),
                ("TOTAL ASSETS", "54959992.73")])
    assert scored["association_accuracy"] == 1.0


def _mock_dev004_result():
    t = truth.derive_all(GT)["DEV-004"]
    return schema.blank_result("tesseract", "5.4.1", "cpu", "DEV-004") | {
        "latency_ms": 812.0,
        "text": " ".join(t["text_anchors"]),
        "fields": {"transaction_id": "TX-2019-013526", "amount": "24371.25",
                   "tax_amount": "3655.69", "total_amount": "28026.94"},
    }


def _mock_dev010_result():
    t = truth.derive_all(GT)["DEV-010"]
    grid = [[list(pair) for pair in
             truth.expected_table_pairs(t)]]
    return schema.blank_result("tesseract", "5.4.1", "cpu", "DEV-010") | {
        "latency_ms": 640.0,
        "text": "Statement of Financial Position TOTAL ASSETS 54,959,992.73",
        "fields": dict(t["numeric_values"]),
        "tables": grid,
    }


def test_scoring_script_on_synthetic_results():
    import score_ocr_results as scorer
    all_truth = truth.derive_all(GT)
    s4 = scorer.score_result(_mock_dev004_result(), all_truth)
    assert s4["schema_errors"] == []
    assert s4["identifiers"]["accuracy"] == 1.0
    assert s4["numerics_exact"]["accuracy"] == 1.0
    assert s4["anchors"]["recall"] == 1.0
    s10 = scorer.score_result(_mock_dev010_result(), all_truth)
    assert s10["table"]["association_accuracy"] == 1.0
    summary = scorer.summarize([s4, s10])
    assert summary["n_documents"] == 2
    assert summary["mean_numeric_exact_accuracy"] == 1.0
    assert summary["mean_table_association_accuracy"] == 1.0
    assert "overall_score" not in summary  # never a combined score
    assert summary["latency_ms"] == [812.0, 640.0]


def test_tolerance_diagnostic_never_replaces_exact():
    import score_ocr_results as scorer
    all_truth = truth.derive_all(GT)
    off = _mock_dev004_result()
    off["fields"]["total_amount"] = "28027.00"  # +0.06, within ±0.5%
    scored = scorer.score_result(off, all_truth)
    assert scored["numerics_exact"]["per_field"]["total_amount"] is False
    assert scored["numerics_tol_diagnostic"]["accuracy"] > \
        scored["numerics_exact"]["accuracy"]
    assert any(m["field"] == "total_amount"
               for m in scored["numerics_exact"]["misses"])


def test_result_metadata_preserves_model_identity():
    import score_ocr_results as scorer
    all_truth = truth.derive_all(GT)
    result = _mock_dev004_result()
    result.update({"model": "paddleocr-vl",
                   "model_version": "PaddleOCR-VL-1.6",
                   "device": "kaggle-T4"})
    scored = scorer.score_result(result, all_truth)
    assert (scored["model"], scored["model_version"],
            scored["device"]) == ("paddleocr-vl", "PaddleOCR-VL-1.6",
                                  "kaggle-T4")
    broken = dict(result)
    del broken["fields"]
    assert scorer.score_result(broken, all_truth)["schema_errors"]


def test_export_bundle_allowlist(tmp_path):
    from export_kaggle_bundle import build_export
    out = tmp_path / "bundle.zip"
    report = build_export(ROOT / "data" / "benchmark" / "dataset_v0.1",
                          ROOT / "notebooks", out)
    import zipfile
    names = zipfile.ZipFile(out).namelist()
    prefix = "mizaniq-ocr-benchmark-v0.1/"
    assert all(n.startswith(prefix) for n in names)
    inner = sorted(n[len(prefix):] for n in names)
    assert inner == sorted([
        "inputs/DEV-002_p1.png", "inputs/DEV-003_p1.png",
        "inputs/DEV-004.png", "inputs/DEV-008.png",
        "inputs/DEV-009.jpg", "inputs/DEV-010_p1.png",
        "manifest.json", "ocr_benchmark_truth.json",
        "ocr_vision_benchmark.ipynb", "README_KAGGLE.md",
        "PROMPTS.md", "RESULT_SCHEMA.json"])
    assert report["size_bytes"] < 10 * 1024 * 1024
    banned = (".csv", ".pt", ".bin", ".safetensors", ".env")
    assert not any(n.endswith(b) for n in names for b in banned)
    assert "ground_truth" not in "\n".join(names)


def test_notebook_experiment_sections():
    nb = json.loads((ROOT / "notebooks" / "ocr_vision_benchmark.ipynb")
                    .read_text(encoding="utf-8"))
    sources = "\n".join("".join(c.get("source", []))
                        for c in nb["cells"])
    for section in ("0. Environment", "1. Load benchmark",
                    "2. Common result", "3. Experiment A",
                    "4. Experiment B", "5. Experiment C", "6. Export"):
        assert section in sources, f"notebook missing {section}"
    for token in ("PaddleOCR-VL-1.6", "Qwen/Qwen3-VL-8B-Instruct",
                  "Qwen/Qwen3-VL-4B-Instruct", "RUN_EXPERIMENT_C",
                  "experiment_meta.json",
                  "Transcribe all visible text faithfully"):
        assert token in sources, f"notebook missing {token}"


def test_prompts_module_contract():
    assert set(prompts.PROMPTS) == {"ocr", "table", "chart", "kpi"}
    assert set(prompts.DOC_PROMPTS) == set(truth.BENCHMARK_DOCS)
    for text in prompts.PROMPTS.values():
        assert "Do not" in text  # extraction restraint in every template
    assert "trend" in prompts.PROMPTS["chart"].lower()
    assert "verdict" in prompts.PROMPTS["kpi"].lower()


# --------------------------------------------------------------------------
# PaddleOCR-VL-1.6 adapter + runnable notebook (no real inference here)
# --------------------------------------------------------------------------
def test_paddle_identity_locked_to_v16():
    assert paddle_adapter.MODEL_ID == "PaddleOCR-VL-1.6"
    assert paddle_adapter.PIPELINE_VERSION == "v1.6"
    assert paddle_adapter.CANDIDATE == "paddleocr-vl"
    ident = paddle_adapter.identity_for("3.0.0", "3.1.0", "kaggle-T4")
    assert ident["model_id"] == "PaddleOCR-VL-1.6"
    assert ident["pipeline_version"] == "v1.6"
    assert ident["paddleocr_version"] == "3.0.0"
    assert ident["device"] == "kaggle-T4"
    for key in ("doc_orientation_classify", "doc_unwarping",
                "layout_detection", "chart_recognition"):
        assert "official-default" in \
            ident["pipeline_options"][key]


def test_paddle_adapter_text_shapes():
    text, _, _, _, _ = paddle_adapter.adapt_output("plain ocr string")
    assert text == "plain ocr string"
    text, _, _, _, notes = paddle_adapter.adapt_output(
        {"rec_texts": ["Cash", "5,087,893.71"]})
    assert "Cash" in text and "5,087,893.71" in text
    assert any("rec_texts" in n for n in notes)
    text, _, _, _, _ = paddle_adapter.adapt_output(
        {"blocks": [{"block_label": "text", "text": "TOTAL ASSETS"},
                    {"block_label": "table", "text": ""}]})
    assert "TOTAL ASSETS" in text
    text, _, _, _, _ = paddle_adapter.adapt_output({"markdown": "# Title"})
    assert text == "# Title"
    weird, _, _, _, notes = paddle_adapter.adapt_output({"weird": 1})
    assert weird and any("stringified" in n for n in notes)


def test_paddle_adapter_tables_preserved():
    grid = [["Cash", "5,087,893.71"], ["Debt", "17,842,587.42"]]
    _, _, tables, _, _ = paddle_adapter.adapt_output({"tables": [grid]})
    assert tables == [grid]
    _, _, tables, _, notes = paddle_adapter.adapt_output(
        {"tables": [{"rows": grid}]})
    assert tables == [grid]
    _, _, tables, _, notes = paddle_adapter.adapt_output(
        {"tables": ["<table><tr><td>Cash</td></tr></table>"]})
    assert tables == [[["<table><tr><td>Cash</td></tr></table>"]]]
    _, _, tables, _, _ = paddle_adapter.adapt_output(
        {"blocks": [{"block_label": "table", "text": "",
                     "cells": grid}]})
    assert tables == [grid]


def test_paddle_session_created_once():
    calls = []

    class FakePipeline:
        def __init__(self, pipeline_version=None):
            calls.append(pipeline_version)

    ticks = iter([100.0, 100.5, 200.0, 200.5])
    session = paddle_adapter.PaddleSession(
        importer=lambda: FakePipeline,
        clock=lambda: next(ticks))
    first, second = session.ensure(), session.ensure()
    assert first is second
    assert calls == ["v1.6"]  # exactly one creation, locked version
    assert session.created
    assert session.init_latency_ms == pytest.approx(500.0)


def test_paddle_blocker_and_build_result_schema_valid():
    blocked = paddle_adapter.blocker_result(
        "DEV-004", "PADDLE_BLOCKER: paddleocr not installed", "kaggle-CPU")
    assert schema.validate_result(blocked) == []
    assert blocked["model_version"] == "PaddleOCR-VL-1.6"
    assert any("BLOCKER" in w for w in blocked["warnings"])
    ident = paddle_adapter.identity_for("3.0.0", "3.1.0", "kaggle-T4")
    built = paddle_adapter.build_result(
        "DEV-010", {"rec_texts": ["Cash"]}, 640.0, ident)
    assert schema.validate_result(built) == []
    assert built["model"] == "paddleocr-vl"
    assert "Cash" in built["text"]


def _notebook_paddle_section():
    nb = json.loads((ROOT / "notebooks" / "ocr_vision_benchmark.ipynb")
                    .read_text(encoding="utf-8"))
    sources = "\n".join("".join(c.get("source", []))
                        for c in nb["cells"])
    start = sources.index("## 4. Experiment B")
    end = sources.index("## 5. Experiment C")
    return sources, sources[start:end]


def test_notebook_paddle_runnable_no_placeholder():
    sources, paddle = _notebook_paddle_section()
    for token in ("from paddleocr import PaddleOCRVL",
                  "pipeline_version=PADDLE_PIPELINE_VERSION",
                  "PADDLE_PIPELINE_VERSION = 'v1.6'",
                  "PADDLE_MODEL_ID = 'PaddleOCR-VL-1.6'",
                  "_paddle_adapt", "init_latency_ms", "paddle_raw_",
                  "PADDLE_DOCS = ['DEV-004', 'DEV-010']"):
        assert token in paddle, f"paddle section missing {token}"
    for banned in ("NotImplementedError", "fill run_paddleocr_vl",
                   "fill `run_candidate`"):
        assert banned not in paddle, f"placeholder remains: {banned}"
    assert "%pip install -q paddleocr" in sources
    assert "paddlepaddle-gpu==" not in sources  # no hard-coded CUDA wheel
    assert "nvcc" in sources  # env reported before install guidance


def test_notebook_still_valid_and_minimal():
    nb_path = ROOT / "notebooks" / "ocr_vision_benchmark.ipynb"
    assert nb_path.stat().st_size < 30 * 1024
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert all(not c.get("outputs") for c in nb["cells"]
               if c["cell_type"] == "code")


def test_readme_kaggle_steps_present():
    readme = (ROOT / "notebooks" / "README.md").read_text(encoding="utf-8")
    for phrase in ("Internet", "CPU", "Tesseract", "GPU T4",
                   "CUDA-matched", "DEV-004 and DEV-010",
                   "RUN_PADDLE_OPTIONAL", "kaggle_results.json",
                   "experiment_meta.json", "data/benchmark/results/",
                   "score_ocr_results.py"):
        assert phrase in readme, f"README missing: {phrase}"
