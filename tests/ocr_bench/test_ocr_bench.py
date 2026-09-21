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

from ocrbench import metrics, normalize, schema, truth  # noqa: E402
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
    assert derived["DEV-010"]["labels_verified"] is False


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
    assert nb_path.stat().st_size < 20 * 1024, "notebook must stay minimal"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    sources = "\n".join("".join(c.get("source", []))
                        for c in nb["cells"])
    for token in ("run_candidate", "kaggle_results.json", "latency_ms",
                  "MODEL_VERSION", "DEVICE"):
        assert token in sources, f"notebook missing {token}"
    assert all(not c.get("outputs") for c in nb["cells"]
               if c["cell_type"] == "code"), "outputs must be cleared"
    assert "safetensors" not in sources and ".bin" not in sources
