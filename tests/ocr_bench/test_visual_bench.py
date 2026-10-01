"""Deterministic tests for the DEV-008/009 visual-reasoning benchmark.

Covers the twelve locked gates using synthetic fixtures only (no models,
no weights, no network, no frozen-truth modification):

 1. exact chart year/value association passes
 2. correct values on the wrong year fail association
 3. correct KPI value on the wrong label fails association
 4. safe thousands-separator formatting still matches
 5. digit substitution fails exact numeric scoring
 6. hallucinated numeric values are counted
 7. hallucinated labels are counted
 8. unsupported trend/cause claims can be flagged
 9. mixed Arabic/English required anchors score independently
 10. empty/partial visual results score safely (no crash)
 11. Qwen fallback model identity is preserved explicitly
 12. no overall combined score appears anywhere
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ocrbench import metrics, qwen_adapter, schema, truth  # noqa: E402
import score_ocr_results as scorer  # noqa: E402


def _chart_truth():
    return {"x": ["2015", "2016"],
            "series": [{"label": "company revenue (SAR)",
                        "values": ["51855389.81", "55430851.44"]}],
            "trend": "steady growth with 2024 highest."}


# 1. exact chart year/value association passes --------------------------------
def test_1_chart_association_exact_passes():
    disp = metrics.expected_chart_display_values(_chart_truth())
    assert disp == ["51.9M", "55.4M"]
    pred = [("2015", "51.9M"), ("2016", "55.4M")]
    scored = metrics.score_display_association(pred, list(zip(
        ["2015", "2016"], disp)))
    assert scored["association_accuracy"] == 1.0
    assert scored["label_recall"] == 1.0
    assert scored["value_accuracy"] == 1.0


# 2. correct values on the wrong year fail ------------------------------------
def test_2_chart_swapped_years_fail():
    disp = metrics.expected_chart_display_values(_chart_truth())
    pred = [("2015", "55.4M"), ("2016", "51.9M")]  # right digits, wrong rows
    scored = metrics.score_display_association(pred, list(zip(
        ["2015", "2016"], disp)))
    assert scored["label_recall"] == 1.0  # labels found ...
    assert scored["association_accuracy"] == 0.0  # ... but mis-associated


# 3. correct KPI value on the wrong label fails --------------------------------
def test_3_kpi_value_wrong_label_fails():
    expected = [("Revenue", "23.90M"), ("Net income", "2.80M")]
    pred = [("Revenue", "2.80M"), ("Net income", "2.80M")]
    scored = metrics.score_display_association(pred, expected)
    assert scored["association_accuracy"] == 0.5
    assert scored["missing_labels"] == []


# 4. safe thousands-separator formatting still matches -------------------------
def test_4_grouping_insensitive_display_match():
    expected = [("Variance", "+52,904.62")]
    for variant in ("+52,904.62 SAR", "+52904.62", "+52,904.62"):
        scored = metrics.score_display_association(
            [("Variance", variant)], expected)
        assert scored["association_accuracy"] == 1.0, variant


# 5. digit substitution fails exact numeric scoring ------------------------------
def test_5_digit_substitution_fails():
    scored = metrics.score_chart_display_text(
        "2015 51.8M 2016 55.4M", ["51.9M", "55.4M"])
    assert scored["chart_numeric_exact_accuracy"] == 0.5
    assert scored["per_value"] == {"51.9M": False, "55.4M": True}
    num = metrics.score_numeric_text(
        "les 28,026.95 SAR", {"total_amount": "28026.94"})
    assert num["accuracy"] == 0.0


# 6. hallucinated numeric values are counted --------------------------------------
def test_6_hallucinated_numerics_counted():
    expected = ["2015", "51.9M", "51855389.81"]
    scored = metrics.count_hallucinated_numerics(
        "2015 51.9M and also 99.9M", expected)
    assert scored["hallucinated_numeric_count"] == 1
    assert scored["hallucinated"] == ["99.9M"]
    clean = metrics.count_hallucinated_numerics("2015 51.9M", expected)
    assert clean["hallucinated_numeric_count"] == 0


# 7. hallucinated labels are counted -----------------------------------------------
def test_7_hallucinated_labels_counted():
    scored = metrics.count_hallucinated_labels(
        ["Revenue", "Invented KPI"], ["Revenue", "Net income"])
    assert scored["hallucinated_label_count"] == 1
    assert scored["hallucinated"] == ["Invented KPI"]


# 8. unsupported trend/cause claims can be flagged -----------------------------------
def test_8_unsupported_cause_claim_flagged():
    supported = ["growth", "2015", "2016", "highest"]
    bad = metrics.flag_unsupported_claims(
        "Steady growth overall. Revenue grew because of aliens.",
        supported)
    assert bad["unsupported_claim_count"] == 1
    assert len(bad["flagged"]) == 1
    good = metrics.flag_unsupported_claims("Steady growth overall.",
                                           supported)
    assert good["unsupported_claim_count"] == 0
    trend = metrics.score_trend_semantics(
        "Steady growth because of aliens.", {}, _chart_truth())
    assert trend["causal_language_flag"] is True
    assert "because" in trend["causal_markers_found"]


# 9. mixed Arabic/English anchors score independently ----------------------------------
def test_9_arabic_english_anchors_independent():
    anchors = ["KPI Dashboard Q4 2023", "لوحة مؤشرات الأداء",
               "Budget met"]
    split = metrics.split_anchors_by_script(anchors)
    assert split["arabic"] == ["لوحة مؤشرات الأداء"]
    assert split["english"] == ["KPI Dashboard Q4 2023", "Budget met"]
    text = "KPI Dashboard Q4 2023 Budget met"  # Arabic anchor missing
    arabic = metrics.score_anchors(text, split["arabic"])
    english = metrics.score_anchors(text, split["english"])
    assert arabic["recall"] == 0.0
    assert english["recall"] == 1.0


# 10. empty/partial visual results score safely -----------------------------------------
def test_10_empty_partial_visual_safe():
    assert metrics.score_chart_labels("", [])["chart_label_recall"] == 1.0
    assert metrics.score_chart_labels("", ["2015"])["chart_label_recall"] \
        == 0.0
    assert metrics.score_display_association([], [])[
        "association_accuracy"] == 1.0
    assert metrics.count_hallucinated_numerics("", [])[
        "hallucinated_numeric_count"] == 0
    assert metrics.score_budget_status("", "", None, None)["accuracy"] \
        == 0.0
    assert metrics.flag_unsupported_claims(
        "", [])["unsupported_claim_count"] == 0
    empty = schema.blank_result("qwen3-vl",
                                qwen_adapter.PRIMARY_MODEL_ID,
                                "cpu", "DEV-008")
    assert schema.validate_result(empty) == []


# 11. Qwen fallback model identity is preserved ------------------------------------------
def test_11_qwen_fallback_identity_preserved():
    model_id, notes = qwen_adapter.select_model_id()
    assert model_id == qwen_adapter.PRIMARY_MODEL_ID
    assert all("8B" in n for n in notes)
    model_id, notes = qwen_adapter.select_model_id("CUDA OOM on T4")
    assert model_id == qwen_adapter.FALLBACK_MODEL_ID
    assert any("CUDA OOM on T4" in n for n in notes)
    ident = qwen_adapter.identity_for("4.0.0", "kaggle-T4",
                                      model_id=model_id)
    assert ident["is_fallback"] is True
    assert ident["primary_model_id"] == qwen_adapter.PRIMARY_MODEL_ID
    built = qwen_adapter.build_result(
        "DEV-008", "Annual Company Revenue", "Steady growth.", {}, [],
        1234.0, ident)
    assert built["model_version"] == qwen_adapter.FALLBACK_MODEL_ID
    assert any("fallback" in w for w in built["warnings"])
    assert schema.validate_result(built) == []
    blocked = qwen_adapter.blocker_result(
        "DEV-009", "QWEN_BLOCKER: CUDA OOM on T4",
        attempted_model_id=qwen_adapter.PRIMARY_MODEL_ID)
    assert blocked["model_version"] == qwen_adapter.PRIMARY_MODEL_ID
    assert schema.validate_result(blocked) == []


# 12. no overall combined score appears ------------------------------------------------------
def test_12_no_combined_score():
    gt_dir = ROOT / "data" / "dev" / "dataset_v0.1" / "ground_truth"
    all_truth = truth.derive_all(gt_dir)
    t8 = all_truth["DEV-008"]
    years8 = list(t8["chart"]["x"])
    disp8 = metrics.expected_chart_display_values(t8["chart"])
    text8 = "Annual Company Revenue 2015–2024 " + " ".join(
        f"{y} {d}" for y, d in zip(years8, disp8))
    res8 = schema.blank_result("qwen3-vl", qwen_adapter.PRIMARY_MODEL_ID,
                               "kaggle-T4", "DEV-008") | {
        "latency_ms": 9000.0,
        "text": text8,
        "fields": {"peak_year": "2024",
                   "year_values": dict(zip(years8, disp8))},
        "tables": [[list(pair) for pair in zip(years8, disp8)]],
        "visual_description": ("Steady growth with a 2019 level shift; "
                               "2024 highest."),
    }
    s8 = scorer.score_result(res8, all_truth)
    for banned in ("overall_score", "combined_score", "final_score"):
        assert banned not in s8
    summary = scorer.summarize([s8])
    for banned in ("overall_score", "combined_score", "final_score"):
        assert banned not in summary
    assert summary["mean_chart_association_accuracy"] == 1.0
    assert summary["mean_chart_label_recall"] is not None
    res9 = schema.blank_result("qwen3-vl", qwen_adapter.PRIMARY_MODEL_ID,
                               "kaggle-T4", "DEV-009") | {
        "latency_ms": 9500.0,
        "text": ("KPI Dashboard Q4 2023 Revenue 23.90M SAR Variance "
                 "+52,904.62 SAR Budget met"),
        "fields": {},
        "tables": [],
        "visual_description": "Budget met.",
    }
    s9 = scorer.score_result(res9, all_truth)
    assert s9["budget_status"]["budget_status_accuracy"] == 1.0
    assert s9["kpi_labels"]["kpi_label_recall"] > 0.0
    assert "overall_score" not in scorer.summarize([s8, s9])


# --------------------------------------------------------------------------
# Notebook: visual round runnable independently (Visual A / B / export)
# --------------------------------------------------------------------------
def _notebook_sources():
    nb_path = ROOT / "notebooks" / "ocr_vision_benchmark.ipynb"
    nb = __import__("json").loads(nb_path.read_text(encoding="utf-8"))
    return nb, "\n".join("".join(c.get("source", []))
                         for c in nb["cells"])


def test_13_visual_switches_default_false():
    _, sources = _notebook_sources()
    assert "RUN_PADDLE_VISUAL = False" in sources
    assert "RUN_QWEN_VISUAL = False" in sources


def test_14_visual_sections_and_qwen_contract():
    nb, sources = _notebook_sources()
    for section in ("Visual A", "Visual B", "Visual export"):
        assert section in sources, f"notebook missing {section}"
    for token in ("QWEN_FALLBACK_ID", "Qwen/Qwen3-VL-4B-Instruct",
                  "QWEN_MODEL_ID = 'Qwen/Qwen3-VL-8B-Instruct'",
                  "ALLOW_QWEN_FALLBACK", "QWEN_BLOCKER", "qwen_raw_",
                  "meta['qwen']", "chart_recognition"):
        assert token in sources, f"notebook missing {token}"
    assert "never silent" in sources  # fallback substitution forbidden
    assert nb["nbformat"] == 4
    assert all(not c.get("outputs") for c in nb["cells"]
               if c["cell_type"] == "code")
