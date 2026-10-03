"""Qwen pairs-schema contract tests (offline, no Kaggle, no GPU).

Validates the prompt/adapter fix against (a) the real saved raw outputs
from the first 4B run (legacy broken LABEL shape: parseable, no crash,
no invented associations) and (b) deterministic synthetic corrected
outputs using ONLY the exact associations already present in that saved
prose (fixtures, never written into benchmark truth).
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ocrbench import metrics, qwen_adapter, schema, truth  # noqa: E402

RUN_DIR = (ROOT / "data" / "benchmark" / "results"
           / "kaggle_qwen_qwen3-vl-4b_20261003T061255Z_"
             "omarabdallah12-mizaniq-qwen-visual")
GT_DIR = ROOT / "data" / "dev" / "dataset_v0.1" / "ground_truth"

DEV008_PAIRS = [
    ("2015", "51.9M"), ("2016", "55.4M"), ("2017", "58.3M"),
    ("2018", "61.8M"), ("2019", "72.7M"), ("2020", "86.4M"),
    ("2021", "86.4M"), ("2022", "93.8M"), ("2023", "97.8M"),
    ("2024", "105.2M"),
]
DEV009_PAIRS = [
    ("Revenue", "23.90M SAR"), ("Gross profit", "8.48M SAR"),
    ("Operating expenses", "5.35M SAR"), ("Net income", "2.80M SAR"),
    ("Riyadh", "13.83M"), ("Jeddah", "6.86M"), ("Dammam", "3.21M"),
    ("Budget", "23.85M SAR"), ("Actual", "23.90M SAR"),
    ("Variance", "+52,904.62 SAR"),
]


def _fenced(pairs):
    body = ",\n    ".join(
        f'{{"label": "{label}", "value": "{value}"}}'
        for label, value in pairs)
    return ("Chart facts.\n\nTrend description: steady growth, 2024 highest."
            f'\n\n```json\n{{"pairs": [\n    {body}\n  ]}}\n```')


def _raw(doc_id):
    return json.loads(
        (RUN_DIR / f"qwen_raw_{doc_id}.json").read_text(encoding="utf-8")
    )["generated_text"]


# 1-3. pairs schema ------------------------------------------------------------------
def test_1_pairs_schema_parsed_correctly():
    fields = qwen_adapter.extract_json_fields(_fenced(DEV008_PAIRS))
    assert list(fields.items()) == DEV008_PAIRS
    assert len(fields) == 10


def test_2_unicode_labels_preserved():
    gen = ('facts\n```json\n{"pairs": ['
           '{"label": "لوحة مؤشرات الأداء", "value": "23.90M SAR"}]}'
           '\n```')
    fields = qwen_adapter.extract_json_fields(gen)
    assert fields == {"لوحة مؤشرات الأداء": "23.90M SAR"}


def test_3_pair_order_preserved():
    fields = qwen_adapter.extract_json_fields(_fenced(DEV008_PAIRS))
    assert list(fields) == [label for label, _ in DEV008_PAIRS]


# 4-6. legacy behavior ------------------------------------------------------------------
def test_4_flat_map_legacy_shape_still_works():
    fields = qwen_adapter.extract_json_fields(
        'note\n```json\n{"Cash": "5.00", "Debt": "7.00"}\n```')
    assert fields == {"Cash": "5.00", "Debt": "7.00"}


def test_5_duplicate_label_legacy_shape_not_valid_multi_pair():
    # Real DEV-009 legacy output: 12 repeated literal keys collapse at
    # JSON parse time (irreversible) to ONE pair — never multi-pair data.
    fields = qwen_adapter.extract_json_fields(_raw("DEV-009"))
    assert fields.keys() == {"LABEL"}  # single collapsed pair, not 12
    # New schema instead keeps the FIRST duplicate explicitly + notes it.
    dup = ('x\n```json\n{"pairs": [{"label": "A", "value": "1"}, '
           '{"label": "A", "value": "2"}]}\n```')
    parsed, state = qwen_adapter._extract_json_block(dup)
    assert state == "ok"
    f2, grid, notes = qwen_adapter.pairs_to_fields(parsed)
    assert f2 == {"A": "1"} and grid == [["A", "1"]]
    assert any("duplicate" in n for n in notes)


def test_6_truncated_json_warns_without_inventing_pairs():
    parsed, state = qwen_adapter._extract_json_block(_raw("DEV-008"))
    assert state == "truncated"
    assert parsed is None
    assert qwen_adapter.extract_json_fields(_raw("DEV-008")) == {}


# 7-8. prompt + token cap ------------------------------------------------------------------
def _notebook_source():
    nb = json.loads((ROOT / "notebooks" / "ocr_vision_benchmark.ipynb")
                    .read_text(encoding="utf-8"))
    return "\n".join("".join(c.get("source", [])) for c in nb["cells"])


def test_7_new_prompt_has_no_literal_label_key():
    sources = _notebook_source()
    assert '"LABEL"' not in sources
    assert '"pairs"' in sources
    assert "never a placeholder key" in sources


def test_8_max_new_tokens_is_1024():
    sources = _notebook_source()
    assert "max_new_tokens=1024" in sources
    assert "max_new_tokens=512" not in sources


# 9-10. synthetic survival ------------------------------------------------------------------
def test_9_dev008_synthetic_pairs_survive_adapter():
    fields = qwen_adapter.extract_json_fields(_fenced(DEV008_PAIRS))
    assert len(fields) == 10
    parsed, _ = qwen_adapter._extract_json_block(_fenced(DEV008_PAIRS))
    _, grid, _ = qwen_adapter.pairs_to_fields(parsed)
    assert grid == [[label, value] for label, value in DEV008_PAIRS]
    ident = qwen_adapter.identity_for("4.53.0", "kaggle-T4",
                                      model_id=qwen_adapter.
                                      FALLBACK_MODEL_ID)
    built = qwen_adapter.build_result(
        "DEV-008", "prose", "Steady growth, 2024 highest.",
        fields, grid, 1000.0, ident)
    assert schema.validate_result(built) == []
    assert len(built["fields"]) == 10


def test_10_dev009_synthetic_pairs_survive_adapter():
    fields = qwen_adapter.extract_json_fields(_fenced(DEV009_PAIRS))
    assert len(fields) == 10
    assert "LABEL" not in fields
    parsed, _ = qwen_adapter._extract_json_block(_fenced(DEV009_PAIRS))
    _, grid, _ = qwen_adapter.pairs_to_fields(parsed)
    assert ["Revenue", "23.90M SAR"] in grid
    assert ["Variance", "+52,904.62 SAR"] in grid


# 11-12. scorer recognizes explicit pairs -----------------------------------------------------
def test_11_chart_association_scorer_recognizes_explicit_pairs():
    all_truth = truth.derive_all(GT_DIR)
    chart = all_truth["DEV-008"]["chart"]
    years = list(chart["x"])
    disp = metrics.expected_chart_display_values(chart)
    fields = qwen_adapter.extract_json_fields(_fenced(DEV008_PAIRS))
    pred = metrics.pairs_from_fields_map(fields)
    scored = metrics.score_display_association(
        pred, list(zip(years, disp)))
    assert scored["association_accuracy"] == 1.0


def test_12_kpi_association_scorer_recognizes_explicit_pairs():
    all_truth = truth.derive_all(GT_DIR)
    t9 = all_truth["DEV-009"]
    numerics = truth.scorable_numerics(t9)
    disp_map = {}
    for key, raw in numerics.items():
        if key == "variance":
            disp_map[key] = f"+{metrics.normalize_amount(raw):,.2f}"
        else:
            disp_map[key] = metrics.display_millions(raw, 2)
    exp_kpi = [(label, disp_map[field])
               for label, field in truth.DEV009_VERIFIED_LABELS
               if field in disp_map]
    fields = qwen_adapter.extract_json_fields(_fenced(DEV009_PAIRS))
    pred = metrics.pairs_from_fields_map(fields)
    scored = metrics.score_display_association(pred, exp_kpi)
    assert scored["association_accuracy"] == 1.0


# 13-15. splitters ------------------------------------------------------------------------------
def test_13_trend_description_heading_captured():
    text, visual = qwen_adapter.split_prose_visual(_raw("DEV-008"))
    assert "highest" in visual and "2024" in visual
    all_truth = truth.derive_all(GT_DIR)
    scored = metrics.score_trend_semantics(
        visual, {}, all_truth["DEV-008"]["chart"])
    assert scored["trend_recall"] > 0.0


def test_14_visual_description_heading_captured():
    text, visual = qwen_adapter.split_prose_visual(
        "Facts here.\nVisual description: a red car.")
    assert text == "Facts here."
    assert visual.startswith("Visual description:")


def test_15_step2_legacy_split_still_works():
    text, visual = qwen_adapter.split_prose_visual(
        "Step 1 facts.\nStep 2 separate: trend talk.")
    assert text == "Step 1 facts."
    assert visual.startswith("Step 2")
    text, visual = qwen_adapter.split_prose_visual("just prose")
    assert (text, visual) == ("just prose", "")


# 16-18. guards -----------------------------------------------------------------------------------
def test_16_scorer_semantics_unchanged():
    expected = [("Cash", "5087893.71"), ("Total assets", "54959992.73")]
    pred = [("Cash", "54959992.73"), ("Total assets", "54959992.73")]
    scored = metrics.score_display_association(pred, expected)
    assert scored["association_accuracy"] == 0.5  # wrong row still fails
    assert scored["label_recall"] == 1.0


def test_17_frozen_truth_untouched():
    manifest = json.loads(
        (ROOT / "data" / "dev" / "dataset_v0.1" / "manifest.json")
        .read_text(encoding="utf-8"))
    for dev_id in ("DEV-008", "DEV-009"):
        path = GT_DIR / f"{dev_id}.json"
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        assert sha == manifest["ground_truth_files"][f"{dev_id}.json"][
            "sha256"], dev_id


def test_18_no_kaggle_auto_execution():
    sources = _notebook_source()
    assert "RUN_QWEN_VISUAL = False" in sources  # manual default kept
    adapter_src = (ROOT / "src" / "ocrbench" / "qwen_adapter.py"
                   ).read_text(encoding="utf-8")
    assert "import transformers" not in adapter_src  # stdlib only
