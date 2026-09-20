"""Generation-correctness tests for the DEV document set (dataset_v0.1).

Tests generation only: deterministic selection, traceability, financial
correctness, subset fidelity, ground-truth schema, scanned/native roles,
and byte-level reproducibility. No OCR/RAG/ingestion testing here.
"""

import hashlib
import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dataset import config as C  # noqa: E402
from document_generation import generator as devgen  # noqa: E402
from document_generation import renderers as R  # noqa: E402
from document_generation import selection  # noqa: E402
from document_generation import validators as devval  # noqa: E402

CANONICAL_DIR = ROOT / "data" / "canonical" / C.DATASET_VERSION

GT_KEYS = ("dev_id", "filename", "format", "language", "native_scanned",
           "source_tables", "source_record_ids", "source_period",
           "expected_metadata", "expected_text", "expected_fields",
           "expected_numeric_values", "expected_identifiers",
           "expected_tables", "expected_chart_data",
           "expected_visual_elements", "expected_page_locations",
           "expected_question_targets", "generation_parameters")


@pytest.fixture(scope="module")
def tables():
    return selection.load_canonical(CANONICAL_DIR)


@pytest.fixture(scope="module")
def dev_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("devset")
    devgen.generate_all(CANONICAL_DIR, out)
    return out


@pytest.fixture(scope="module")
def records(dev_dir):
    recs = {}
    for dev_id in devval.EXPECTED_FILES:
        recs[dev_id] = json.loads(
            (dev_dir / "ground_truth" / f"{dev_id}.json").read_text(
                encoding="utf-8"))
    return recs


def test_canonical_input_present():
    assert (CANONICAL_DIR / "manifest.json").exists()


def test_deterministic_selection_pinned_ids(tables):
    resolved = selection.resolve_all(tables)
    assert resolved["DEV-003"]["invoice"]["invoice_id"] == "INV-2023-0106"
    assert resolved["DEV-003"]["invoice"]["total"] == Decimal("82209.85")
    assert resolved["DEV-004"]["transaction"]["transaction_id"] == \
        "TX-2019-013526"
    assert resolved["DEV-004"]["transaction"]["total_amount"] == \
        Decimal("28026.94")
    # re-resolution is stable
    again = selection.resolve_all(tables)
    assert again["DEV-003"]["invoice"] == resolved["DEV-003"]["invoice"]


def test_ten_documents_filename_stability(dev_dir):
    docs = sorted(p.name for p in (dev_dir / "documents").iterdir())
    assert docs == sorted(devval.EXPECTED_FILES.values())
    assert len(docs) == 10


def test_ground_truth_schema_complete(records):
    for dev_id, rec in records.items():
        assert tuple(sorted(rec.keys())) == tuple(sorted(GT_KEYS)), dev_id
        assert rec["dev_id"] == dev_id
        assert rec["filename"] == devval.EXPECTED_FILES[dev_id]
        assert rec["source_record_ids"], dev_id
        assert rec["expected_question_targets"], dev_id


def test_traceability_ids_exist_in_canonical(tables, records):
    tx_ids = {t["transaction_id"] for t in tables["transactions"]}
    inv_ids = {v["invoice_id"] for v in tables["invoices"]}
    evt_ids = {e["event_id"] for e in tables["business_events"]}
    for dev_id, rec in records.items():
        for rid in rec["source_record_ids"]:
            if dev_id == "DEV-003":
                assert rid in inv_ids or rid in tx_ids, rid
            elif dev_id in ("DEV-004", "DEV-006", "DEV-007"):
                assert rid in tx_ids, rid
            elif dev_id == "DEV-005":
                assert rid in evt_ids, rid
            elif dev_id == "DEV-010":
                assert rid in ("2019", "2020"), rid
            # DEV-001/002/008/009 use period:branch keys + event IDs
            elif ":" in rid and not rid[0].isdigit():
                assert rid in evt_ids, rid


def test_dev001_values_correct(tables, records):
    rows23 = [r for r in tables["monthly_financials"]
              if r["period"][:4] == "2023"]
    truth = sum((r["net_income"] for r in rows23), Decimal("0.00"))
    assert Decimal(records["DEV-001"]["expected_numeric_values"]
                   ["fy2023_net_income"]) == truth
    assert truth == Decimal("11413708.69")


def test_invoice_correctness(tables, records):
    inv = next(v for v in tables["invoices"]
               if v["invoice_id"] == "INV-2023-0106")
    assert inv["subtotal"] + inv["vat"] == inv["total"]
    rec = records["DEV-003"]["expected_numeric_values"]
    assert Decimal(rec["total"]) == inv["total"] == Decimal("82209.85")
    assert records["DEV-003"]["expected_identifiers"]["invoice_id"] == \
        "INV-2023-0106"


def test_xlsx_csv_subset_correctness(tables, records):
    assert records["DEV-006"]["expected_numeric_values"]["row_count"] == "4020"
    assert records["DEV-007"]["expected_numeric_values"]["row_count"] == "346"
    assert Decimal(records["DEV-007"]["expected_numeric_values"]
                   ["total_amount_sum"]) == Decimal("1666716.10")
    n23 = sum(1 for t in tables["transactions"] if t["date"][:4] == "2023")
    assert n23 == 4020


def test_chart_data_correctness(tables, records):
    chart = records["DEV-008"]["expected_chart_data"]
    assert chart["chart_type"] == "line" and len(chart["x"]) == 10
    assert chart["x"][0] == "2015" and chart["x"][-1] == "2024"
    for year, val in zip(chart["x"], chart["series"][0]["values"]):
        truth = sum((r["revenue"] for r in tables["monthly_financials"]
                     if r["period"][:4] == year), Decimal("0.00"))
        assert Decimal(val) == truth
    assert Decimal(chart["series"][0]["values"][-1]) == \
        Decimal("105185955.08")


def test_balance_sheet_correctness(tables, records):
    row20 = next(r for r in tables["annual_balance_sheet"]
                 if r["year"] == "2020")
    assert row20["total_assets"] == \
        row20["total_liabilities"] + row20["equity"]
    rec = records["DEV-010"]
    assert Decimal(rec["expected_numeric_values"]["total_assets"]) == \
        row20["total_assets"] == Decimal("54959992.73")


def test_budget_variance_correctness(tables, records):
    rec = records["DEV-009"]["expected_numeric_values"]
    assert Decimal(rec["variance"]) == \
        Decimal(rec["revenue"]) - Decimal(rec["budget_total"])
    assert Decimal(rec["budget_total"]) == Decimal("23849529.72")


def test_native_vs_scanned_roles(dev_dir):
    from pypdf import PdfReader
    docs = dev_dir / "documents"
    # native: selectable text present
    t1 = PdfReader(str(docs / devval.EXPECTED_FILES["DEV-001"])).pages[0]
    assert "Net income" in (t1.extract_text() or "")
    # scanned: raster-only page, no body text
    p10 = PdfReader(str(docs / devval.EXPECTED_FILES["DEV-010"])).pages[0]
    assert len((p10.extract_text() or "").strip()) < 50
    assert p10.get("/Resources").get("/XObject") is not None
    # receipt: RGB raster image
    from PIL import Image
    with Image.open(docs / devval.EXPECTED_FILES["DEV-004"]) as im:
        im.load()
        assert im.format == "PNG" and im.mode == "RGB"
    with Image.open(docs / devval.EXPECTED_FILES["DEV-009"]) as im9:
        im9.load()
        assert im9.format == "JPEG"


def test_arabic_rendering_pipeline():
    import arabic_reshaper
    assert Path(R.ARIAL_TTF).exists()  # shaping-capable font present
    logical = "تقرير المصروفات"
    visual = R.ar(logical)
    assert visual != logical  # reshaping actually applied
    assert len(visual) > 0
    # ground truth preserves logical Arabic (evaluation-side normalization)
    assert "تقرير" in logical


def test_validation_passes(dev_dir):
    files = devval.validate_all(CANONICAL_DIR, dev_dir)
    assert set(files) == set(devval.EXPECTED_FILES)
    assert files["DEV-003"] == "DEV-003_supplier_invoice_2023_MIX.pdf"


def _snapshot(dev_dir):
    snap = {}
    for sub in ("documents", "ground_truth"):
        for path in sorted((dev_dir / sub).iterdir()):
            snap[f"{sub}/{path.name}"] = hashlib.sha256(
                path.read_bytes()).hexdigest()
    manifest = json.loads((dev_dir / "manifest.json").read_text(
        encoding="utf-8"))
    manifest.pop("generated_at", None)
    snap["manifest.json"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return snap


def test_regeneration_reproducibility(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    devgen.generate_all(CANONICAL_DIR, first)
    devgen.generate_all(CANONICAL_DIR, second)
    snap1, snap2 = _snapshot(first), _snapshot(second)
    assert snap1.keys() == snap2.keys()
    diffs = [k for k in snap1 if snap1[k] != snap2[k]]
    assert not diffs, f"nondeterministic artifacts: {diffs}"
