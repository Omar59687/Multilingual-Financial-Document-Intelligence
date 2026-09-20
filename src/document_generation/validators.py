"""Validators for the DEV document set (dataset_v0.1). Fail-closed.

``validate_all(canonical_dir, dev_dir)`` checks every mapping on disk:
documents exist with the specified formats, source IDs resolve in canonical
truth, rendered numbers recompute from canonical rows, scanned fixtures are
really raster, and ground truth stays out of the document directory.
"""

import csv
import hashlib
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from .generator import BUILDERS
from .selection import load_canonical

EXPECTED_FILES = {
    "DEV-001": "DEV-001_income_statement_2023_EN.pdf",
    "DEV-002": "DEV-002_branch_expenses_2022_AR.pdf",
    "DEV-003": "DEV-003_supplier_invoice_2023_MIX.pdf",
    "DEV-004": "DEV-004_scanned_receipt_2019_AR.png",
    "DEV-005": "DEV-005_management_commentary_2023_MIX.docx",
    "DEV-006": "DEV-006_monthly_transactions_2023_EN.xlsx",
    "DEV-007": "DEV-007_transactions_extract_2024_MIX.csv",
    "DEV-008": "DEV-008_revenue_trend_2015_2024_EN.png",
    "DEV-009": "DEV-009_kpi_dashboard_Q4-2023_MIX.jpg",
    "DEV-010": "DEV-010_balance_sheet_2020_EN_scanned.pdf",
}

GT_KEYS = ("dev_id", "filename", "format", "language", "native_scanned",
           "source_tables", "source_record_ids", "source_period",
           "expected_metadata", "expected_text", "expected_fields",
           "expected_numeric_values", "expected_identifiers",
           "expected_tables", "expected_chart_data",
           "expected_visual_elements", "expected_page_locations",
           "expected_question_targets", "generation_parameters")


class DevValidationError(Exception):
    """Raised on the first DEV-set invariant violation."""


def _fail(msg: str) -> None:
    raise DevValidationError(msg)


def _pdf_text(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _ar_word_present(text: str, logical_word: str) -> bool:
    """Check a single Arabic word in pypdf-extracted output.

    Rendering uses reshape+bidi (visual order in the PDF). pypdf >= 6
    reorders RTL runs back toward logical order on extraction, so anchors
    are compared with reshape-only (logical-order) presentation forms.
    Verified empirically against generated PDFs.
    """
    import arabic_reshaper
    return arabic_reshaper.reshape(logical_word) in text


def _pdf_page_count(path: Path) -> int:
    from pypdf import PdfReader
    return len(PdfReader(str(path)).pages)


def _pdf_has_image(path: Path) -> bool:
    from pypdf import PdfReader
    for page in PdfReader(str(path)).pages:
        try:
            if list(page.images):
                return True
        except Exception:
            pass
        resources = page.get("/Resources")
        if resources and resources.get("/XObject"):
            return True
    return False


def validate_all(canonical_dir, dev_dir) -> dict:
    canonical_dir, dev_dir = Path(canonical_dir), Path(dev_dir)
    docs_dir, gt_dir = dev_dir / "documents", dev_dir / "ground_truth"
    tables = load_canonical(canonical_dir)

    # -- 1. ten IDs, ten files, formats ------------------------------------
    if {d for d, _ in BUILDERS} != set(EXPECTED_FILES):
        _fail("builder registry does not cover exactly DEV-001..DEV-010")
    doc_files = sorted(p.name for p in docs_dir.iterdir() if p.is_file())
    if doc_files != sorted(EXPECTED_FILES.values()):
        _fail(f"documents/ must hold exactly the 10 DEV files; got {doc_files}")
    gt_files = sorted(p.name for p in gt_dir.iterdir() if p.is_file())
    if gt_files != sorted(f"{d}.json" for d in EXPECTED_FILES):
        _fail(f"ground_truth/ must hold exactly 10 JSONs; got {gt_files}")
    records = {}
    for dev_id in EXPECTED_FILES:
        rec = json.loads((gt_dir / f"{dev_id}.json").read_text(
            encoding="utf-8"))
        if tuple(sorted(rec.keys())) != tuple(sorted(GT_KEYS)):
            missing = set(GT_KEYS) - set(rec.keys())
            extra = set(rec.keys()) - set(GT_KEYS)
            _fail(f"{dev_id}.json schema mismatch "
                  f"(missing={missing}, extra={extra})")
        if rec["dev_id"] != dev_id or rec["filename"] != EXPECTED_FILES[dev_id]:
            _fail(f"{dev_id}.json identity mismatch")
        records[dev_id] = rec

    monthly = tables["monthly_financials"]
    tx_by_id = {t["transaction_id"]: t for t in tables["transactions"]}
    inv_by_id = {v["invoice_id"]: v for v in tables["invoices"]}

    def num(dev_id, key):
        return Decimal(records[dev_id]["expected_numeric_values"][key])

    # -- 2. DEV-001: company 2023 aggregates recompute ----------------------
    rows23 = [r for r in monthly if r["period"][:4] == "2023"]
    if len(rows23) != 36:
        _fail("DEV-001: expected 36 canonical 2023 rows")
    for metric in ("revenue", "cogs", "gross_profit", "operating_expenses",
                   "operating_profit", "other_income", "finance_costs",
                   "tax_expense", "net_income"):
        truth = sum((r[metric] for r in rows23), Decimal("0.00"))
        if num("DEV-001", f"fy2023_{metric}") != truth:
            _fail(f"DEV-001 fy2023_{metric}: ground truth != canonical sum")
    t1 = _pdf_text(docs_dir / EXPECTED_FILES["DEV-001"])
    for needle in ("Noor Retail", "EVT-2023-001",
                   f"{num('DEV-001', 'fy2023_net_income'):,.2f}"):
        if needle not in t1:
            _fail(f"DEV-001 PDF text missing {needle!r} (not selectable?)")
    if _pdf_page_count(docs_dir / EXPECTED_FILES["DEV-001"]) != 1:
        _fail("DEV-001 must be a stable 1-page layout")

    # -- 3. DEV-002: Jeddah 2022 opex + categories --------------------------
    rows22j = [r for r in monthly
               if r["period"][:4] == "2022" and r["branch_id"] == "BR-JED"]
    truth_opex = sum((r["operating_expenses"] for r in rows22j),
                     Decimal("0.00"))
    if num("DEV-002", "opex_2022_total") != truth_opex:
        _fail("DEV-002 headline total != canonical Jeddah 2022 opex")
    cat = defaultdict(Decimal)
    for t in tables["transactions"]:
        if t["date"][:4] == "2022" and t["branch_id"] == "BR-JED" \
                and t["transaction_type"] == "expense":
            cat[t["category"]] += t["total_amount"]
    for c, total in cat.items():
        if num("DEV-002", f"extract_{c}") != total:
            _fail(f"DEV-002 extract_{c} != canonical subset sum")
    t2 = _pdf_text(docs_dir / EXPECTED_FILES["DEV-002"])
    for needle in ("EVT-2022-001",
                   f"{truth_opex:,.2f}",
                   f"2022 : {truth_opex:,.2f} SAR",
                   "Sample size: 372 transactions"):
        if needle not in t2:
            _fail(f"DEV-002 PDF text missing {needle!r} (not selectable?)")
    for word in ("تقرير", "المصروفات", "جدة", "الرواتب"):
        if not _ar_word_present(t2, word):
            _fail(f"DEV-002 PDF text missing Arabic word {word!r}")

    # -- 4. DEV-003: exact invoice row --------------------------------------
    inv = inv_by_id.get("INV-2023-0106")
    if inv is None:
        _fail("DEV-003 source invoice INV-2023-0106 missing in canonical")
    rec3 = records["DEV-003"]["expected_numeric_values"]
    if Decimal(rec3["subtotal"]) != inv["subtotal"] or \
            Decimal(rec3["vat"]) != inv["vat"] or \
            Decimal(rec3["total"]) != inv["total"]:
        _fail("DEV-003 ground truth != canonical invoice row")
    if inv["subtotal"] + inv["vat"] != inv["total"]:
        _fail("DEV-003 invoice arithmetic violated")
    if inv["related_transaction_id"] not in tx_by_id:
        _fail("DEV-003 linked transaction missing")
    t3 = _pdf_text(docs_dir / EXPECTED_FILES["DEV-003"])
    for needle in ("INV-2023-0106", f"{inv['total']:,.2f}",
                   inv["invoice_date"]):
        if needle not in t3:
            _fail(f"DEV-003 PDF missing {needle!r}")
    if not _ar_word_present(t3, "فاتورة"):
        _fail("DEV-003 PDF missing Arabic word 'فاتورة'")

    # -- 5. DEV-004: receipt row + raster -----------------------------------
    tx = tx_by_id.get("TX-2019-013526")
    if tx is None:
        _fail("DEV-004 source transaction missing in canonical")
    rec4 = records["DEV-004"]["expected_numeric_values"]
    if Decimal(rec4["total_amount"]) != tx["total_amount"] or \
            Decimal(rec4["amount"]) != tx["amount"] or \
            Decimal(rec4["tax_amount"]) != tx["tax_amount"]:
        _fail("DEV-004 ground truth != canonical transaction row")
    if tx["amount"] + tx["tax_amount"] != tx["total_amount"]:
        _fail("DEV-004 transaction arithmetic violated")
    from PIL import Image
    with Image.open(docs_dir / EXPECTED_FILES["DEV-004"]) as im:
        im.load()
        if im.format != "PNG" or im.mode != "RGB":
            _fail("DEV-004 must be an RGB PNG raster (no text layer)")

    # -- 6. DEV-005: events + numbers ---------------------------------------
    evt_ids = {e["event_id"] for e in tables["business_events"]}
    for eid in records["DEV-005"]["expected_identifiers"]["event_ids"]:
        if eid not in evt_ids:
            _fail(f"DEV-005 event {eid} missing in canonical")
    dip = sum((r["revenue"] for r in monthly
               if r["branch_id"] == "BR-JED"
               and r["period"][:7] in ("2023-09", "2023-10")), Decimal("0.00"))
    if num("DEV-005", "jeddah_sep_oct_2023") != dip:
        _fail("DEV-005 Jeddah dip != canonical recomputation")
    from docx import Document as DocxDocument
    paras = [p.text for p in
             DocxDocument(str(docs_dir / EXPECTED_FILES["DEV-005"])).paragraphs]
    blob = "\n".join(paras)
    for needle in ("EVT-2023-001", "تعليق الإدارة",
                   f"{num('DEV-005', 'jeddah_sep_oct_2023'):,.2f}"):
        if needle not in blob:
            _fail(f"DEV-005 DOCX missing {needle!r}")

    # -- 7. DEV-006: XLSX subset == canonical rows --------------------------
    from openpyxl import load_workbook
    wb = load_workbook(docs_dir / EXPECTED_FILES["DEV-006"], read_only=True,
                       data_only=True)
    ws = wb["transactions_2023"]
    xrows = list(ws.iter_rows(min_row=2, values_only=True))
    canon23 = sorted(
        (t for t in tables["transactions"] if t["date"][:4] == "2023"),
        key=lambda t: (t["date"], t["transaction_id"]))
    if len(xrows) != len(canon23) != 4020:
        _fail("DEV-006 must hold exactly the 4,020 canonical 2023 rows")
    for xr, ct in zip(xrows, canon23):
        if xr[0] != ct["transaction_id"] or xr[1] != ct["date"]:
            _fail("DEV-006 row order/IDs != canonical selection")
        for idx, col in ((8, "amount"), (9, "tax_amount"),
                         (10, "total_amount")):
            if Decimal(str(xr[idx])) != ct[col]:
                _fail(f"DEV-006 {ct['transaction_id']}.{col} mismatch")
    proc = sum((Decimal(str(r[10])) for r in xrows
                if r[3] == "DEP-PROC" and r[1][:7] in ("2023-10", "2023-11",
                                                      "2023-12")),
               Decimal("0.00"))
    if proc != num("DEV-006", "procurement_q4_2023_total"):
        _fail("DEV-006 Q4 procurement aggregate mismatch")

    # -- 8. DEV-007: CSV subset verbatim ------------------------------------
    with open(docs_dir / EXPECTED_FILES["DEV-007"], newline="",
              encoding="utf-8") as f:
        creader = list(csv.DictReader(f))
    canon_log = sorted(
        (t for t in tables["transactions"]
         if t["department_id"] == "DEP-LOG"
         and t["date"][:7] in ("2024-01", "2024-02", "2024-03", "2024-04",
                               "2024-05", "2024-06")),
        key=lambda t: (t["date"], t["transaction_id"]))
    if len(creader) != len(canon_log) != 346:
        _fail("DEV-007 must hold exactly the 346 canonical H1-2024 LOG rows")
    for cr, ct in zip(creader, canon_log):
        if cr["transaction_id"] != ct["transaction_id"] or \
                Decimal(cr["total_amount"]) != ct["total_amount"] or \
                cr["description_ar"] != ct["description_ar"]:
            _fail(f"DEV-007 {ct['transaction_id']} cell mismatch")
    if sum((Decimal(c["total_amount"]) for c in creader),
           Decimal("0.00")) != num("DEV-007", "total_amount_sum"):
        _fail("DEV-007 total != canonical subset sum")

    # -- 9. DEV-008: chart points == annual truth ----------------------------
    chart = records["DEV-008"]["expected_chart_data"]["series"][0]["values"]
    truth_annual = []
    for year in range(2015, 2025):
        truth_annual.append(sum(
            (r["revenue"] for r in monthly if r["period"][:4] == str(year)),
            Decimal("0.00")))
    if [Decimal(v) for v in chart] != truth_annual:
        _fail("DEV-008 chart points != canonical annual sums")
    with Image.open(docs_dir / EXPECTED_FILES["DEV-008"]) as im:
        im.load()
        if im.format != "PNG" or im.size[0] < 800:
            _fail("DEV-008 must be a readable PNG chart image")

    # -- 10. DEV-009: KPI recomputation -------------------------------------
    q4 = [r for r in monthly if r["period"][:7] in ("2023-10", "2023-11",
                                                    "2023-12")]
    for metric in ("revenue", "gross_profit", "operating_expenses",
                   "net_income"):
        if num("DEV-009", metric) != sum((r[metric] for r in q4),
                                         Decimal("0.00")):
            _fail(f"DEV-009 {metric} != canonical Q4 sum")
    bud_total = sum((b["budget_amount"] for b in tables["budgets"]
                     if b["period"] == "2023-Q4" and b["metric"] == "revenue"
                     and b["branch_id"]), Decimal("0.00"))
    if num("DEV-009", "budget_total") != bud_total:
        _fail("DEV-009 budget_total != canonical budget rows")
    if num("DEV-009", "variance") != num("DEV-009", "revenue") - bud_total:
        _fail("DEV-009 variance arithmetic violated")
    with Image.open(docs_dir / EXPECTED_FILES["DEV-009"]) as im:
        im.load()
        if im.format != "JPEG":
            _fail("DEV-009 must be a JPG dashboard image")

    # -- 11. DEV-010: balance row + raster-only PDF --------------------------
    row20 = next(r for r in tables["annual_balance_sheet"]
                 if r["year"] == "2020")
    for key in ("cash", "accounts_receivable", "inventory",
                "other_current_assets", "property_and_equipment",
                "total_assets", "accounts_payable", "debt",
                "other_liabilities", "total_liabilities", "equity"):
        if num("DEV-010", key) != row20[key]:
            _fail(f"DEV-010 {key} != canonical 2020 balance row")
    if row20["total_assets"] != row20["total_liabilities"] + row20["equity"]:
        _fail("DEV-010 balance-sheet equation violated")
    p10 = docs_dir / EXPECTED_FILES["DEV-010"]
    if len(_pdf_text(p10).strip()) >= 50:
        _fail("DEV-010 must be raster-only (found selectable body text)")
    if _pdf_page_count(p10) != 1 or not _pdf_has_image(p10):
        _fail("DEV-010 must be a single raster-page PDF")

    # -- 12. manifest + leakage ----------------------------------------------
    manifest_path = dev_dir / "manifest.json"
    if not manifest_path.exists():
        _fail("missing dev manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("canonical_dataset") != "dataset_v0.1":
        _fail("dev manifest must point at canonical dataset_v0.1")
    for filename, meta in manifest.get("documents", {}).items():
        digest = hashlib.sha256(
            (docs_dir / filename).read_bytes()).hexdigest()
        if meta.get("sha256") != digest:
            _fail(f"dev manifest sha256 mismatch for {filename}")
    if set(manifest["documents"]) != set(EXPECTED_FILES.values()):
        _fail("dev manifest must list exactly the 10 DEV files")
    # ground truth must not leak into the corpus directory
    for p in docs_dir.iterdir():
        if p.suffix == ".json":
            _fail("ground-truth JSON inside documents/: leakage")

    return {dev_id: EXPECTED_FILES[dev_id] for dev_id in EXPECTED_FILES}
