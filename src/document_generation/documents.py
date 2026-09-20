"""Content builders for the 10 DEV documents (dataset_v0.1).

Each ``build_devXXX(selection, docs_dir)`` renders one artifact from canonical
rows and returns ``(filename, ground_truth_dict)``. No numbers are invented:
every rendered value comes from the selection context, and the ground-truth
record recomputes the same values independently where practical.

Conventions:
- ``fmt_money`` (thousands separators) for human-facing text; ``plain`` 2dp
  strings inside ground-truth ``expected_numeric_values``.
- Arabic for reportlab/Pillow/matplotlib goes through ``renderers.ar``;
  DOCX Arabic is written logically (Word shapes natively, paragraphs flagged
  RTL).
"""

from decimal import Decimal

from . import renderers as R
from .selection import AR_MONTHS

COMPANY_EN = "Noor Retail & Distribution Co."
COMPANY_AR = "شركة نور للتجزئة والتوزيع"

QTYPE = {
    "lookup": 1, "delta": 2, "total": 3, "margin": 4, "branch_cmp": 5,
    "period_cmp": 6, "reconcile": 7, "explain": 8, "ar_ar": 10,
    "ar_en": 11, "en_ar": 12, "mixed": 13, "identifier": 14,
    "table_cell": 15, "table_agg": 16, "chart": 17, "chart_text": 18,
}


def _gt(dev_id, filename, fmt, lang, native_scanned, source_tables,
        source_record_ids, source_period, **kw):
    record = {
        "dev_id": dev_id, "filename": filename, "format": fmt,
        "language": lang, "native_scanned": native_scanned,
        "source_tables": source_tables,
        "source_record_ids": source_record_ids,
        "source_period": source_period,
        "expected_metadata": None, "expected_text": None,
        "expected_fields": None, "expected_numeric_values": None,
        "expected_identifiers": None, "expected_tables": None,
        "expected_chart_data": None, "expected_visual_elements": None,
        "expected_page_locations": None, "expected_question_targets": None,
        "generation_parameters": None,
    }
    record.update(kw)
    return record


def _qt(question, qtype, answer, sources):
    return {"question": question, "question_type": qtype, "answer": answer,
            "sources": sources}


def _yoy(curr: Decimal, prev: Decimal) -> str:
    pct = (curr - prev) / prev * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


# ---------------------------------------------------------------------------
# DEV-001 — English native income-statement PDF
# ---------------------------------------------------------------------------
PNL_LABELS = (("Revenue", "revenue"), ("Cost of sales (COGS)", "cogs"),
              ("Gross profit", "gross_profit"),
              ("Operating expenses", "operating_expenses"),
              ("Operating profit", "operating_profit"),
              ("Other income", "other_income"),
              ("Finance costs", "finance_costs"),
              ("Tax expense", "tax_expense"),
              ("Net income", "net_income"))


def build_dev001(sel, docs_dir):
    R.ensure_pdf_fonts()
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

    t23, t22 = sel["totals23"], sel["totals22"]
    styles = getSampleStyleSheet()
    title = ParagraphStyle("Title2", parent=styles["Title"],
                           fontName="Arial-Bold", fontSize=16)
    normal = ParagraphStyle("Body", parent=styles["Normal"],
                            fontName="Arial", fontSize=10, leading=14)
    cell = ParagraphStyle("Cell", parent=normal, fontSize=9, leading=12)
    cell_r = ParagraphStyle("CellR", parent=cell, alignment=2)
    head = ParagraphStyle("Head", parent=cell, fontName="Arial-Bold",
                          textColor=colors.white, alignment=1)

    header = ["Line item", "FY 2023 (SAR)", "FY 2022 (SAR)", "Change"]
    data = [[Paragraph(f"<font name='Arial-Bold'>{h}</font>", head)
             for h in header]]
    for label, metric in PNL_LABELS:
        data.append([
            Paragraph(label, cell),
            Paragraph(R.fmt_money(t23[metric]), cell_r),
            Paragraph(R.fmt_money(t22[metric]), cell_r),
            Paragraph(_yoy(t23[metric], t22[metric]), cell_r),
        ])
    table = Table(data, colWidths=[62 * mm, 38 * mm, 38 * mm, 28 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))

    br = sel["branch_revenue"]
    dip23 = sum(r["revenue"] for r in sel["rows23"]
                if r["branch_id"] == "BR-JED" and r["period"][:7]
                in ("2023-09", "2023-10"))
    dip22 = sum(r["revenue"] for r in sel["rows22"]
                if r["branch_id"] == "BR-JED" and r["period"][:7]
                in ("2022-09", "2022-10"))
    flow = [
        Paragraph(COMPANY_EN, title),
        Paragraph("Financial Performance Report &mdash; Fiscal Year 2023 "
                  "(all figures in SAR)", normal),
        Spacer(1, 6),
        table,
        Spacer(1, 8),
        Paragraph(
            f"Revenue by branch in 2023: Riyadh SAR {R.fmt_money(br['BR-RUH'])}; "
            f"Jeddah SAR {R.fmt_money(br['BR-JED'])}; Dammam SAR "
            f"{R.fmt_money(br['BR-DMM'])}. Riyadh remained the largest "
            "contributor to company revenue.", normal),
        Spacer(1, 4),
        Paragraph(
            f"In September&ndash;October 2023 part of the Jeddah branch was "
            f"closed for refit works (event EVT-2023-001). Jeddah revenue for "
            f"those two months was SAR {R.fmt_money(dip23)}, compared with SAR "
            f"{R.fmt_money(dip22)} a year earlier, before recovering on full "
            "reopening.", normal),
        Spacer(1, 4),
        Paragraph("Source: canonical monthly financials &middot; dataset_v0.1",
                  normal),
    ]
    filename = "DEV-001_income_statement_2023_EN.pdf"
    R.build_pdf(docs_dir / filename, flow,
                "Noor Retail - Financial Performance Report FY2023",
                pagesize=A4)

    numeric = {f"fy2023_{m}": R.plain(t23[m]) for _, m in PNL_LABELS}
    numeric.update({f"fy2022_{m}": R.plain(t22[m]) for _, m in PNL_LABELS})
    numeric.update({f"branch_revenue_2023_{b}": R.plain(v)
                    for b, v in br.items()})
    numeric.update({"jeddah_sep_oct_2023": R.plain(dip23),
                    "jeddah_sep_oct_2022": R.plain(dip22)})
    gt = _gt("DEV-001", filename, "pdf", "en", "native",
             ["monthly_financials", "business_events"],
             sel["source_record_ids"], "2023 (comparatives 2022)",
             expected_metadata={"title": "Financial Performance Report FY2023",
                                "pages": 1, "template": "stmt_en_v1"},
             expected_text=[COMPANY_EN, "Financial Performance Report",
                            "EVT-2023-001",
                            f"Net income row: {R.fmt_money(t23['net_income'])}"],
             expected_fields=["revenue", "cogs", "gross_profit",
                              "operating_expenses", "operating_profit",
                              "other_income", "finance_costs", "tax_expense",
                              "net_income"],
             expected_numeric_values=numeric,
             expected_tables=[{"title": "FY2023 P&L with comparatives",
                               "columns": header, "rows": len(PNL_LABELS)}],
             expected_page_locations=["page 1: title + P&L table",
                                      "page 1: branch paragraph",
                                      "page 1: EVT-2023-001 paragraph"],
             expected_question_targets=[
                 _qt("What was net income in 2023?", QTYPE["lookup"],
                     R.plain(t23["net_income"]) + " SAR", ["page 1, Net income row"]),
                 _qt("What was gross margin in 2023 vs 2022?", QTYPE["margin"],
                     f"{t23['gross_profit']/t23['revenue']*100:.2f}% vs "
                     f"{t22['gross_profit']/t22['revenue']*100:.2f}%",
                     ["page 1, Gross profit + Revenue rows"]),
                 _qt("Which branch contributed most to 2023 revenue?",
                     QTYPE["branch_cmp"], "Riyadh (BR-RUH)",
                     ["page 1, branch paragraph"])],
             generation_parameters={"renderer": "reportlab platypus",
                                    "metadata": "fixed (SOURCE_DATE_EPOCH)"})
    return filename, gt


# ---------------------------------------------------------------------------
# DEV-002 — Arabic branch expense report PDF
# ---------------------------------------------------------------------------
CAT_AR = {"salaries": "الرواتب والمزايا", "logistics": "اللوجستيات والنقل",
          "marketing": "التسويق", "maintenance": "الصيانة",
          "utilities": "المرافق", "rent": "الإيجار"}


def build_dev002(sel, docs_dir):
    R.ensure_pdf_fonts()
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

    A = R.ar
    # A4 content width (595.27pt minus 72pt default margins each side).
    AP = lambda text, size: R.ar_para(text, "Arial", size, 450.0)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("ArTitle", parent=styles["Title"],
                           fontName="Arial-Bold", fontSize=16, alignment=2)
    normal = ParagraphStyle("ArBody", parent=styles["Normal"],
                            fontName="Arial", fontSize=10, leading=15,
                            alignment=2)
    kpi = ParagraphStyle("ArKpi", parent=styles["Normal"],
                         fontName="Arial-Bold", fontSize=13, leading=17,
                         alignment=1)
    cell = ParagraphStyle("ArCell", parent=normal, fontSize=9, leading=12,
                          alignment=1)
    head = ParagraphStyle("ArHead", parent=cell, fontName="Arial-Bold",
                          textColor=colors.white)

    month_rows = [[Paragraph(A("الشهر"), head),
                   Paragraph(A("المصروفات (ر.س)"), head)]]
    for i, r in enumerate(sel["monthly"]):
        month_rows.append([
            Paragraph(A(AR_MONTHS[i]), cell),
            Paragraph(R.fmt_money(r["operating_expenses"]), cell)])
    month_rows.append([
        Paragraph(A("الإجمالي"), head),
        Paragraph(R.fmt_money(sel["total_opex"]), head)])
    t1 = Table(month_rows, colWidths=[60 * mm, 70 * mm])
    t1.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#5f1f1f")),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#5f1f1f")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))

    cat_rows = [[Paragraph(A("البند"), head),
                 Paragraph(A("الإجمالي (ر.س)"), head)]]
    for cat, total in sel["extract_by_category"].items():
        cat_rows.append([Paragraph(A(CAT_AR[cat]), cell),
                         Paragraph(R.fmt_money(total), cell)])
    cat_rows.append([Paragraph(A("مجموع العينة"), head),
                     Paragraph(R.fmt_money(sel["extract_total"]), head)])
    t2 = Table(cat_rows, colWidths=[60 * mm, 70 * mm])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#1f3a5f")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))

    flow = [
        Paragraph(AP(COMPANY_AR, 16), title),
        Paragraph(AP("تقرير المصروفات التشغيلية — فرع جدة — السنة المالية",
                     16), title),
        Spacer(1, 4),
        # Bidi-isolation rule: digits/Latin live on standalone LTR lines so
        # every parser recovers them; Arabic prose stays digit-free.
        Paragraph(AP("إجمالي المصروفات التشغيلية للسنة", 10), normal),
        Paragraph(f"2022 : {R.fmt_money(sel['total_opex'])} SAR", kpi),
        Spacer(1, 6),
        Paragraph(AP("المصروفات التشغيلية الشهرية (ر.س)", 10), normal),
        t1,
        Spacer(1, 8),
        Paragraph(AP("تحليل بنود المصروفات من عينة المعاملات المصنفة", 10),
                  normal),
        t2,
        Spacer(1, 4),
        Paragraph(AP("ملاحظة: يغطي هذا التحليل عينة من المعاملات المصنفة فقط، "
                     "ولا يمثل كامل المصروفات التشغيلية.", 10), normal),
        Paragraph("Sample size: 372 transactions (partial extract).", normal),
        Spacer(1, 4),
        Paragraph(AP("سجلت الرواتب البند الأعلى ضمن العينة المصنفة، تلتها "
                     "تكاليف اللوجستيات. وخلال ربيع عام ألفين واثنين وعشرين "
                     "نفذت الشركة حملة رمضان الترويجية التي رفعت الإنفاق "
                     "التسويقي في ذلك الربع.", 10), normal),
        Paragraph("Reference event: EVT-2022-001 (Ramadan 2022 campaign).",
                  normal),
        Spacer(1, 4),
        Paragraph(AP("المصدر: الحقيقة التشغيلية المعتمدة · مجموعة البيانات "
                     "dataset_v0.1", 10), normal),
    ]
    filename = "DEV-002_branch_expenses_2022_AR.pdf"
    R.build_pdf(docs_dir / filename, flow,
                "Noor Retail - Jeddah Branch Expenses 2022", pagesize=A4)

    numeric = {f"opex_2022_{r['period'][:7]}": R.plain(r["operating_expenses"])
               for r in sel["monthly"]}
    numeric["opex_2022_total"] = R.plain(sel["total_opex"])
    numeric.update({f"extract_{c}": R.plain(v)
                    for c, v in sel["extract_by_category"].items()})
    numeric["extract_total"] = R.plain(sel["extract_total"])
    gt = _gt("DEV-002", filename, "pdf", "ar", "native",
             ["monthly_financials", "transactions"],
             sel["source_record_ids"], "2022, BR-JED",
             expected_metadata={"title_ar": "تقرير المصروفات التشغيلية — فرع جدة",
                                "pages": 1, "template": "branch_exp_ar_v1"},
             expected_text=["تقرير المصروفات التشغيلية", "فرع جدة",
                            "EVT-2022-001",
                            f"2022 : {R.fmt_money(sel['total_opex'])} SAR",
                            "Sample size: 372 transactions"],
             expected_fields=["operating_expenses (monthly + total)",
                              "extract category totals"],
             expected_numeric_values=numeric,
             expected_tables=[
                 {"title_ar": "المصروفات الشهرية", "rows": 12},
                 {"title_ar": "تحليل البنود", "rows": 6}],
             expected_page_locations=["page 1: monthly table",
                                      "page 1: category analysis table"],
             expected_question_targets=[
                 _qt("ما إجمالي المصروفات التشغيلية لفرع جدة في 2022؟",
                     QTYPE["ar_ar"], R.plain(sel["total_opex"]) + " SAR",
                     ["page 1, headline + total row"]),
                 _qt("أي بند مصروفات كان الأعلى ضمن العينة؟", QTYPE["table_agg"],
                     "salaries (الرواتب والمزايا)",
                     ["page 1, category table"])],
             generation_parameters={"renderer": "reportlab platypus",
                                    "arabic": "arabic_reshaper+python-bidi, Arial",
                                    "digits": "Western"})
    return filename, gt


# ---------------------------------------------------------------------------
# DEV-003 — mixed-language supplier invoice PDF
# ---------------------------------------------------------------------------
VENDOR_AR = {"Al-Safi Foods Supplier": "شركة الصافي للأغذية",
             "Gulf Beverages Trading": "الخليج لتجارة المشروبات",
             "National Packaging Co.": "الشركة الوطنية للتغليف",
             "Fresh Harvest Suppliers": "موردو الحصاد الطازج"}


def build_dev003(sel, docs_dir):
    R.ensure_pdf_fonts()
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

    inv = sel["invoice"]
    A = R.ar
    styles = getSampleStyleSheet()
    title = ParagraphStyle("InvTitle", parent=styles["Title"],
                           fontName="Arial-Bold", fontSize=16, alignment=1)
    normal = ParagraphStyle("InvBody", parent=styles["Normal"],
                            fontName="Arial", fontSize=10, leading=14)
    right = ParagraphStyle("InvR", parent=normal, alignment=2)
    cell = ParagraphStyle("InvCell", parent=normal, fontSize=9, leading=12)
    cell_r = ParagraphStyle("InvCellR", parent=cell, alignment=2)
    head = ParagraphStyle("InvHead", parent=cell, fontName="Arial-Bold",
                          textColor=colors.white, alignment=1)

    vendor_ar = VENDOR_AR.get(inv["vendor_customer"], inv["vendor_customer"])
    # Bidi-isolation rule (see DEV-002): Arabic cells carry labels only;
    # IDs/dates/amounts live in LTR cells so every parser recovers them.
    meta = [
        [Paragraph(f"Invoice No: <b>{inv['invoice_id']}</b>", cell),
         Paragraph(A("رقم الفاتورة"), cell_r)],
        [Paragraph(f"Date: {inv['invoice_date']}", cell),
         Paragraph(A("التاريخ"), cell_r)],
        [Paragraph(f"Branch: Riyadh ({inv['branch_id']})", cell),
         Paragraph(A("الفرع: الرياض"), cell_r)],
        [Paragraph(f"Vendor: {inv['vendor_customer']}", cell),
         Paragraph(A(f"المورد: {vendor_ar}"), cell_r)],
    ]
    meta_table = Table(meta, colWidths=[85 * mm, 85 * mm])
    meta_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))

    lines = [[Paragraph("Description", head),
              Paragraph("Amount (SAR)", head),
              Paragraph(A("البيان"), head)]]
    lines.append([
        Paragraph(f"Wholesale packaged foods — ref {inv['related_transaction_id']}",
                  cell),
        Paragraph(R.fmt_money(inv["subtotal"]), cell_r),
        Paragraph(A("بضائع جملة معلبة"), cell)])
    lines_table = Table(lines, colWidths=[85 * mm, 45 * mm, 40 * mm])
    lines_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))

    totals = [
        [Paragraph("Subtotal", cell),
         Paragraph(R.fmt_money(inv["subtotal"]), cell_r),
         Paragraph(A("المجموع الفرعي"), cell)],
        [Paragraph("VAT 15%", cell),
         Paragraph(R.fmt_money(inv["vat"]), cell_r),
         Paragraph(A("ضريبة القيمة المضافة 15%"), cell)],
        [Paragraph("<b>Total due</b>", cell),
         Paragraph(f"<b>{R.fmt_money(inv['total'])}</b>", cell_r),
         Paragraph(A("الإجمالي المستحق"), cell)],
    ]
    totals_table = Table(totals, colWidths=[85 * mm, 45 * mm, 40 * mm])
    totals_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("LINEBELOW", (0, -1), (-1, -1), 1.5, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))

    flow = [
        Paragraph(f"TAX INVOICE / {A('فاتورة ضريبية')}", title),
        Paragraph(COMPANY_EN, normal),
        Spacer(1, 6),
        meta_table,
        Spacer(1, 6),
        lines_table,
        Spacer(1, 4),
        totals_table,
        Spacer(1, 6),
        Paragraph(f"Related transaction: {inv['related_transaction_id']} "
                  f"&middot; Status: {inv['status']} &middot; dataset_v0.1",
                  normal),
    ]
    filename = "DEV-003_supplier_invoice_2023_MIX.pdf"
    R.build_pdf(docs_dir / filename, flow,
                f"Supplier invoice {inv['invoice_id']}", pagesize=A4)

    gt = _gt("DEV-003", filename, "pdf", "mixed", "native",
             ["invoices", "transactions"], sel["source_record_ids"],
             inv["invoice_date"],
             expected_metadata={"template": "invoice_mix_v1", "pages": 1},
             expected_text=[inv["invoice_id"], "TAX INVOICE",
                            "فاتورة ضريبية",
                            f"Total due {R.fmt_money(inv['total'])}"],
             expected_fields=["invoice_id", "invoice_date", "vendor_customer",
                              "branch_id", "subtotal", "vat", "total",
                              "status", "related_transaction_id"],
             expected_numeric_values={
                 "subtotal": R.plain(inv["subtotal"]),
                 "vat": R.plain(inv["vat"]),
                 "total": R.plain(inv["total"])},
             expected_identifiers={"invoice_id": inv["invoice_id"],
                                   "related_transaction_id":
                                       inv["related_transaction_id"]},
             expected_tables=[{"title": "invoice lines + totals", "rows": 4}],
             expected_page_locations=["page 1: header + invoice ID",
                                      "page 1: totals block"],
             expected_question_targets=[
                 _qt(f"What is the total of invoice {inv['invoice_id']}?",
                     QTYPE["identifier"], R.plain(inv["total"]) + " SAR",
                     ["page 1, totals block"]),
                 _qt("Who is the vendor on that invoice?", QTYPE["identifier"],
                     inv["vendor_customer"], ["page 1, vendor block"])],
             generation_parameters={"renderer": "reportlab platypus",
                                    "arabic": "arabic_reshaper+python-bidi, Arial"})
    return filename, gt


# ---------------------------------------------------------------------------
# DEV-004 — scanned Arabic receipt (PNG raster)
# ---------------------------------------------------------------------------
def build_dev004(sel, docs_dir):
    from PIL import ImageDraw

    t = sel["transaction"]
    A = R.ar
    W, H = 900, 1150
    img = R.new_page(W, H)
    draw = ImageDraw.Draw(img)
    f_title = R.get_font(44, bold=True)
    f_body = R.get_font(30)
    f_small = R.get_font(24)
    cx = W // 2

    def text_c(y, s, font):
        box = draw.textbbox((0, 0), s, font=font)
        draw.text(((W - (box[2] - box[0])) / 2, y), s, font=font,
                  fill="black")

    y = 60
    text_c(y, A("إيصال مصروفات"), f_title)
    y += 70
    text_c(y, A("فرع جدة — نسخة ممسوحة من النظام"), f_small)
    y += 60
    # faint ruled background
    for ly in range(y, H - 140, 56):
        draw.line([(70, ly), (W - 70, ly)], fill=(225, 225, 225), width=1)
    rows = [
        (A("البائع"), A("وكالة المدينة للتسويق")),
        (A("التاريخ"), t["date"]),
        (A("الوصف"), A("خدمات تسويقية — حملة يونيو")),
        (A("المبلغ"), f"{R.fmt_money(t['amount'])} SAR"),
        (A("الضريبة 15%"), f"{R.fmt_money(t['tax_amount'])} SAR"),
        (A("الإجمالي"), f"{R.fmt_money(t['total_amount'])} SAR"),
        (A("المرجع"), t["transaction_id"]),
    ]
    for label, value in rows:
        draw.text((70, y), label, font=f_body, fill="black")
        vb = draw.textbbox((0, 0), value, font=f_body)
        draw.text((W - 70 - (vb[2] - vb[0]), y), value, font=f_body,
                  fill="black")
        y += 56
    # stamp box
    draw.rectangle([(W - 330, H - 260), (W - 70, H - 100)], outline="black",
                   width=3)
    text_c(H - 220, A("ختم الفرع"), f_body)
    draw.text((80, H - 160), A("توقيع أمين الصندوق"), font=f_small,
              fill="black")
    draw.line([(80, H - 120), (380, H - 120)], fill="black", width=2)

    params = {"rotation_deg": 2.5, "blur_radius_px": 0.6,
              "brightness_factor": 1.04, "noise_sigma": 3.0,
              "format": "PNG (lossless)", "seed": 42}
    img = R.degrade(img, rotation=params["rotation_deg"],
                    blur=params["blur_radius_px"],
                    brightness=params["brightness_factor"],
                    noise_sigma=params["noise_sigma"], seed=params["seed"])
    filename = "DEV-004_scanned_receipt_2019_AR.png"
    img.save(docs_dir / filename, format="PNG")

    gt = _gt("DEV-004", filename, "png", "ar", "scanned",
             ["transactions"], sel["source_record_ids"], t["date"],
             expected_metadata={"width_px": img.size[0],
                                "height_px": img.size[1],
                                "template": "receipt_scan_ar_v1"},
             expected_text=["إيصال مصروفات", "وكالة المدينة للتسويق",
                            "ختم الفرع", t["transaction_id"],
                            R.fmt_money(t["total_amount"])],
             expected_fields=["vendor_customer", "date", "amount",
                              "tax_amount", "total_amount",
                              "reference(transaction_id)"],
             expected_numeric_values={
                 "amount": R.plain(t["amount"]),
                 "tax_amount": R.plain(t["tax_amount"]),
                 "total_amount": R.plain(t["total_amount"])},
             expected_identifiers={"transaction_id": t["transaction_id"]},
             expected_visual_elements=["receipt header", "amount lines",
                                       "stamp box", "signature line",
                                       "ruled background"],
             expected_page_locations=["whole image (no pages)"],
             expected_question_targets=[
                 _qt("What amount is shown on the 2019 receipt?",
                     QTYPE["ar_ar"], R.plain(t["total_amount"]) + " SAR",
                     ["whole image, total line"]),
                 _qt("Which vendor issued it?", QTYPE["ar_ar"],
                     "City Marketing Agency (وكالة المدينة للتسويق)",
                     ["whole image, vendor line"])],
             generation_parameters={"renderer": "Pillow composition",
                                    "degradation": params,
                                    "arabic": "arabic_reshaper+python-bidi, Arial",
                                    "clean_text_note": "expected_text verified "
                                    "against pre-degradation render strings"})
    return filename, gt


# ---------------------------------------------------------------------------
# DEV-005 — mixed-language management commentary (DOCX)
# ---------------------------------------------------------------------------
def _docx_par(doc, text, *, rtl=False, bold=False, size=11):
    from docx.shared import Pt, RGBColor
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    p = doc.add_paragraph()
    if rtl:
        pPr = p._p.get_or_add_pPr()
        bidi = OxmlElement("w:bidi")
        bidi.set(qn("w:val"), "1")
        pPr.append(bidi)
        p.alignment = 2  # right
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
    if rtl:
        rPr = run._r.get_or_add_rPr()
        rFonts = OxmlElement("w:rFonts")
        for attr in ("w:ascii", "w:hAnsi", "w:cs"):
            rFonts.set(qn(attr), "Arial")
        rPr.append(rFonts)
        szCs = OxmlElement("w:szCs")
        szCs.set(qn("w:val"), str(size * 2))
        rPr.append(szCs)
    else:
        run.font.name = "Calibri"
    return p


def build_dev005(sel, docs_dir):
    from docx import Document

    doc = Document()
    _docx_par(doc, "Noor Retail & Distribution Co.", bold=True, size=16)
    _docx_par(doc, "Management Commentary — Fiscal Year 2023", bold=True,
              size=13)
    _docx_par(doc, COMPANY_AR, rtl=True, bold=True, size=14)
    _docx_par(doc, "تعليق الإدارة — السنة المالية 2023", rtl=True, bold=True,
              size=12)

    dip = sel["jed_sep_oct_2023"]
    dip_prev = sel["jed_sep_oct_2022"]
    rev = sel["company_revenue_2023"]
    en_paras = [
        "In 2023 the company delivered revenue of SAR "
        f"{R.fmt_money(rev)}. Growth was led by the Riyadh branch while the "
        "Dammam branch continued the ramp-up that began with its opening on "
        "2019-01-01 (event EVT-2019-001).",
        "In September and October 2023 part of the Jeddah branch was closed "
        "for refit works (event EVT-2023-001). Jeddah revenue for those two "
        f"months was SAR {R.fmt_money(dip)}, compared with SAR "
        f"{R.fmt_money(dip_prev)} a year earlier, and recovered after the "
        "branch fully reopened. Management treats this dip as temporary and "
        "non-recurring.",
        "The spring 2022 Ramadan promotional campaign (event EVT-2022-001) "
        "lifted revenue above the seasonal norm at the cost of higher campaign "
        "spending in the same quarter; the 2021–2022 supplier cost increase "
        "compressed gross margin until prices were renegotiated. Both effects "
        "had faded from year-on-year comparisons by late 2023.",
    ]
    for para in en_paras:
        _docx_par(doc, para)
    ar_paras = [
        "أغلق فرع جدة جزئياً خلال سبتمبر وأكتوبر 2023 لأعمال التطوير "
        "(EVT-2023-001)، فانخفضت إيرادات الفرع في تلك الفترة مقارنة بالعام "
        "السابق، ثم تعافت بعد إعادة الافتتاح الكامل. وتعد الإدارة هذا "
        "الانخفاض مؤقتاً وغير متكرر.",
        f"على مستوى الشركة بلغت إيرادات 2023 ما قيمته {R.fmt_money(rev)} ر.س، "
        "بقيادة فرع الرياض واستمرار نمو فرع الدمام منذ افتتاحه في 2019-01-01 "
        "(EVT-2019-001).",
        "أما حملة رمضان الترويجية في ربيع 2022 (EVT-2022-001) فقد رفعت "
        "الإيرادات فوق المعتاد الموسمي مقابل إنفاق أعلى في الربع نفسه، بينما "
        "ضغط ارتفاع تكاليف الموردين في 2021–2022 على هامش الربح حتى أعيد "
        "التفاوض على الأسعار.",
    ]
    for para in ar_paras:
        _docx_par(doc, para, rtl=True)
    _docx_par(doc, "Source events: EVT-2023-001, EVT-2022-001, EVT-2019-001 "
                   "· canonical business_events · dataset_v0.1")

    filename = "DEV-005_management_commentary_2023_MIX.docx"
    path = docs_dir / filename
    doc.save(str(path))
    R.normalize_zip(path)

    gt = _gt("DEV-005", filename, "docx", "mixed", "native",
             ["business_events", "monthly_financials"],
             sel["source_record_ids"], "2023",
             expected_metadata={"paragraphs": 3 + 3 + 4,
                                "template": "commentary_mix_v1"},
             expected_text=["Management Commentary", "تعليق الإدارة",
                            "EVT-2023-001", "EVT-2022-001", "EVT-2019-001",
                            R.fmt_money(dip)],
             expected_fields=["event explanations (EN+AR)",
                              "company_revenue_2023",
                              "jeddah Sep-Oct 2023 vs 2022"],
             expected_numeric_values={
                 "company_revenue_2023": R.plain(rev),
                 "jeddah_sep_oct_2023": R.plain(dip),
                 "jeddah_sep_oct_2022": R.plain(dip_prev)},
             expected_identifiers={"event_ids": ["EVT-2023-001",
                                                 "EVT-2022-001",
                                                 "EVT-2019-001"]},
             expected_page_locations=["p1: title + EN paragraphs",
                                      "p2: AR paragraphs (RTL)"],
             expected_question_targets=[
                 _qt("Why did Jeddah revenue dip in late 2023, per management?",
                     QTYPE["explain"],
                     "Partial refit closure Sep-Oct 2023 (EVT-2023-001); "
                     "temporary, recovered after reopening.",
                     ["EN para 2 / AR para 1"]),
                 _qt("ماذا قالت الإدارة عن سبب ارتفاع المصروفات؟",
                     QTYPE["mixed"],
                     "Ramadan 2022 campaign spending (EVT-2022-001).",
                     ["EN para 3 / AR para 3"])],
             generation_parameters={"renderer": "python-docx",
                                    "arabic": "logical text + RTL paragraphs "
                                    "(Word shapes natively)",
                                    "content_rule": "every fact traces to "
                                    "canonical rows/event rows; no invented facts"})
    return filename, gt


# ---------------------------------------------------------------------------
# DEV-006 — 2023 transaction workbook (XLSX)
# ---------------------------------------------------------------------------
XLSX_COLUMNS = ("transaction_id", "date", "branch_id", "department_id",
                "transaction_type", "category", "vendor_customer",
                "description_en", "amount", "tax_amount", "total_amount",
                "payment_method", "reference_number")
XLSX_MONEY = {"amount", "tax_amount", "total_amount"}


def build_dev006(sel, docs_dir):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill

    wb = Workbook()
    ws = wb.active
    ws.title = "transactions_2023"
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="1F3A5F")
    for col, name in enumerate(XLSX_COLUMNS, start=1):
        cell = ws.cell(row=1, column=col, value=name)
        cell.font = header_font
        cell.fill = header_fill
    for i, t in enumerate(sel["rows"], start=2):
        for col, name in enumerate(XLSX_COLUMNS, start=1):
            value = t[name]
            if name in XLSX_MONEY:
                value = float(value)  # canonical Decimal -> number cell
            cell = ws.cell(row=i, column=col, value=value)
            if name in XLSX_MONEY:
                cell.number_format = "#,##0.00"
    widths = {"transaction_id": 16, "date": 12, "branch_id": 10,
              "department_id": 12, "transaction_type": 14, "category": 14,
              "vendor_customer": 30, "description_en": 52, "amount": 14,
              "tax_amount": 13, "total_amount": 15, "payment_method": 14,
              "reference_number": 16}
    for idx, name in enumerate(XLSX_COLUMNS, start=1):
        ws.column_dimensions[ws.cell(row=1, column=idx).column_letter].width = \
            widths[name]
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:M{len(sel['rows']) + 1}"

    filename = "DEV-006_monthly_transactions_2023_EN.xlsx"
    path = docs_dir / filename
    wb.save(str(path))
    R.normalize_zip(path)

    first_id = sel["rows"][0]["transaction_id"]
    last_id = sel["rows"][-1]["transaction_id"]
    proc_q4 = sum((t["total_amount"] for t in sel["rows"]
                   if t["department_id"] == "DEP-PROC"
                   and t["date"][:7] in ("2023-10", "2023-11", "2023-12")),
                  Decimal("0.00"))
    gt = _gt("DEV-006", filename, "xlsx", "en", "native",
             ["transactions"], sel["source_record_ids"], "2023",
             expected_metadata={"sheet": "transactions_2023",
                                "rows_data": len(sel["rows"]),
                                "columns": list(XLSX_COLUMNS)},
             expected_text=["transactions_2023", first_id, last_id],
             expected_fields=list(XLSX_COLUMNS),
             expected_numeric_values={
                 "row_count": str(len(sel["rows"])),
                 "procurement_q4_2023_total": R.plain(proc_q4)},
             expected_identifiers={"first_transaction_id": first_id,
                                   "last_transaction_id": last_id},
             expected_tables=[{"title": "transactions_2023",
                               "rows": len(sel["rows"]),
                               "columns": len(XLSX_COLUMNS)}],
             expected_page_locations=["sheet transactions_2023 (single table)"],
             expected_question_targets=[
                 _qt("What were total Q4 2023 procurement costs?",
                     QTYPE["table_agg"], R.plain(proc_q4) + " SAR",
                     ["sheet transactions_2023, DEP-PROC + Q4 filter"]),
                 _qt("How many transactions are in the 2023 extract?",
                     QTYPE["total"], str(len(sel["rows"])),
                     ["sheet row count"])],
             generation_parameters={"renderer": "openpyxl",
                                    "money_cells": "numbers with #,##0.00; "
                                    "equality holds via Decimal(str(cell))",
                                    "order": "date, transaction_id"})
    return filename, gt


# ---------------------------------------------------------------------------
# DEV-007 — H1-2024 logistics transaction extract (CSV, mixed)
# ---------------------------------------------------------------------------
CSV_COLUMNS = ("transaction_id", "date", "branch_id", "department_id",
               "transaction_type", "category", "vendor_customer",
               "description_en", "description_ar", "amount", "tax_amount",
               "total_amount", "payment_method", "reference_number")


def build_dev007(sel, docs_dir):
    import csv as csvmod

    filename = "DEV-007_transactions_extract_2024_MIX.csv"
    path = docs_dir / filename
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csvmod.DictWriter(f, fieldnames=list(CSV_COLUMNS),
                                   lineterminator="\n")
        writer.writeheader()
        for t in sel["rows"]:
            # Verbatim canonical cells (Decimal -> plain 2dp strings).
            writer.writerow({c: (R.plain(t[c]) if c in
                                 ("amount", "tax_amount", "total_amount")
                                 else t[c]) for c in CSV_COLUMNS})

    gt = _gt("DEV-007", filename, "csv", "mixed", "native",
             ["transactions"], sel["source_record_ids"],
             "2024-01..2024-06, DEP-LOG",
             expected_metadata={"rows_data": len(sel["rows"]),
                                "encoding": "UTF-8, no BOM"},
             expected_text=["description_ar present",
                            sel["rows"][0]["transaction_id"]],
             expected_fields=list(CSV_COLUMNS),
             expected_numeric_values={
                 "row_count": str(len(sel["rows"])),
                 "total_amount_sum": R.plain(sel["total"])},
             expected_identifiers={"first_transaction_id":
                                   sel["rows"][0]["transaction_id"],
                                   "last_transaction_id":
                                   sel["rows"][-1]["transaction_id"]},
             expected_tables=[{"title": "flat extract",
                               "rows": len(sel["rows"])}],
             expected_page_locations=["whole file (no pages)"],
             expected_question_targets=[
                 _qt("Total Logistics spending in H1 2024?",
                     QTYPE["table_agg"], R.plain(sel["total"]) + " SAR",
                     ["whole file, sum total_amount"]),
                 _qt("How many transactions exceed 10,000 SAR?",
                     QTYPE["table_agg"], "computed from the 346 rows",
                     ["whole file, amount filter"])],
             generation_parameters={"renderer": "csv module, LF, UTF-8",
                                    "cells": "verbatim canonical strings"})
    return filename, gt


# ---------------------------------------------------------------------------
# DEV-008 — 10-year company revenue chart (PNG)
# ---------------------------------------------------------------------------
def build_dev008(sel, docs_dir):
    years = [y for y, _ in sel["annual"]]
    values = [v for _, v in sel["annual"]]
    img = R.draw_line_chart(
        f"{COMPANY_EN} — Annual Company Revenue 2015-2024 (SAR)",
        "Year", "Revenue (SAR)", years, [float(v) for v in values],
        [f"{float(v) / 1e6:.1f}M" for v in values])

    filename = "DEV-008_revenue_trend_2015_2024_EN.png"
    img.save(docs_dir / filename, format="PNG")

    gt = _gt("DEV-008", filename, "png", "en", "native",
             ["monthly_financials"], sel["source_record_ids"], "2015..2024",
             expected_metadata={"width_px": 1600, "height_px": 1000,
                                "template": "revenue_line_en_v1"},
             expected_text=["Annual Company Revenue 2015–2024"],
             expected_chart_data={
                 "chart_type": "line", "x": years,
                 "series": [{"label": "company revenue (SAR)",
                             "values": [R.plain(v) for v in values]}],
                 "trend": "steady growth with a 2019 level shift from the "
                          "Dammam opening and a 2020 demand surge; 2024 highest."},
             expected_visual_elements=["line + markers", "value labels (M)",
                                       "grid", "title + axes"],
             expected_page_locations=["whole image (no pages)"],
             expected_question_targets=[
                 _qt("In which year was revenue highest?", QTYPE["chart"],
                     "2024", ["whole image, last point"]),
                 _qt("Show me the revenue trend chart.", QTYPE["chart"],
                     "DEV-008 image", ["whole image"])],
             generation_parameters={"renderer": "Pillow line chart 1600x1000",
                                    "labels": "millions, 1dp; exact sums in "
                                    "expected_chart_data"})
    return filename, gt


# ---------------------------------------------------------------------------
# DEV-009 — Q4-2023 KPI dashboard (JPG, mixed)
# ---------------------------------------------------------------------------
def build_dev009(sel, docs_dir):
    A = R.ar
    kpis = sel["kpis"]
    branch_rev = sel["branch_revenue"]

    cards = [("Revenue", f"{float(kpis['revenue']) / 1e6:.2f}M SAR"),
             ("Gross profit", f"{float(kpis['gross_profit']) / 1e6:.2f}M SAR"),
             ("Operating expenses",
              f"{float(kpis['operating_expenses']) / 1e6:.2f}M SAR"),
             ("Net income", f"{float(kpis['net_income']) / 1e6:.2f}M SAR")]
    bar_labels = ["Riyadh", "Jeddah", "Dammam"]
    bar_values = [float(branch_rev[b]) for b in ("BR-RUH", "BR-JED", "BR-DMM")]
    # Budget-verdict semantic rule (spec §DEV-009, no tolerance threshold):
    # revenue-type targets: actual >= budget -> met; expense-type budgets
    # would use actual <= budget -> met. This dashboard is revenue-type.
    status = ("Budget met" if sel["variance"] >= 0 else "Budget missed")
    panel_lines = [
        f"Budget:  {float(sel['budget_total']) / 1e6:.2f}M SAR",
        f"Actual:  {float(kpis['revenue']) / 1e6:.2f}M SAR",
        f"Variance: {float(sel['variance']):+,.2f} SAR",
        f"({float(sel['variance_pct']):+.2f}%)  {status}",
        A("تحقق المستهدف الربع سنوي"),
    ]
    img = R.draw_dashboard(
        f"{COMPANY_EN} — KPI Dashboard Q4 2023  |  "
        f"{A('لوحة المؤشرات — الربع الرابع 2023')}",
        cards, "Q4 2023 revenue by branch (SAR)", bar_labels, bar_values,
        f"Budget vs actual — revenue  {A('(الميزانية مقابل الفعلي)')}",
        panel_lines)

    filename = "DEV-009_kpi_dashboard_Q4-2023_MIX.jpg"
    img.save(docs_dir / filename, format="JPEG", quality=92, subsampling=1,
             optimize=False)

    gt = _gt("DEV-009", filename, "jpg", "mixed", "native",
             ["monthly_financials", "budgets"], sel["source_record_ids"],
             "2023-Q4",
             expected_metadata={"width_px": 1600, "height_px": 1000,
                                "jpeg_quality": 92, "exif": "none",
                                "template": "kpi_dash_mix_v1"},
             expected_text=["KPI Dashboard Q4 2023", "لوحة مؤشرات الأداء",
                            "Budget met" if sel["variance"] >= 0
                            else "Budget missed"],
             expected_numeric_values={
                 "revenue": R.plain(kpis["revenue"]),
                 "gross_profit": R.plain(kpis["gross_profit"]),
                 "operating_expenses": R.plain(kpis["operating_expenses"]),
                 "net_income": R.plain(kpis["net_income"]),
                 "budget_total": R.plain(sel["budget_total"]),
                 "variance": R.plain(sel["variance"]),
                 "branch_revenue_BR-RUH": R.plain(branch_rev["BR-RUH"]),
                 "branch_revenue_BR-JED": R.plain(branch_rev["BR-JED"]),
                 "branch_revenue_BR-DMM": R.plain(branch_rev["BR-DMM"])},
             expected_visual_elements=["4 KPI cards", "branch bar chart",
                                       "budget-vs-actual panel"],
             expected_page_locations=["whole image: KPI cards (top)",
                                      "whole image: branch bars (left)",
                                      "whole image: budget panel (right)"],
             expected_question_targets=[
                 _qt("Which branch led Q4 2023 revenue?", QTYPE["chart"],
                     "Riyadh (BR-RUH)", ["branch bar panel"]),
                 _qt("Was the quarterly revenue budget met?", QTYPE["chart_text"],
                     f"Yes: +{R.plain(sel['variance'])} SAR "
                     f"({float(sel['variance_pct']):+.2f}%)",
                     ["budget panel"])],
             generation_parameters={"renderer": "Pillow dashboard 1600x1000 -> "
                                    "JPEG q92, no EXIF",
                                    "arabic": "arabic_reshaper+python-bidi, Arial",
                                    "jpeg_note": "artifacts presentation-only; "
                                    "exact 2dp values in ground truth"})
    return filename, gt


# ---------------------------------------------------------------------------
# DEV-010 — scanned English balance sheet (raster-page PDF)
# ---------------------------------------------------------------------------
def build_dev010(sel, docs_dir):
    from PIL import ImageDraw

    row = sel["row2020"]
    W, H = 1240, 1754
    img = R.new_page(W, H)
    draw = ImageDraw.Draw(img)
    f_title = R.get_font(40, bold=True)
    f_head = R.get_font(30, bold=True)
    f_body = R.get_font(28)

    def text_c(y, s, font, fill="black"):
        box = draw.textbbox((0, 0), s, font=font)
        draw.text(((W - (box[2] - box[0])) / 2, y), s, font=font, fill=fill)

    y = 70
    text_c(y, COMPANY_EN, f_title)
    y += 60
    text_c(y, "Statement of Financial Position — 31 December 2020 (SAR)",
           f_head)
    y += 80

    def table(title, items, total_label, total_value):
        nonlocal y
        draw.text((90, y), title, font=f_head, fill="black")
        y += 52
        draw.rectangle([(90, y), (W - 90, y + 2)], fill="black")
        y += 12
        for label, value in items:
            draw.text((110, y), label, font=f_body, fill="black")
            v = R.fmt_money(value)
            vb = draw.textbbox((0, 0), v, font=f_body)
            draw.text((W - 110 - (vb[2] - vb[0]), y), v, font=f_body,
                      fill="black")
            y += 48
        draw.rectangle([(90, y), (W - 90, y + 2)], fill="black")
        y += 12
        draw.text((110, y), total_label, font=f_head, fill="black")
        v = R.fmt_money(total_value)
        vb = draw.textbbox((0, 0), v, font=f_head)
        draw.text((W - 110 - (vb[2] - vb[0]), y), v, font=f_head,
                  fill="black")
        y += 70

    table("ASSETS", [("Cash", row["cash"]),
                     ("Accounts receivable", row["accounts_receivable"]),
                     ("Inventory", row["inventory"]),
                     ("Other current assets", row["other_current_assets"]),
                     ("Property and equipment (net)",
                      row["property_and_equipment"])],
          "TOTAL ASSETS", row["total_assets"])
    table("LIABILITIES", [("Accounts payable", row["accounts_payable"]),
                          ("Debt", row["debt"]),
                          ("Other liabilities", row["other_liabilities"])],
          "TOTAL LIABILITIES", row["total_liabilities"])
    draw.text((90, y), "EQUITY", font=f_head, fill="black")
    v = R.fmt_money(row["equity"])
    vb = draw.textbbox((0, 0), v, font=f_head)
    draw.text((W - 110 - (vb[2] - vb[0]), y), v, font=f_head, fill="black")
    y += 60
    text_c(y, "Total liabilities + Equity = "
           f"{R.fmt_money(row['total_liabilities'] + row['equity'])}",
           f_body)
    y += 50
    text_c(y, "Scanned from branch archive · page 1 of 1", R.get_font(22),
           fill=(90, 90, 90))

    params = {"rotation_deg": 1.8, "blur_radius_px": 0.5,
              "brightness_factor": 0.97, "noise_sigma": 2.5,
              "bands": [(400, 414, 0.05), (1100, 1112, 0.04)],
              "page_encoding": "JPEG q85 (scan-realistic)",
              "seed": 42}
    img = R.degrade(img, rotation=params["rotation_deg"],
                    blur=params["blur_radius_px"],
                    brightness=params["brightness_factor"],
                    noise_sigma=params["noise_sigma"], seed=params["seed"],
                    bands=params["bands"])
    filename = "DEV-010_balance_sheet_2020_EN_scanned.pdf"
    R.pdf_from_image(docs_dir / filename, img,
                     "Noor Retail - Balance Sheet 2020 (scan)",
                     jpeg_quality=85)

    numeric = {k: R.plain(row[k]) for k in
               ("cash", "accounts_receivable", "inventory",
                "other_current_assets", "property_and_equipment",
                "total_assets", "accounts_payable", "debt",
                "other_liabilities", "total_liabilities", "equity")}
    numeric["equity_2019_context"] = R.plain(sel["equity2019"])
    gt = _gt("DEV-010", filename, "pdf", "en", "scanned",
             ["annual_balance_sheet"], sel["source_record_ids"], "2020",
             expected_metadata={"pages": 1, "page_type": "single raster image",
                                "template": "bs_scan_en_v1"},
             expected_text=["Statement of Financial Position",
                            "TOTAL ASSETS",
                            R.fmt_money(row["total_assets"])],
             expected_fields=["cash", "accounts_receivable", "inventory",
                              "other_current_assets",
                              "property_and_equipment", "total_assets",
                              "accounts_payable", "debt", "other_liabilities",
                              "total_liabilities", "equity"],
             expected_numeric_values=numeric,
             expected_tables=[{"title": "Assets", "rows": 5},
                              {"title": "Liabilities", "rows": 3},
                              {"title": "Equity", "rows": 1}],
             expected_page_locations=["page 1 (image): Assets table",
                                      "page 1 (image): Liabilities table",
                                      "page 1 (image): Equity line"],
             expected_question_targets=[
                 _qt("What were total assets at end of 2020?",
                     QTYPE["table_cell"], R.plain(row["total_assets"]) + " SAR",
                     ["page 1, TOTAL ASSETS row"]),
                 _qt("Compare 2020 vs 2019 equity.", QTYPE["delta"],
                     f"{R.plain(row['equity'])} vs "
                     f"{R.plain(sel['equity2019'])}",
                     ["page 1, Equity line + truth context"])],
             generation_parameters={"renderer": "Pillow composition -> "
                                    "single-image PDF",
                                    "degradation": params})
    return filename, gt
