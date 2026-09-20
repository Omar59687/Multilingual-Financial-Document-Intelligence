"""Central configuration for the MizanIQ canonical synthetic-data generator.

All numeric assumptions live here — the generator itself must not contain
undocumented magic numbers. Every parameter below is part of dataset_v0.1
and is recorded (by name/value where practical) in the dataset manifest.

Money policy: SAR, 2 decimal places, ROUND_HALF_UP. Canonical money values
are ``decimal.Decimal`` quantized at row-write time. ``random.Random`` (and
``random.gauss`` / ``random.lognormvariate``) supplies stochastic inputs;
outputs are always quantized, so no binary float ever reaches a CSV cell.
"""

from decimal import Decimal

# ---------------------------------------------------------------------------
# Dataset identity / reproducibility
# ---------------------------------------------------------------------------
DATASET_VERSION = "dataset_v0.1"
SCHEMA_VERSION = "0.1"
RANDOM_SEED = 42  # single fixed seed; recorded in manifest.json

START_YEAR = 2015
END_YEAR = 2024  # 2024 is the default final forecasting holdout (never fitted)
CURRENCY = "SAR"

COMPANY_NAME_EN = "Noor Retail & Distribution Co."
COMPANY_NAME_AR = "شركة نور للتجزئة والتوزيع"

# ---------------------------------------------------------------------------
# Simplified tax assumptions (V1 — intentionally NOT historically accurate)
# ---------------------------------------------------------------------------
# Saudi VAT actually changed over time (5% from 2018, 15% from July 2020).
# V1 uses ONE flat 15% rate on every invoice/transaction tax field,
# regardless of date. Recorded in the manifest as a known simplification.
VAT_RATE = Decimal("0.15")

# Generic synthetic income-statement tax line. Saudi Zakat rules are NOT
# modeled (architect decision). Flat rate on positive pre-tax profit.
TAX_RATE_ON_POSITIVE_PRETAX = Decimal("0.11")

SNAPSHOT_DATE = "2024-12-31"  # invoice status (paid/pending/overdue) is as-of

# ---------------------------------------------------------------------------
# Branches — Dammam opens 2019-01-01 (architect-approved). Pre-opening
# periods are NOT APPLICABLE / NULL: absent from tables, never zero.
# ---------------------------------------------------------------------------
BRANCHES = [
    {
        "branch_id": "BR-RUH",
        "branch_name_en": "Riyadh",
        "branch_name_ar": "الرياض",
        "city": "Riyadh",
        "opened_date": "2010-06-01",
        "status": "open",
    },
    {
        "branch_id": "BR-JED",
        "branch_name_en": "Jeddah",
        "branch_name_ar": "جدة",
        "city": "Jeddah",
        "opened_date": "2012-04-01",
        "status": "open",
    },
    {
        "branch_id": "BR-DMM",
        "branch_name_en": "Dammam",
        "branch_name_ar": "الدمام",
        "city": "Dammam",
        "opened_date": "2019-01-01",
        "status": "open",
    },
]

# ---------------------------------------------------------------------------
# Departments — flat list in V1 (no hierarchy, no cost centers).
# ---------------------------------------------------------------------------
DEPARTMENTS = [
    {"department_id": "DEP-SALES", "department_name_en": "Sales",
     "department_name_ar": "المبيعات"},
    {"department_id": "DEP-PROC", "department_name_en": "Procurement",
     "department_name_ar": "المشتريات"},
    {"department_id": "DEP-LOG", "department_name_en": "Warehouse & Logistics",
     "department_name_ar": "المستودعات والخدمات اللوجستية"},
    {"department_id": "DEP-ADM", "department_name_en": "Administration",
     "department_name_ar": "الإدارة"},
]

# ---------------------------------------------------------------------------
# Monthly financial series parameters, per branch.
#
# value(t) = baseline * growth(t) * seasonality(t) [+ ramadan/summer peaks]
#            * noise(t) [* event multipliers] [+ event adders]
#
# * base_revenue: revenue of the branch's first active month (SAR).
# * annual_growth: compound annual growth applied monthly.
# * ramadan_uplift / summer_uplift: peak-month demand multipliers (share).
# * revenue_noise_cv: multiplicative Gaussian noise std-dev on revenue.
# * cogs_ratio / cogs_noise_sd: cost-of-sales share of revenue + noise.
# * opex_ratio / opex_noise_sd: operating-expense share of revenue + noise.
# * opening_ramp_months (Dammam only): linear 0.55 -> 1.0 scale-up after
#   opening — the documented structural break (never back-filled as zero).
# ---------------------------------------------------------------------------
MONTHLY_PARAMS = {
    "BR-RUH": {
        "base_revenue": 1850000.0,
        "annual_growth": 0.070,
        "ramadan_uplift": 0.18,
        "summer_uplift": 0.06,
        "revenue_noise_cv": 0.040,
        "cogs_ratio": 0.640,
        "cogs_noise_sd": 0.012,
        "opex_ratio": 0.218,
        "opex_noise_sd": 0.010,
        "opening_ramp_months": 0,
    },
    "BR-JED": {
        "base_revenue": 1250000.0,
        "annual_growth": 0.060,
        "ramadan_uplift": 0.20,
        "summer_uplift": 0.08,
        "revenue_noise_cv": 0.045,
        "cogs_ratio": 0.648,
        "cogs_noise_sd": 0.013,
        "opex_ratio": 0.224,
        "opex_noise_sd": 0.011,
        "opening_ramp_months": 0,
    },
    "BR-DMM": {
        "base_revenue": 620000.0,  # first active month: 2019-01
        "annual_growth": 0.120,  # faster catch-up growth after opening
        "ramadan_uplift": 0.16,
        "summer_uplift": 0.07,
        "revenue_noise_cv": 0.060,
        "cogs_ratio": 0.655,
        "cogs_noise_sd": 0.014,
        "opex_ratio": 0.235,
        "opex_noise_sd": 0.012,
        "opening_ramp_months": 12,
        "opening_ramp_start": 0.55,
    },
}

# Ramadan peak month per year (documented approximation of the lunar drift;
# the moving peak defeats naive month-dummy memorization). Adjacent months
# receive half the uplift.
RAMADAN_PEAK_MONTH = {
    2015: 6, 2016: 6, 2017: 5, 2018: 5, 2019: 5,
    2020: 4, 2021: 4, 2022: 4, 2023: 3, 2024: 3,
}

SUMMER_PEAK_MONTHS = (7, 8)  # beverage-season uplift window

# Sparse below-operating lines: probability a month carries the line, and
# the uniform amount band (SAR). Keeps net_income ~= operating_profit in
# most months while exercising the full formula where it matters.
OTHER_INCOME_PROB = 0.08
OTHER_INCOME_BAND = (2000.0, 15000.0)
FINANCE_COSTS_PROB = 0.10
FINANCE_COSTS_BAND = (1000.0, 9000.0)

# ---------------------------------------------------------------------------
# Business events (~8-12 curated). `effect` drives the generator:
#   revenue_mult   — multiplicative factor on revenue in window (per branch
#                    listed in branches; null/None means company-wide)
#   cogs_pp        — additive points on the COGS ratio in window
#   opex_mult      — multiplicative factor on operating expenses in window
# Events double as explanatory-RAG ground truth (bilingual, aligned).
# ---------------------------------------------------------------------------
BUSINESS_EVENTS = [
    {
        "event_id": "EVT-2016-001",
        "start_date": "2016-07-01", "end_date": "2016-08-31",
        "event_type": "demand_shock",
        "affected_branch": None,
        "affected_metric": "revenue",
        "title_en": "Summer beverage demand surge 2016",
        "title_ar": "ارتفاع الطلب على المشروبات صيف 2016",
        "explanation_en": (
            "An unusually hot summer lifted beverage sales across all branches "
            "in July-August 2016. Revenue ran above trend while operating "
            "expenses stayed near normal levels."
        ),
        "explanation_ar": (
            "أدى الصيف شديد الحرارة إلى ارتفاع مبيعات المشروبات في جميع الفروع "
            "خلال يوليو وأغسطس 2016. تجاوزت الإيرادات الاتجاه المعتاد بينما "
            "بقيت المصروفات التشغيلية قرب مستوياتها الطبيعية."
        ),
        "expected_effect": "revenue up ~5-8% during event window",
        "effect": {"revenue_mult": 1.06, "cogs_pp": 0.0, "opex_mult": 1.0},
    },
    {
        "event_id": "EVT-2017-001",
        "start_date": "2017-03-01", "end_date": "2017-08-31",
        "event_type": "warehouse_expansion",
        "affected_branch": "BR-RUH",
        "affected_metric": "operating_expenses",
        "title_en": "Riyadh warehouse expansion 2017",
        "title_ar": "توسعة مستودع الرياض 2017",
        "explanation_en": (
            "The Riyadh warehouse was expanded between March and August 2017, "
            "adding storage capacity for wholesale distribution. Operating "
            "expenses ran above trend from extra staffing, equipment rental, "
            "and one-off fit-out costs."
        ),
        "explanation_ar": (
            "تمت توسعة مستودع الرياض بين مارس وأغسطس 2017 لزيادة الطاقة "
            "التخزينية لتوزيع الجملة. تجاوزت المصروفات التشغيلية الاتجاه المعتاد "
            "بسبب التوظيف الإضافي وتأجير المعدات وتكاليف التجهيز لمرة واحدة."
        ),
        "expected_effect": "Riyadh operating expenses up ~10-15% during event window",
        "effect": {"revenue_mult": 1.0, "cogs_pp": 0.0, "opex_mult": 1.12},
    },
    {
        "event_id": "EVT-2018-001",
        "start_date": "2018-01-01", "end_date": "2018-12-31",
        "event_type": "supply_cost_increase",
        "affected_branch": "BR-JED",
        "affected_metric": "operating_expenses",
        "title_en": "Jeddah lease renewal at higher rent 2018",
        "title_ar": "تجديد عقد إيجار جدة بقيمة أعلى 2018",
        "explanation_en": (
            "The Jeddah branch lease was renewed in January 2018 at materially "
            "higher rent. Operating expenses for Jeddah stayed elevated for "
            "the full year with no matching revenue effect."
        ),
        "explanation_ar": (
            "تم تجديد عقد إيجار فرع جدة في يناير 2018 بإيجار أعلى بكثير. بقيت "
            "المصروفات التشغيلية لفرع جدة مرتفعة طوال العام دون أثر مقابل على "
            "الإيرادات."
        ),
        "expected_effect": "Jeddah operating expenses up ~5% across 2018",
        "effect": {"revenue_mult": 1.0, "cogs_pp": 0.0, "opex_mult": 1.05},
    },
    {
        "event_id": "EVT-2019-001",
        "start_date": "2019-01-01", "end_date": "2019-01-01",
        "event_type": "branch_opening",
        "affected_branch": "BR-DMM",
        "affected_metric": "revenue",
        "title_en": "Dammam branch opening",
        "title_ar": "افتتاح فرع الدمام",
        "explanation_en": (
            "The Dammam branch opened on 2019-01-01. Company revenue shows a "
            "permanent level shift from that month, and Dammam ramps up over "
            "its first year. Periods before opening are not applicable and "
            "are never recorded as zero activity."
        ),
        "explanation_ar": (
            "افتُتح فرع الدمام في 2019-01-01. تُظهر إيرادات الشركة قفزة دائمة "
            "في المستوى اعتبارا من ذلك الشهر، ويتدرج فرع الدمام خلال عامه الأول. "
            "الفترات السابقة للافتتاح غير منطبقة ولا تُسجل أبدا كنشاط صفري."
        ),
        "expected_effect": "company revenue level shift up from 2019-01; Dammam ramp over 12 months",
        "effect": {"revenue_mult": 1.0, "cogs_pp": 0.0, "opex_mult": 1.0},
    },
    {
        "event_id": "EVT-2020-001",
        "start_date": "2020-03-01", "end_date": "2020-06-30",
        "event_type": "demand_shock",
        "affected_branch": None,
        "affected_metric": "revenue",
        "title_en": "COVID-period pantry demand shock 2020",
        "title_ar": "صدمة الطلب التمويني خلال جائحة 2020",
        "explanation_en": (
            "During March-June 2020 households stocked packaged foods, lifting "
            "retail revenue well above trend. The surge faded in the second "
            "half of 2020 as buying patterns normalized."
        ),
        "explanation_ar": (
            "خلال مارس-يونيو 2020 خزّنت الأسر الأغذية المعلبة فارتفعت إيرادات "
            "التجزئة كثيرا فوق الاتجاه. تلاشت الموجة في النصف الثاني من 2020 مع "
            "عودة أنماط الشراء إلى طبيعتها."
        ),
        "expected_effect": "revenue up ~12-18% during event window",
        "effect": {"revenue_mult": 1.15, "cogs_pp": 0.0, "opex_mult": 1.0},
    },
    {
        "event_id": "EVT-2020-002",
        "start_date": "2020-04-01", "end_date": "2020-08-31",
        "event_type": "logistics_disruption",
        "affected_branch": None,
        "affected_metric": "logistics_cost",
        "title_en": "2020 logistics disruption and freight spike",
        "title_ar": "اضطراب اللوجستيات وارتفاع الشحن 2020",
        "explanation_en": (
            "Transport restrictions between April and August 2020 raised "
            "freight and last-mile costs. Logistics-driven operating expenses "
            "ran roughly 20-30% above normal while the disruption lasted."
        ),
        "explanation_ar": (
            "أدت قيود النقل بين أبريل وأغسطس 2020 إلى ارتفاع تكاليف الشحن "
            "والتوصيل. تجاوزت المصروفات التشغيلية المرتبطة باللوجستيات المستويات "
            "الطبيعية بنحو 20-30% طوال فترة الاضطراب."
        ),
        "expected_effect": "logistics operating expenses up ~20-30% during event window",
        "effect": {"revenue_mult": 1.0, "cogs_pp": 0.0, "opex_mult": 1.25},
    },
    {
        "event_id": "EVT-2021-001",
        "start_date": "2021-06-01", "end_date": "2022-03-31",
        "event_type": "supply_cost_increase",
        "affected_branch": None,
        "affected_metric": "cogs",
        "title_en": "Supplier cost increase 2021-2022",
        "title_ar": "ارتفاع تكاليف الموردين 2021-2022",
        "explanation_en": (
            "Key food suppliers raised prices from June 2021 through March "
            "2022. The cost-of-sales ratio ran about two to three points above "
            "normal, compressing gross margin until prices were renegotiated."
        ),
        "explanation_ar": (
            "رفع موردو الأغذية الرئيسيون الأسعار من يونيو 2021 حتى مارس 2022. "
            "ارتفعت نسبة تكلفة المبيعات بنحو نقطتين إلى ثلاث نقاط فوق المعتاد، "
            "مما ضغط على هامش الربح حتى أُعيد التفاوض على الأسعار."
        ),
        "expected_effect": "cogs ratio up ~2-3pp during event window",
        "effect": {"revenue_mult": 1.0, "cogs_pp": 0.025, "opex_mult": 1.0},
    },
    {
        "event_id": "EVT-2022-001",
        "start_date": "2022-03-01", "end_date": "2022-05-31",
        "event_type": "promotional_campaign",
        "affected_branch": None,
        "affected_metric": "revenue",
        "title_en": "Ramadan promotional campaign 2022",
        "title_ar": "الحملة الترويجية لرمضان 2022",
        "explanation_en": (
            "A company-wide Ramadan promotion in spring 2022 combined discounts "
            "with heavy marketing. Revenue rose above the seasonal norm while "
            "campaign spending lifted operating expenses in the same quarter."
        ),
        "explanation_ar": (
            "جمعت حملة رمضان الترويجية على مستوى الشركة في ربيع 2022 بين "
            "الخصومات والتسويق المكثف. ارتفعت الإيرادات فوق المعتاد الموسمي بينما "
            "رفع الإنفاق على الحملة المصروفات التشغيلية في الربع نفسه."
        ),
        "expected_effect": "revenue up ~8-10%, opex up ~5% during event window",
        "effect": {"revenue_mult": 1.09, "cogs_pp": 0.0, "opex_mult": 1.05},
    },
    {
        "event_id": "EVT-2023-001",
        "start_date": "2023-09-01", "end_date": "2023-10-31",
        "event_type": "temporary_closure",
        "affected_branch": "BR-JED",
        "affected_metric": "revenue",
        "title_en": "Jeddah partial refit disruption 2023",
        "title_ar": "اضطراب التطوير الجزئي في جدة 2023",
        "explanation_en": (
            "Part of the Jeddah branch closed for refit works during "
            "September-October 2023. Jeddah revenue dipped while the works "
            "ran, then recovered once the branch fully reopened."
        ),
        "explanation_ar": (
            "أُغلق جزء من فرع جدة لأعمال التطوير خلال سبتمبر وأكتوبر 2023. "
            "انخفضت إيرادات جدة أثناء الأعمال ثم تعافت بعد إعادة الافتتاح الكامل."
        ),
        "expected_effect": "Jeddah revenue down ~20% during event window",
        "effect": {"revenue_mult": 0.80, "cogs_pp": 0.0, "opex_mult": 1.0},
    },
    {
        "event_id": "EVT-2024-001",
        "start_date": "2024-01-01", "end_date": "2024-03-31",
        "event_type": "logistics_disruption",
        "affected_branch": None,
        "affected_metric": "logistics_cost",
        "title_en": "Red Sea shipping delays Q1 2024",
        "title_ar": "تأخيرات الشحن في البحر الأحمر في الربع الأول 2024",
        "explanation_en": (
            "Red Sea shipping delays in Q1 2024 raised inbound freight costs "
            "for imported lines. Logistics-driven operating expenses ran above "
            "trend for the quarter."
        ),
        "explanation_ar": (
            "أدت تأخيرات الشحن في البحر الأحمر في الربع الأول من 2024 إلى ارتفاع "
            "تكاليف الشحن الوارد لأصناف الاستيراد. تجاوزت المصروفات التشغيلية "
            "المرتبطة باللوجستيات الاتجاه المعتاد خلال الربع."
        ),
        "expected_effect": "logistics operating expenses up ~15% during event window",
        "effect": {"revenue_mult": 1.0, "cogs_pp": 0.0, "opex_mult": 1.15},
    },
]

# ---------------------------------------------------------------------------
# Transactions — selected financial/operational transactions (NOT every POS
# receipt). Deterministic count per active branch-month; total target ~36k.
# TX-YYYY-NNNNNN: global chronological sequence.
# ---------------------------------------------------------------------------
TRANSACTIONS_PER_BRANCH_MONTH = {"BR-RUH": 135, "BR-JED": 110, "BR-DMM": 90}
# 120*135 + 120*110 + 72*90 = 16,200 + 13,200 + 6,480 = 35,880 (~36k target)

TRANSACTION_TYPES = ("sale", "purchase", "expense", "adjustment")

# Department sampling weights and the type each department mostly produces.
DEPARTMENT_TX_WEIGHTS = {
    "DEP-SALES": 0.45, "DEP-PROC": 0.25, "DEP-LOG": 0.18, "DEP-ADM": 0.12,
}
DEPARTMENT_TX_TYPE = {
    "DEP-SALES": "sale", "DEP-PROC": "purchase",
    "DEP-LOG": "expense", "DEP-ADM": "expense",
}
# Small share of rows that become adjustments regardless of department.
ADJUSTMENT_SHARE = 0.01

CATEGORIES_BY_TYPE = {
    "sale": ("retail", "wholesale"),
    "purchase": ("packaged_foods", "beverages", "household"),
    "expense": ("rent", "salaries", "utilities", "logistics", "marketing",
                "maintenance"),
    "adjustment": ("correction",),
}

# (English, Arabic) counterparty names per transaction type.
COUNTERPARTIES = {
    "sale": [
        ("Walk-in retail customers", "عملاء التجزئة"),
        ("Al-Rawdah Mini Markets", "أسواق الروضة"),
        ("Neighborhood Grocers Group", "مجموعة البقالات"),
        ("Wholesale client - Central Region", "عميل الجملة - المنطقة الوسطى"),
    ],
    "purchase": [
        ("Al-Safi Foods Supplier", "مورد الصافي للأغذية"),
        ("Gulf Beverages Trading", "الخليج لتجارة المشروبات"),
        ("National Packaging Co.", "الشركة الوطنية للتغليف"),
        ("Fresh Harvest Suppliers", "موردو الحصاد الطازج"),
    ],
    "expense": [
        ("Branch landlord", "مالك العقار"),
        ("Saudi Power & Water Utility", "شركة الكهرباء والمياه"),
        ("Payroll - branch staff", "رواتب موظفي الفرع"),
        ("Swift Freight Carriers", "شركة النقل السريع"),
        ("City Marketing Agency", "وكالة المدينة للتسويق"),
        ("Facilities Maintenance Services", "خدمات صيانة المرافق"),
    ],
    "adjustment": [
        ("Internal adjustment", "تسوية داخلية"),
    ],
}

# Category -> plausible counterparty pool (keeps descriptions coherent, e.g.
# salaries are always paid to payroll, never to the landlord). Categories
# absent here fall back to the full per-type pool above.
COUNTERPARTY_BY_CATEGORY = {
    "retail": [("Walk-in retail customers", "عملاء التجزئة")],
    "wholesale": [
        ("Al-Rawdah Mini Markets", "أسواق الروضة"),
        ("Neighborhood Grocers Group", "مجموعة البقالات"),
        ("Wholesale client - Central Region", "عميل الجملة - المنطقة الوسطى"),
    ],
    "rent": [("Branch landlord", "مالك العقار")],
    "salaries": [("Payroll - branch staff", "رواتب موظفي الفرع")],
    "utilities": [("Saudi Power & Water Utility", "شركة الكهرباء والمياه")],
    "logistics": [("Swift Freight Carriers", "شركة النقل السريع")],
    "marketing": [("City Marketing Agency", "وكالة المدينة للتسويق")],
    "maintenance": [
        ("Facilities Maintenance Services", "خدمات صيانة المرافق"),
        ("Branch landlord", "مالك العقار"),
    ],
}
# Lognormal amount parameters (mu, sigma) per transaction type — sampled in
# float, then quantized to Decimal SAR at row-write.
AMOUNT_PARAMS = {
    "sale": (7.2, 0.9),
    "purchase": (8.4, 0.8),
    "expense": (8.0, 1.0),
    "adjustment": (6.5, 0.7),
}

PAYMENT_METHODS = {
    "bank_transfer": 0.60, "cash": 0.25, "credit": 0.15,
}

# Share of purchase transactions that carry an invoice reference. Invoices
# are generated independently (20/company-month); linkage below consumes
# purchase rows deterministically, so the realized share may differ slightly
# and is reported in the manifest.
INVOICE_LINK_TARGET_SHARE = 0.35

# ---------------------------------------------------------------------------
# Invoices — own canonical entity for exact-identifier (BM25) testing.
# INV-YYYY-####: per-year sequence. Flat 15% VAT: total = subtotal + vat.
# ---------------------------------------------------------------------------
INVOICES_PER_COMPANY_MONTH = 20  # 20 * 120 months = 2,400 invoices
INVOICE_BRANCH_WEIGHTS = {"BR-RUH": 0.45, "BR-JED": 0.33, "BR-DMM": 0.22}
INVOICE_SUBTOTAL_PARAMS = (9.0, 0.7)  # lognormal (mu, sigma) in SAR
# Status as-of SNAPSHOT_DATE: recent invoices pending, a small seeded share
# of older ones overdue, the rest paid.
INVOICE_PENDING_CUTOFF = "2024-11-01"  # dated on/after -> pending
INVOICE_OVERDUE_SHARE = 0.03

# ---------------------------------------------------------------------------
# Budgets — quarterly, budget truth ONLY (no stored actuals; actuals are
# derived by aggregating monthly_financials). Two slices:
#   (a) branch x metric for active branches (RUH/JED 40 quarters, DMM 24)
#   (b) company-wide department operating-expenses (4 depts x 40 quarters)
# Budget = prior-year same-quarter actual * (1 + planned growth + noise).
# ---------------------------------------------------------------------------
BUDGET_METRICS = ("revenue", "cogs", "operating_expenses", "net_income")
BUDGET_PLANNED_GROWTH = {
    "revenue": 0.06, "cogs": 0.06,
    "operating_expenses": 0.05, "net_income": 0.08,
}
BUDGET_NOISE_SD = 0.02  # Gaussian noise on the growth factor

# Department shares of company operating expenses, used ONLY for the
# company-wide department budget slice (slice b). Actuals stay derived.
DEPT_OPEX_SHARES = {
    "DEP-SALES": 0.20, "DEP-PROC": 0.15,
    "DEP-LOG": 0.35, "DEP-ADM": 0.30,
}

# ---------------------------------------------------------------------------
# Annual balance sheet — one company-level row per year (10 rows).
# Components scale from that year's company revenue with ratios below, plus
# capex step events (2017 warehouse expansion, 2019 Dammam fit-out).
# equity is the plug: total_assets - total_liabilities (invariant enforced).
# ---------------------------------------------------------------------------
BALANCE_SHEET_RATIOS = {
    "cash": 0.060,
    "accounts_receivable": 0.090,
    "inventory": 0.110,
    "other_current_assets": 0.020,
    "ppe_base": 0.280,  # property_and_equipment, before capex steps
    "accounts_payable": 0.100,
    "debt_base": 0.180,  # before capex-linked borrowing
    "other_liabilities": 0.030,
}
# (year, ppe_step, debt_step): one-off additions in SAR for capex years.
BALANCE_SHEET_CAPEX_STEPS = {
    2017: (2500000.0, 1500000.0),  # Riyadh warehouse expansion
    2019: (4000000.0, 2500000.0),  # Dammam branch fit-out
}
# Small deterministic drift so the sheet is not a pure revenue multiple.
BALANCE_SHEET_NOISE_CV = 0.015

# ---------------------------------------------------------------------------
# Output files (CSV preferred) + manifest.
# ---------------------------------------------------------------------------
OUTPUT_FILES = (
    "branches.csv",
    "departments.csv",
    "monthly_financials.csv",
    "transactions.csv",
    "budgets.csv",
    "invoices.csv",
    "business_events.csv",
    "annual_balance_sheet.csv",
)
MANIFEST_FILE = "manifest.json"
