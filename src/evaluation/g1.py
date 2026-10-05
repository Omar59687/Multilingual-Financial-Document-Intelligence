"""G1 extraction-measurement gold derivation and scoring (Phase 3, bounded slice).

Evaluation-side only (never imported by product code): derives
field-level gold from canonical truth + DEV ground-truth metadata and
scores extracted ``FinancialRecord``s per Evaluation Plan §3
(field correct iff metric + value + period + source ALL match; 2dp
normalization; ±0.5% tolerance ONLY for OCR-noise slices; identifiers
exact; every numeric miss listed expected-vs-extracted).

Gold rule: expected fields are truth-recomputed from canonical CSVs
(origin-1), scoped by each DEV document's rendered content
(origin-2 fixtures + spec). Ground-truth JSON values are NEVER copied
as gold — canonical rows are re-read and re-aggregated here.
Unverifiable/prose/visual/derived/budget grains are classified
out-of-scope with rationale (EVAL §2: unverified items excluded from
reported scores) and listed for human verification (origin-3).

Field key: (document_id, metric, period, branch-or-None, value-2dp).
Branch is included beyond the §3 minimum: a right-number-wrong-branch
extraction must fail (branch comparisons depend on it). "Source" is
the document: document_id is the stable traceability unit shared
across ingestion (document_id_for), gold (dev_id), and extraction —
page/element alignment is not deterministically available across all
grains (statement cells vs CSV rows vs grid pairs), so finer source
identity cannot be frozen without inventing alignment rules.
Residual: a same-valued swap between two same-grain rows is
indistinguishable to any value-based gold (documented; architect to
confirm this reading at the threshold gate).

NoLLM, no training, no threshold decisions here (thresholds TBD →
human gate). 2024 holdout: measurement never trains; threshold-setting
must not overfit DEV-007 (flagged in the report).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any

__all__ = [
    "PNL_METRICS",
    "BS_COLUMNS",
    "TX_FIELDS",
    "INV_FIELDS",
    "norm_2dp",
    "load_canonical",
    "expected_for_doc",
    "score_doc",
    "aggregate",
]

PNL_METRICS: tuple[str, ...] = (
    "revenue", "cogs", "gross_profit", "operating_expenses",
    "operating_profit", "other_income", "finance_costs",
    "tax_expense", "net_income",
)

#: Canonical annual_balance_sheet columns in file order; the property
#: column maps to the frozen ``property_plant_equipment`` metric.
BS_COLUMNS: tuple[tuple[str, str], ...] = (
    ("cash", "cash"),
    ("accounts_receivable", "accounts_receivable"),
    ("inventory", "inventory"),
    ("other_current_assets", "other_current_assets"),
    ("property_and_equipment", "property_plant_equipment"),
    ("total_assets", "total_assets"),
    ("accounts_payable", "accounts_payable"),
    ("debt", "debt"),
    ("other_liabilities", "other_liabilities"),
    ("total_liabilities", "total_liabilities"),
    ("equity", "equity"),
)

TX_FIELDS: tuple[str, ...] = ("amount", "tax_amount", "total_amount")
INV_FIELDS: tuple[str, ...] = ("subtotal", "vat", "total")

_CENT = Decimal("0.01")


def norm_2dp(raw: Any) -> str:
    """Normalize a canonical money string to exact 2dp (raises on bad)."""
    try:
        dec = Decimal(str(raw).strip().replace(",", ""))
    except (InvalidOperation, ValueError, AttributeError):
        raise ValueError(f"gold money unparseable: {raw!r}") from None
    if not dec.is_finite():
        raise ValueError(f"gold money non-finite: {raw!r}")
    return format(dec.quantize(_CENT, rounding=ROUND_HALF_UP), "f")


def load_canonical(canonical_dir: str | Path) -> dict[str, Any]:
    """Load canonical CSVs into keyed lookups (origin-1 truth)."""
    base = Path(canonical_dir)
    monthly: dict[tuple[str, str], dict] = {}
    with open(base / "monthly_financials.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            monthly[(row["period"], row["branch_id"])] = row
    invoices: dict[str, dict] = {}
    with open(base / "invoices.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            invoices[row["invoice_id"]] = row
    txns: dict[str, dict] = {}
    with open(base / "transactions.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            txns[row["transaction_id"]] = row
    balance: dict[str, dict] = {}
    with open(base / "annual_balance_sheet.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            balance[row["year"]] = row
    budgets: dict[tuple[str, str, str], dict] = {}
    with open(base / "budgets.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            budgets[(row["period"], row["branch_id"], row["metric"])] = row
    return {"monthly": monthly, "invoices": invoices, "txns": txns,
            "balance": balance, "budgets": budgets}


def _field(doc: str, metric: str, period: str, branch: str | None, value: str) -> tuple:
    return (doc, metric, period, branch, norm_2dp(value))


def expected_for_doc(dev_id: str, source_ids: list, canon: dict) -> tuple[list, list]:
    """Derive (expected_fields, out_of_scope) for one DEV document.

    Expected is a LIST (multiset): identical field tuples from distinct
    source rows are distinct facts (two identical CSV rows must both be
    extractable without a duplicate penalty). Raises KeyError on
    missing canonical rows (gold gaps are harness bugs, never silent).
    """
    monthly = canon["monthly"]
    expected: list[tuple] = []
    oos: list[tuple[str, str, str]] = []

    def monthly_sums(periods: list[str], branches: list[str]) -> dict[str, Decimal]:
        totals: dict[str, Decimal] = defaultdict(lambda: Decimal("0.00"))
        for p in periods:
            for b in branches:
                row = monthly[(p, b)]
                for m in PNL_METRICS:
                    totals[m] += Decimal(row[m])
        return totals

    if dev_id == "DEV-001":
        branches = ["BR-RUH", "BR-JED", "BR-DMM"]
        months_23 = [f"2023-{m:02d}-01" for m in range(1, 13)]
        months_22 = [f"2022-{m:02d}-01" for m in range(1, 13)]
        for year, months in (("2023", months_23), ("2022", months_22)):
            for m, total in monthly_sums(months, branches).items():
                expected.append((dev_id, m, year, None, norm_2dp(total)))
        for b in branches:
            rev = sum(Decimal(monthly[(p, b)]["revenue"]) for p in months_23)
            expected.append((dev_id, "revenue", "2023", b, norm_2dp(rev)))
        oos.append((dev_id, "derived-yoy", "YoY Change% column is derived (R-computed), not a field"))
    elif dev_id == "DEV-002":
        for m in range(1, 13):
            p = f"2022-{m:02d}-01"
            expected.append(_field(dev_id, "operating_expenses", p, "BR-JED",
                                   monthly[(p, "BR-JED")]["operating_expenses"]))
        oos.append((dev_id, "derived-annual-total",
                    "headline annual opex total is a derived 12-month sum (G4 territory, not a field)"))
        oos.append((dev_id, "derived-categories", "6 category aggregates are derived sums (G4 territory, not fields)"))
        oos.append((dev_id, "non-rendered-txns", "372 txn rows feed aggregates only (explicit partial-extract design)"))
    elif dev_id == "DEV-003":
        inv = canon["invoices"]["INV-2023-0106"]
        for m in INV_FIELDS:
            expected.append(_field(dev_id, m, inv["invoice_date"], inv["branch_id"], inv[m]))
    elif dev_id == "DEV-004":
        txn = canon["txns"]["TX-2019-013526"]
        for m in TX_FIELDS:
            expected.append(_field(dev_id, m, txn["date"], txn["branch_id"], txn[m]))
    elif dev_id == "DEV-005":
        oos.append((dev_id, "prose-numbers", "commentary prose figures have no extraction route (human-verify backlog)"))
    elif dev_id == "DEV-006":
        for tid in source_ids:
            row = canon["txns"][tid]
            for m in TX_FIELDS:
                expected.append(_field(dev_id, m, row["date"], row["branch_id"], row[m]))
    elif dev_id == "DEV-007":
        for tid in source_ids:
            row = canon["txns"][tid]
            for m in TX_FIELDS:
                expected.append(_field(dev_id, m, row["date"], row["branch_id"], row[m]))
    elif dev_id == "DEV-008":
        oos.append((dev_id, "visual-annual-10y",
                    "chart annual revenues are M-rounded on-chart (approximate-only grain, human-verify backlog)"))
    elif dev_id == "DEV-009":
        # Frozen visual-benchmark evidence (ocrbench/metrics docstring):
        # dashboard strings are M-abbreviated DISPLAY values; exact
        # canonical sums are NOT visible (except the variance panel,
        # which is derived budget-grain with no route). Exact-match
        # field scoring against invisible precision would demand hidden
        # digits in reverse — procedurally unsound. Approximate visual
        # evaluation belongs to the visual benchmark, not G1-field-exact.
        oos.append((dev_id, "visual-kpi-rounded",
                    "4 KPI cards are M-abbreviated displays (approximate-only grain, human-verify backlog)"))
        oos.append((dev_id, "visual-branch-bars-rounded",
                    "branch bars are M-abbreviated displays (approximate-only grain, human-verify backlog)"))
        oos.append((dev_id, "budget-grain", "budget_total/variance have no budget-table route (no budget tables in corpus)"))
    elif dev_id == "DEV-010":
        row = canon["balance"]["2020"]
        for col, metric in BS_COLUMNS:
            expected.append(_field(dev_id, metric, "2020", None, row[col]))
        oos.append((dev_id, "context-2019", "2019 comparatives are unevaluated context (spec: NOT rendered)"))
    else:
        raise ValueError(f"unknown DEV document: {dev_id}")
    void = [f for f in expected if f[0] != dev_id]
    assert not void
    return expected, oos


def _record_key(rec: Any) -> tuple:
    """Extracted record → comparable field key (value normalized 2dp)."""
    metric = getattr(rec.metric, "value", rec.metric)
    value = rec.value if isinstance(rec.value, Decimal) else Decimal(str(rec.value))
    return (rec.document_id, str(metric), rec.period,
            rec.branch_id, norm_2dp(value))


def score_doc(expected: list, extracted: list, ocr_noise: bool = False) -> dict:
    """Score one document: TP/FP/FN, P/R/F1, numeric misses.

    Multiset matching on full field tuples (doc, metric, period,
    branch, value): identical tuples from distinct source rows are
    distinct facts, so occurrence counts decide (TP += min, extras are
    FP, shortfalls are FN). This is required for transaction grains,
    where dozens of distinct facts share (metric, period, branch) and
    differ only by value.
    Miss classification (listing only — counts already decided):
    unmatched extracted whose (doc, metric, period, branch) has NO
    expected value → "invented" (first) / "duplicate" (rest);
    unmatched extracted whose partial key HAS expected value(s) →
    "numeric-mismatch" with expected counterpart(s); over-counts of
    matched tuples → "duplicate"; unmatched expected → "missed".
    ``ocr_noise`` selects the ±0.5% relative band for scan-noise
    slices (EVAL §3 restricted use): a within-band value counts TP.
    """
    from collections import Counter
    exp_counts: Counter = Counter(expected)
    ext_tuples: list[tuple] = []
    misses: list[dict] = []
    for rec in extracted:
        try:
            ext_tuples.append(_record_key(rec))
        except Exception:
            misses.append({"kind": "unreadable-record", "detail": repr(rec)[:200]})
    ext_counts: Counter = Counter(ext_tuples)
    exp_partials: dict[tuple, set] = {}
    for doc, metric, period, branch, value in exp_counts:
        exp_partials.setdefault((doc, metric, period, branch), set()).add(value)
    tp = 0
    fp = len(misses)  # unreadable records already listed above
    fn = 0
    band_matches = 0
    # Phase 1: exact multiset intersection (occurrence-counted).
    rem_exp: Counter = Counter()
    rem_ext: Counter = Counter()
    for full, n in ext_counts.items():
        m = min(n, exp_counts.get(full, 0))
        tp += m
        if n - m:
            rem_ext[full] = n - m
    for full, n in exp_counts.items():
        if n - ext_counts.get(full, 0):
            rem_exp[full] = n - ext_counts.get(full, 0)
    # Phase 2 (scan-noise slices only): maximum-cardinality one-to-one
    # band matching per partial key (Kuhn augmenting-path; distance only
    # breaks ties deterministically). Greedy-nearest is NOT sufficient:
    # it can strand matchable occurrences. Each occurrence matches at
    # most once — no double rescue, no negative counts.
    if ocr_noise:
        from collections import defaultdict
        exp_by_partial: dict = defaultdict(list)
        ext_by_partial: dict = defaultdict(list)
        for full, n in rem_exp.items():
            exp_by_partial[full[:4]].extend([full] * n)
        for full, n in rem_ext.items():
            ext_by_partial[full[:4]].extend([full] * n)
        for partial in sorted(exp_by_partial, key=str):
            exp_nodes = sorted(exp_by_partial[partial], key=str)
            ext_nodes = sorted(ext_by_partial.get(partial, []), key=str)
            adj: list[list[tuple[int, Decimal]]] = [[] for _ in exp_nodes]
            for i, efull in enumerate(exp_nodes):
                try:
                    denom = abs(Decimal(efull[4]))
                except (InvalidOperation, ValueError):
                    continue
                if denom == 0:
                    continue
                for j, xfull in enumerate(ext_nodes):
                    try:
                        rel = abs(Decimal(xfull[4]) - Decimal(efull[4])) / denom
                    except (InvalidOperation, ValueError):
                        continue
                    if rel <= Decimal("0.005"):
                        adj[i].append((j, rel))
                adj[i].sort(key=lambda t: (t[1], str(ext_nodes[t[0]])))
            match_ext: dict[int, int] = {}

            def _augment(i: int, seen: set[int]) -> bool:
                for j, _rel in adj[i]:
                    if j in seen:
                        continue
                    seen.add(j)
                    if j not in match_ext or _augment(match_ext[j], seen):
                        match_ext[j] = i
                        return True
                return False

            for i in range(len(exp_nodes)):
                _augment(i, set())
            for j, i in match_ext.items():
                tp += 1
                band_matches += 1
                rem_exp[exp_nodes[i]] -= 1
                rem_ext[ext_nodes[j]] -= 1
        rem_exp = +rem_exp
        rem_ext = +rem_ext
    # Remainders: classify for the miss listing (counts decided above).
    for full, n in rem_ext.items():
        if full in exp_counts:
            kinds = ["duplicate"] * n  # over-count of a matched value
        elif full[:4] not in exp_partials:
            kinds = ["invented"] + ["duplicate"] * (n - 1)
        else:
            kinds = ["numeric-mismatch"] * n
        for kind in kinds:
            fp += 1
            entry: dict = {"kind": kind, "key": list(full[:4])}
            if kind == "numeric-mismatch":
                entry["expected"] = sorted(exp_partials[full[:4]])
                entry["extracted"] = full[4]
            else:
                entry["value"] = full[4]
            misses.append(entry)
    for full, n in rem_exp.items():
        for _ in range(n):
            fn += 1
            misses.append({"kind": "missed", "key": list(full[:4]),
                           "expected": full[4]})
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    active_mismatch = sum(1 for m in misses if m["kind"] == "numeric-mismatch")
    exacta = tp / (tp + active_mismatch) if (tp + active_mismatch) else 1.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1, "numeric_accuracy": exacta,
            "band_matches": band_matches, "misses": misses}


def aggregate(per_doc: dict[str, dict]) -> dict:
    """Micro-average P/R/F1 over documents (plus min-slice support)."""
    tp = sum(d["tp"] for d in per_doc.values())
    fp = sum(d["fp"] for d in per_doc.values())
    fn = sum(d["fn"] for d in per_doc.values())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1}
