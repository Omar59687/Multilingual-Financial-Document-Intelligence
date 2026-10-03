"""Phase 3 — deterministic extraction validation (I5).

Fail-closed, deterministic, provenance-safe validators for structured
financial extraction. Mirrors the canonical ``src/dataset/validators.py``
fail-closed style and the Phase 2 no-fabrication policy
(``docs/PHASE_2_OCR_VISION_EXIT.md`` §7, ``src/ocrbench/canonical.py``).

Scope owned by I5 (parallel Phase-3 workstream):
  - ``src/extraction/validation.py`` (this file)
  - ``tests/extraction/test_phase3_extraction.py``

What this module enforces:

1. Sign rules
   - NEVER negative: revenue, cogs, operating_expenses, other_income,
     finance_costs, tax_expense, subtotal, vat, amount, tax_amount,
     total_amount, total, budget / budget_amount, actual.
   - MAY be negative: gross_profit, operating_profit, net_income,
     variance (and derived variance_pct).
   - Transactions convention: amounts are magnitudes (>= 0) plus a
     separate ``transaction_type``; refunds/adjustments are expressed
     by type, never by a negative amount. A negative ``amount`` is
     ``E_SIGN`` even when the transaction type is ``refund``.

2. 2dp SAR money
   - Values are coerced via ``Decimal(str(value))`` (commas/spaces as
     thousands separators are stripped; ``Decimal`` inputs pass
     through; ``bool`` is rejected as non-money).
   - Valid money has exponent exactly -2 (exactly two decimal places),
     i.e. ``Decimal("100.00")`` passes, ``"100"`` / ``"100.0"`` /
     ``"100.123"`` fail, and ``abs(value) < 1e12``.
   - Rounding helper uses ``ROUND_HALF_UP`` to 2dp where rounding is
     explicitly required (variance / pct derivation); validation
     itself never silently rounds — non-2dp input is ``E_MONEY``.

3. Precision safety (no fabrication)
   - Precision classes mirror Phase 2 canonical policy:
     ``exact-visible`` (fully specified 2dp figure), ``display-rounded``
     (suffixed/coarse display, hidden precision unknowable),
     ``unknown`` (unparseable display normalizes to None, never zero).
   - A record claiming exact equality (``claims_exact is True`` or any
     ``precision == "exact-visible"``) while its ``display_value``
     normalizes to rounded/unparseable is
     ``E_PRECISION:exact_claim_on_rounded`` (or
     ``..._on_unparseable`` / ``..._on_missing_display``). Display-rounded
     values must stay rounded; they never gain fabricated precision.
   - Unknown metric labels are passed through, never invented into
     canonical keys: ``validate_record(..., strict=False)`` (default)
     emits no error for them (see ``warnings_for_record`` for the
     ``W_LABEL`` pass-through warning); ``strict=True`` escalates to
     ``E_LABEL:unknown_label``.

4. Period / branch (LOCAL checks; periods.py unification note)
   - CHOICE (documented per I5 contract): this module implements LOCAL
     lightweight regex checks and performs NO import of
     ``src/extraction/periods.py`` — not even at module top level —
     to avoid import cycles during parallel Phase-3 integration.
     The future orchestrator / I-owner of ``periods.py`` will unify
     period parsing in one place; until then this file is the
     authoritative extraction-side implementation with stable
     ``E_PERIOD`` / ``E_BRANCH`` strings. A deferred
     ``from .periods import parse_period`` inside a function body
     remains the approved migration path (optional import with local
     regex fallback) precisely because it avoids the top-level cycle.
   - Accepted period shapes: ``YYYY-MM-01`` (monthly), ``YYYY-Qn``
     (quarterly budgets), ``YYYY-MM-DD`` (transaction/invoice dates),
     ``YYYY`` (annual). Year must be 2015–2024 inclusive (canonical
     dataset range); months/days are range-checked (incl. leap years).
   - Dammam (``BR-DMM``) opened 2019-01-01: any Dammam record with a
     period before 2019-01-01 (``YYYY-MM-01``/``YYYY-MM-DD`` string
     compare, ``YYYY-Qn`` quarter compare, ``YYYY`` year compare) is
     ``E_BRANCH:dammam_pre_opening``. Pre-opening Dammam activity is
     absent/NULL upstream — it must never validate as zero activity.
   - ``branch_id``, when present, must be one of
     ``BR-RUH`` / ``BR-JED`` / ``BR-DMM``.

5. Reconciliation helpers (tolerance 0.01, never divide by zero)
   - ``check_r1_r3(...)`` verifies R1/R2/R3 within abs 0.01.
   - ``check_invoice_total(subtotal, vat, total)`` verifies
     total == subtotal + vat within 0.01.
   - ``budget_variance(actual, budget)`` returns
     ``{"variance", "variance_pct", "budget_zero"}`` with
     variance = actual - budget (2dp ROUND_HALF_UP) and pct = None
     when budget == 0 (never divides by zero).

6. Provenance completeness (Rule 10 traceability)
   - ``provenance`` dict (or top-level equivalent keys) must carry
     non-empty ``document_id`` + ``source_label`` + ``display_value``
     + ``precision``. ``precision`` must be one of
     ``exact-visible`` / ``display-rounded`` / ``unknown``.
   - ``page``, when present, must be an int >= 1.
   - ``branch_id``, when present, must be a valid enum (see §4).

7. Determinism
   - No randomness, no ``now()``/wall-clock, no iteration over
     unordered sets for error construction. ``validate_record``
     returns errors in a fixed stage order
     (provenance → branch → period → money → sign → precision →
     reconciliation → label); within money, a fixed field order is
     used. ``validate_batch`` preserves input order.

8. Compatibility
   - Stdlib + ``decimal`` only. No top-level import of pydantic,
     schemas, provenance/store, ``periods.py``, ingestion, ocrbench
     scorers, or frozen truth. Inputs are plain dicts (duck-typing)
     so tests pass before integration. This file never modifies
     ``src/dataset/validators.py``, ingestion, scorers, or fixtures.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

__all__ = [
    "BRANCH_IDS",
    "NONNEGATIVE_FIELDS",
    "MAY_BE_NEGATIVE_FIELDS",
    "KNOWN_METRICS",
    "MONEY_FIELDS_ORDER",
    "MAX_ABS_MONEY",
    "MONEY_TOL",
    "is_valid_money",
    "is_valid_period",
    "is_valid_branch",
    "is_valid_provenance",
    "is_valid_record",
    "check_r1_r3",
    "check_invoice_total",
    "check_transaction_total",
    "budget_variance",
    "check_budget_variance",
    "check_precision_claim",
    "warnings_for_record",
    "validate_record",
    "validate_batch",
]

# ---------------------------------------------------------------------------
# Constants (fixed order where determinism matters)
# ---------------------------------------------------------------------------

BRANCH_IDS = ("BR-RUH", "BR-JED", "BR-DMM")

#: Money fields that must never be negative (extraction-side mirror of the
#: canonical truth invariants). Fixed tuple => deterministic iteration order.
MONEY_FIELDS_ORDER = (
    "revenue",
    "cogs",
    "gross_profit",
    "operating_expenses",
    "operating_profit",
    "other_income",
    "finance_costs",
    "tax_expense",
    "net_income",
    "subtotal",
    "vat",
    "total",
    "amount",
    "tax_amount",
    "total_amount",
    "budget",
    "budget_amount",
    "actual",
    "variance",
)

NONNEGATIVE_FIELDS = frozenset({
    "revenue",
    "cogs",
    "operating_expenses",
    "other_income",
    "finance_costs",
    "tax_expense",
    "subtotal",
    "vat",
    "amount",
    "tax_amount",
    "total_amount",
    "total",
    "budget",
    "budget_amount",
    "actual",
})

#: Money fields explicitly allowed to be negative.
MAY_BE_NEGATIVE_FIELDS = frozenset({
    "gross_profit",
    "operating_profit",
    "net_income",
    "variance",
})

#: Labels this validator recognises. Anything else is "unknown" and is
#: passed through (warning) rather than invented into a canonical key.
KNOWN_METRICS = frozenset(set(MONEY_FIELDS_ORDER) | {"variance_pct"})

PRECISION_VALUES = ("exact-visible", "display-rounded", "unknown")

MAX_ABS_MONEY = Decimal("1000000000000")  # 1e12, exclusive bound
MONEY_TOL = Decimal("0.01")
_TWO_DP = Decimal("0.01")

_YEAR_MIN = 2015
_YEAR_MAX = 2024
_DMM_OPEN_DATE = "2019-01-01"
_DMM_OPEN_YEAR = 2019
_DMM_OPEN_QUARTER = (2019, 1)

_RE_MONTH = re.compile(r"^(\d{4})-(\d{2})-01$")
_RE_QUARTER = re.compile(r"^(\d{4})-[Qq]([1-4])$")
_RE_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_RE_YEAR = re.compile(r"^(\d{4})$")

_CURRENCY_TOKENS = ("sar", "$", "usd", "aed", "egp", "qar", "kwd",
                    "ر.س", "ر. س", "ريال")
_MAGNITUDE = {"k": 3, "m": 6, "b": 9}


# ---------------------------------------------------------------------------
# Small Decimal helpers
# ---------------------------------------------------------------------------

def _to_decimal(value) -> Decimal | None:
    """Coerce ``value`` to Decimal, or None when unparseable.

    Rules: None -> None; bool -> None (bools are not money);
    Decimal -> as-is (non-finite -> None); int -> Decimal(int);
    float -> Decimal(str(float)) (non-finite -> None);
    str -> strip, drop commas/spaces/underscores/NBSP, then Decimal
    (non-finite -> None).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value if value.is_finite() else None
    if isinstance(value, int):
        try:
            dec = Decimal(value)
        except (InvalidOperation, ValueError):
            return None
        return dec if dec.is_finite() else None
    if isinstance(value, float):
        try:
            dec = Decimal(str(value))
        except (InvalidOperation, ValueError):
            return None
        return dec if dec.is_finite() else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        text = (text.replace(",", "").replace(" ", "").replace("_", "")
                    .replace("\u00a0", ""))
        if text[:1] in ("+", "-"):
            sign, body = text[0], text[1:]
            if not body:
                return None
            text = sign + body
        try:
            dec = Decimal(text)
        except (InvalidOperation, ValueError):
            return None
        return dec if dec.is_finite() else None
    # Any other type (list/dict/...) is not money.
    return None


def _is_2dp(value: Decimal) -> bool:
    return value.as_tuple().exponent == -2


def is_valid_money(value) -> bool:
    """True iff ``value`` is SAR money: exactly 2dp and abs < 1e12."""
    dec = _to_decimal(value)
    if dec is None:
        return False
    if not _is_2dp(dec):
        return False
    return abs(dec) < MAX_ABS_MONEY


def _quantize_2dp(value: Decimal) -> Decimal:
    return value.quantize(_TWO_DP, rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Branch helpers
# ---------------------------------------------------------------------------

def is_valid_branch(branch_id) -> bool:
    """True iff ``branch_id`` is a known enum value."""
    return isinstance(branch_id, str) and branch_id in BRANCH_IDS


def _check_branch(branch_id) -> list[str]:
    if branch_id is None:
        return []
    if not is_valid_branch(branch_id):
        return [f"E_BRANCH:invalid_branch_id:{branch_id!r} "
                f"expected one of {list(BRANCH_IDS)}"]
    return []


# ---------------------------------------------------------------------------
# Period helpers (LOCAL implementation — see module docstring §4)
# ---------------------------------------------------------------------------

def _days_in_month(year: int, month: int) -> int:
    if month in (1, 3, 5, 7, 8, 10, 12):
        return 31
    if month in (4, 6, 9, 11):
        return 30
    leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    return 29 if leap else 28


def _check_period_format(period) -> list[str]:
    """Validate period shape + calendar range. Returns E_PERIOD errors."""
    if period is None:
        return []
    if not isinstance(period, str) or not period.strip():
        return ["E_PERIOD:missing_or_not_string: period must be a "
                "non-empty string"]
    text = period.strip()
    m = _RE_MONTH.match(text)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        if not (_YEAR_MIN <= year <= _YEAR_MAX):
            return [f"E_PERIOD:out_of_range:{text!r} year must be "
                    f"{_YEAR_MIN}-{_YEAR_MAX}"]
        if not (1 <= month <= 12):
            return [f"E_PERIOD:bad_month:{text!r}"]
        return []
    m = _RE_QUARTER.match(text)
    if m:
        year = int(m.group(1))
        if not (_YEAR_MIN <= year <= _YEAR_MAX):
            return [f"E_PERIOD:out_of_range:{text!r} year must be "
                    f"{_YEAR_MIN}-{_YEAR_MAX}"]
        return []
    m = _RE_DATE.match(text)
    if m:
        year, month, day = (int(m.group(1)), int(m.group(2)),
                            int(m.group(3)))
        if not (_YEAR_MIN <= year <= _YEAR_MAX):
            return [f"E_PERIOD:out_of_range:{text!r} year must be "
                    f"{_YEAR_MIN}-{_YEAR_MAX}"]
        if not (1 <= month <= 12):
            return [f"E_PERIOD:bad_month:{text!r}"]
        if not (1 <= day <= _days_in_month(year, month)):
            return [f"E_PERIOD:bad_day:{text!r}"]
        return []
    m = _RE_YEAR.match(text)
    if m:
        year = int(m.group(1))
        if not (_YEAR_MIN <= year <= _YEAR_MAX):
            return [f"E_PERIOD:out_of_range:{text!r} year must be "
                    f"{_YEAR_MIN}-{_YEAR_MAX}"]
        return []
    return [f"E_PERIOD:bad_format:{text!r} expected YYYY-MM-01, YYYY-Qn, "
            f"YYYY-MM-DD, or YYYY"]


def is_valid_period(period) -> bool:
    """True iff ``period`` matches a known shape and calendar range."""
    if period is None:
        return False
    return _check_period_format(period) == []


def _is_dammam_pre_opening(branch_id, period) -> bool:
    """True iff a Dammam record predates the 2019-01-01 opening.

    Returns False for unparseable periods (format errors are reported
    separately as E_PERIOD) and for non-Dammam branches.
    """
    if branch_id != "BR-DMM":
        return False
    if not isinstance(period, str):
        return False
    text = period.strip()
    m = _RE_MONTH.match(text)
    if m:
        return text < _DMM_OPEN_DATE
    m = _RE_DATE.match(text)
    if m:
        return text < _DMM_OPEN_DATE
    m = _RE_QUARTER.match(text)
    if m:
        return (int(m.group(1)), int(m.group(2))) < _DMM_OPEN_QUARTER
    m = _RE_YEAR.match(text)
    if m:
        return int(m.group(1)) < _DMM_OPEN_YEAR
    return False


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------

_PROVENANCE_REQUIRED = ("document_id", "source_label", "display_value",
                        "precision")


def _provenance_of(rec: dict) -> dict:
    prov = rec.get("provenance")
    if isinstance(prov, dict):
        return prov
    # Top-level fallback: accept document_id/source_label/display_value/
    # precision as flat keys so plain-dict callers are not forced into
    # nesting before integration.
    flat = {k: rec.get(k) for k in _PROVENANCE_REQUIRED if k in rec}
    if flat:
        return flat
    return {}


def is_valid_provenance(provenance) -> bool:
    """True iff provenance carries all required non-empty fields."""
    if not isinstance(provenance, dict):
        return False
    for key in _PROVENANCE_REQUIRED:
        val = provenance.get(key)
        if not isinstance(val, str) or not val.strip():
            return False
    if provenance.get("precision") not in PRECISION_VALUES:
        return False
    page = provenance.get("page", None)
    if page is not None and (isinstance(page, bool)
                             or not isinstance(page, int) or page < 1):
        return False
    branch = provenance.get("branch_id", None)
    if branch is not None and not is_valid_branch(branch):
        return False
    # Top-level page/branch are checked in validate_record as well; keep
    # this helper focused on the provenance dict itself.
    return True


def _check_provenance(rec: dict) -> list[str]:
    errors: list[str] = []
    prov = rec.get("provenance")
    if prov is not None and not isinstance(prov, dict):
        return ["E_PROVENANCE:provenance_not_dict: provenance must be "
                "a dict"]
    source = prov if isinstance(prov, dict) else rec
    # If neither nested nor flat keys exist, report each missing field so
    # the caller knows exactly what to supply (deterministic order).
    for key in _PROVENANCE_REQUIRED:
        val = source.get(key)
        if not isinstance(val, str) or not val.strip():
            errors.append(f"E_PROVENANCE:missing_{key}: provenance "
                          f"requires non-empty {key}")
    precision = source.get("precision")
    if (isinstance(precision, str) and precision.strip()
            and precision not in PRECISION_VALUES):
        errors.append(f"E_PROVENANCE:bad_precision:{precision!r} "
                      f"expected one of {list(PRECISION_VALUES)}")
    # page >= 1 when present (check both nested provenance and top level).
    for holder in (prov if isinstance(prov, dict) else None, rec):
        if holder is None:
            continue
        if "page" in holder and holder["page"] is not None:
            page = holder["page"]
            if (isinstance(page, bool) or not isinstance(page, int)
                    or page < 1):
                errors.append(f"E_PROVENANCE:bad_page:{page!r} "
                              f"page must be int >= 1")
                break  # one page error is enough; keep order stable
    # branch_id enum when present (nested or top level).
    branches_seen: list[str] = []
    for holder in (prov if isinstance(prov, dict) else None, rec):
        if holder is None:
            continue
        if "branch_id" in holder and holder["branch_id"] is not None:
            branches_seen.append(holder["branch_id"])
    for branch in branches_seen:
        if not is_valid_branch(branch):
            errors.append(f"E_BRANCH:invalid_branch_id:{branch!r} "
                          f"expected one of {list(BRANCH_IDS)}")
            break
    return errors


# ---------------------------------------------------------------------------
# Display / precision-safety helpers (mirror canonical.py, local copy)
# ---------------------------------------------------------------------------

def _normalize_display_value(display_value) -> dict:
    """Normalize ONE displayed string (local mirror of canonical policy).

    Returns ``{"normalized_value": Decimal|None, "precision": str}``
    with precision in exact-visible / display-rounded / unknown.
    Unparseable input yields normalized None + unknown — never zero.
    """
    if not isinstance(display_value, str) or not display_value.strip():
        return {"normalized_value": None, "precision": "unknown"}
    text = display_value.strip()
    sign = 1
    if text[:1] in ("+", "-"):
        sign = -1 if text[0] == "-" else 1
        text = text[1:].strip()
        if not text:
            return {"normalized_value": None, "precision": "unknown"}
    lowered = text.casefold()
    for token in _CURRENCY_TOKENS:
        lowered = lowered.replace(token, "")
    lowered = lowered.replace("\u00a0", " ").strip()
    magnitude = 0
    rounded = False
    if lowered and lowered[-1] in _MAGNITUDE:
        magnitude = _MAGNITUDE[lowered[-1]]
        rounded = True  # suffixed dashboard display: hidden precision lost
        lowered = lowered[:-1].strip()
    lowered = lowered.replace(",", "").replace(" ", "")
    if not lowered:
        return {"normalized_value": None, "precision": "unknown"}
    try:
        number = Decimal(lowered)
    except InvalidOperation:
        return {"normalized_value": None, "precision": "unknown"}
    if not number.is_finite():
        return {"normalized_value": None, "precision": "unknown"}
    scaled = number * (Decimal(10) ** magnitude) * sign
    if not scaled.is_finite():
        return {"normalized_value": None, "precision": "unknown"}
    if not rounded:
        # Only a fully-specified 2dp figure is exact-visible; anything
        # coarser or finer stays display-rounded.
        target = scaled.quantize(_TWO_DP, rounding=ROUND_HALF_UP)
        if scaled != target:
            rounded = True
        scaled = target
    return {"normalized_value": scaled,
            "precision": "display-rounded" if rounded else "exact-visible"}


def _claims_exact(rec: dict, provenance: dict) -> bool:
    """True iff the record claims exact-visible equality."""
    for holder in (rec, provenance):
        if not isinstance(holder, dict):
            continue
        precision = holder.get("precision")
        if precision == "exact-visible":
            return True
    for alias in ("claims_exact", "claim_exact", "exact_claim"):
        if rec.get(alias) is True:
            return True
        if isinstance(provenance, dict) and provenance.get(alias) is True:
            return True
    return False


def check_precision_claim(display_value, precision,
                          claims_exact: bool = False) -> list[str]:
    """Check one precision claim. Returns E_PRECISION errors (or [])."""
    claiming = claims_exact or precision == "exact-visible"
    if not claiming:
        return []
    if not isinstance(display_value, str) or not display_value.strip():
        return ["E_PRECISION:exact_claim_on_missing_display: "
                "exact-visible claim requires a display_value"]
    norm = _normalize_display_value(display_value)
    if norm["normalized_value"] is None:
        return ["E_PRECISION:exact_claim_on_unparseable: "
                "unparseable display must not claim exact-visible"]
    if norm["precision"] != "exact-visible":
        return ["E_PRECISION:exact_claim_on_rounded: "
                "display-rounded value must not claim exact equality "
                "with hidden exact"]
    return []


# ---------------------------------------------------------------------------
# Reconciliation helpers (tolerance 0.01 exact, never divide by zero)
# ---------------------------------------------------------------------------

def check_r1_r3(revenue=None, cogs=None, gross_profit=None,
                operating_expenses=None, operating_profit=None,
                other_income=None, finance_costs=None, tax_expense=None,
                net_income=None, tol=None) -> list[str]:
    """Verify R1/R2/R3 within ``tol`` (default 0.01). Returns error list."""
    if tol is not None:
        tolerance = _to_decimal(tol)
        if tolerance is None:
            raise ValueError(f"tol must be numeric, got {tol!r}")
    else:
        tolerance = MONEY_TOL
    errors: list[str] = []
    r = _to_decimal(revenue)
    c = _to_decimal(cogs)
    g = _to_decimal(gross_profit)
    opex = _to_decimal(operating_expenses)
    op = _to_decimal(operating_profit)
    oi = _to_decimal(other_income)
    fc = _to_decimal(finance_costs)
    tax = _to_decimal(tax_expense)
    net = _to_decimal(net_income)
    # R1
    if r is not None and c is not None and g is not None:
        if abs(g - (r - c)) > tolerance:
            errors.append("E_RECONCILIATION:R1: gross_profit != "
                          "revenue - cogs (tol 0.01)")
    elif (revenue is not None or cogs is not None
          or gross_profit is not None):
        errors.append("E_RECONCILIATION:R1_unparseable: "
                      "R1 inputs must be numeric")
    # R2
    if g is not None and opex is not None and op is not None:
        if abs(op - (g - opex)) > tolerance:
            errors.append("E_RECONCILIATION:R2: operating_profit != "
                          "gross_profit - operating_expenses (tol 0.01)")
    elif (gross_profit is not None or operating_expenses is not None
          or operating_profit is not None):
        # Only flag unparseable when the caller supplied at least one
        # R2 input but the triple is incomplete/unparseable AND the
        # caller supplied all three keys (partial triples are skipped
        # silently by validate_record; direct calls with partial Nones
        # get this diagnostic).
        if (gross_profit is not None and operating_expenses is not None
                and operating_profit is not None):
            errors.append("E_RECONCILIATION:R2_unparseable: "
                          "R2 inputs must be numeric")
    # R3
    if (op is not None and oi is not None and fc is not None
            and tax is not None and net is not None):
        if abs(net - (op + oi - fc - tax)) > tolerance:
            errors.append("E_RECONCILIATION:R3: net_income != "
                          "operating_profit + other_income - finance_costs "
                          "- tax_expense (tol 0.01)")
    elif (operating_profit is not None or other_income is not None
          or finance_costs is not None or tax_expense is not None
          or net_income is not None):
        if (operating_profit is not None and other_income is not None
                and finance_costs is not None and tax_expense is not None
                and net_income is not None):
            errors.append("E_RECONCILIATION:R3_unparseable: "
                          "R3 inputs must be numeric")
    return errors


def check_invoice_total(subtotal=None, vat=None, total=None,
                        tol=None) -> list[str]:
    """Verify invoice total == subtotal + vat within tol (default 0.01)."""
    if tol is not None:
        tolerance = _to_decimal(tol)
        if tolerance is None:
            raise ValueError(f"tol must be numeric, got {tol!r}")
    else:
        tolerance = MONEY_TOL
    s = _to_decimal(subtotal)
    v = _to_decimal(vat)
    t = _to_decimal(total)
    if s is None or v is None or t is None:
        return ["E_RECONCILIATION:R4_unparseable: subtotal/vat/total "
                "must be numeric"]
    if abs(t - (s + v)) > tolerance:
        return ["E_RECONCILIATION:R4: total != subtotal + vat (tol 0.01)"]
    return []


def check_transaction_total(amount=None, tax_amount=None, total_amount=None,
                              tol=None) -> list[str]:
    """Verify transaction R5: total_amount == amount + tax_amount (tol 0.01)."""
    if tol is not None:
        tolerance = _to_decimal(tol)
        if tolerance is None:
            raise ValueError(f"tol must be numeric, got {tol!r}")
    else:
        tolerance = MONEY_TOL
    a = _to_decimal(amount)
    t = _to_decimal(tax_amount)
    tot = _to_decimal(total_amount)
    if a is None or t is None or tot is None:
        return ["E_RECONCILIATION:R5_unparseable: amount/tax_amount/total_amount "
                "must be numeric"]
    if abs(tot - (a + t)) > tolerance:
        return ["E_RECONCILIATION:R5: total_amount != amount + tax_amount (tol 0.01)"]
    return []


def budget_variance(actual, budget) -> dict:
    """Derive variance diagnostics without ever dividing by zero.

    Returns ``{"variance": Decimal (2dp), "variance_pct": Decimal|None,
    "budget_zero": bool}``. ``variance = actual - budget`` quantized
    ROUND_HALF_UP to 2dp; ``variance_pct = variance / budget * 100``
    (2dp) when budget != 0 else None. Raises ValueError on
    non-numeric inputs.
    """
    a = _to_decimal(actual)
    b = _to_decimal(budget)
    if a is None or b is None:
        raise ValueError("budget_variance requires numeric actual/budget")
    variance = _quantize_2dp(a - b)
    if b == 0:
        return {"variance": variance, "variance_pct": None,
                "budget_zero": True}
    pct = _quantize_2dp(variance / b * Decimal(100))
    return {"variance": variance, "variance_pct": pct,
            "budget_zero": False}


#: Alias kept so orchestrators/helpers may use either name.
check_budget_variance = budget_variance


def _check_record_reconciliation(rec: dict) -> list[str]:
    """Run R1/R2/R3/R4 + budget-variance checks for keys present in rec."""
    errors: list[str] = []

    def present(*keys) -> bool:
        return all(k in rec and rec[k] is not None for k in keys)

    def parseable(*keys) -> bool:
        return all(_to_decimal(rec[k]) is not None for k in keys)

    # R1
    if present("revenue", "cogs", "gross_profit"):
        if parseable("revenue", "cogs", "gross_profit"):
            r, c, g = (_to_decimal(rec["revenue"]),
                       _to_decimal(rec["cogs"]),
                       _to_decimal(rec["gross_profit"]))
            if abs(g - (r - c)) > MONEY_TOL:
                errors.append("E_RECONCILIATION:R1: gross_profit != "
                              "revenue - cogs (tol 0.01)")
    # R2
    if present("gross_profit", "operating_expenses", "operating_profit"):
        if parseable("gross_profit", "operating_expenses",
                     "operating_profit"):
            g = _to_decimal(rec["gross_profit"])
            opex = _to_decimal(rec["operating_expenses"])
            op = _to_decimal(rec["operating_profit"])
            if abs(op - (g - opex)) > MONEY_TOL:
                errors.append("E_RECONCILIATION:R2: operating_profit != "
                              "gross_profit - operating_expenses "
                              "(tol 0.01)")
    # R3
    r3_keys = ("operating_profit", "other_income", "finance_costs",
               "tax_expense", "net_income")
    if present(*r3_keys):
        if parseable(*r3_keys):
            op = _to_decimal(rec["operating_profit"])
            oi = _to_decimal(rec["other_income"])
            fc = _to_decimal(rec["finance_costs"])
            tax = _to_decimal(rec["tax_expense"])
            net = _to_decimal(rec["net_income"])
            if abs(net - (op + oi - fc - tax)) > MONEY_TOL:
                errors.append("E_RECONCILIATION:R3: net_income != "
                              "operating_profit + other_income - "
                              "finance_costs - tax_expense (tol 0.01)")
    # R4 (invoice)
    if present("subtotal", "vat", "total"):
        if parseable("subtotal", "vat", "total"):
            s = _to_decimal(rec["subtotal"])
            v = _to_decimal(rec["vat"])
            t = _to_decimal(rec["total"])
            if abs(t - (s + v)) > MONEY_TOL:
                errors.append("E_RECONCILIATION:R4: total != subtotal + "
                              "vat (tol 0.01)")
    # R5 (transaction)
    if present("amount", "tax_amount", "total_amount"):
        if parseable("amount", "tax_amount", "total_amount"):
            a = _to_decimal(rec["amount"])
            t = _to_decimal(rec["tax_amount"])
            tot = _to_decimal(rec["total_amount"])
            if abs(tot - (a + t)) > MONEY_TOL:
                errors.append("E_RECONCILIATION:R5: total_amount != amount + "
                              "tax_amount (tol 0.01)")
    # Budget variance field check
    if present("actual", "budget", "variance"):
        if parseable("actual", "budget", "variance"):
            a = _to_decimal(rec["actual"])
            b = _to_decimal(rec["budget"])
            reported = _to_decimal(rec["variance"])
            expected = _quantize_2dp(a - b)
            if abs(reported - expected) > MONEY_TOL:
                errors.append("E_RECONCILIATION:budget_variance: variance "
                              "!= actual - budget (tol 0.01)")
    # variance_pct reconciliation (R8): when actual+budget+variance+variance_pct
    # all parseable and budget != 0, reported pct must match derived pct.
    if present("actual", "budget", "variance", "variance_pct"):
        if parseable("actual", "budget", "variance", "variance_pct"):
            b = _to_decimal(rec["budget"])
            if b != 0:
                try:
                    derived = budget_variance(rec["actual"], rec["budget"])
                    reported_pct = _to_decimal(rec["variance_pct"])
                    if derived["variance_pct"] is not None and abs(reported_pct - derived["variance_pct"]) > MONEY_TOL:
                        errors.append("E_RECONCILIATION:variance_pct: variance_pct != "
                                      "variance / budget * 100 (tol 0.01)")
                except ValueError:
                    pass
            else:
                # budget == 0: variance_pct must be absent/None, never fabricated
                reported_pct = _to_decimal(rec["variance_pct"])
                if reported_pct is not None:
                    errors.append("E_RECONCILIATION:variance_pct: variance_pct must be None "
                                  "when budget == 0 (never divide by zero)")
    return errors


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def warnings_for_record(rec: dict) -> list[str]:
    """Return pass-through W_LABEL warnings (never errors) for unknowns."""
    if not isinstance(rec, dict):
        return []
    warnings: list[str] = []
    for key in ("metric", "label"):
        val = rec.get(key)
        if isinstance(val, str) and val.strip() and val not in KNOWN_METRICS:
            warnings.append(f"W_LABEL:unknown_label:{val!r} passed through, "
                            f"not invented into canonical keys")
    return warnings


def _check_unknown_labels(rec: dict, strict: bool) -> list[str]:
    if not strict:
        return []
    errors: list[str] = []
    for key in ("metric", "label"):
        val = rec.get(key)
        if (isinstance(val, str) and val.strip()
                and val not in KNOWN_METRICS):
            errors.append(f"E_LABEL:unknown_label:{val!r} strict mode "
                          f"rejects unmapped labels")
    return errors


# ---------------------------------------------------------------------------
# Record / batch validation (deterministic, fail-collecting not raise-first)
# ---------------------------------------------------------------------------

def validate_record(rec: dict, strict: bool = False) -> list[str]:
    """Validate one plain-dict extraction record.

    Returns a deterministic errors list (``[]`` means valid). Never
    raises on invalid content; only on a wrong input type it returns a
    single ``E_PROVENANCE`` diagnostic. Stage order is fixed:
    provenance → branch → period → money → sign → precision →
    reconciliation → label.
    """
    if not isinstance(rec, dict):
        return ["E_PROVENANCE:record_not_dict: record must be a dict"]
    errors: list[str] = []

    # 1. provenance (includes nested/top-level page + branch-enum checks)
    errors.extend(_check_provenance(rec))

    # 2. explicit top-level branch fast-path is already covered by
    #    _check_provenance, but a bare invalid branch with no provenance
    #    must still surface exactly once: dedupe by tracking.
    branch = rec.get("branch_id")
    if branch is not None and not is_valid_branch(branch):
        marker = f"E_BRANCH:invalid_branch_id:{branch!r}"
        if not any(e.startswith(marker) for e in errors):
            errors.append(f"{marker} expected one of {list(BRANCH_IDS)}")

    # 3. period format/range + Dammam pre-opening gate
    period = rec.get("period")
    if period is not None:
        errors.extend(_check_period_format(period))
        if branch is not None and is_valid_branch(branch):
            if not _check_period_format(period):  # only gate parseable ones
                if _is_dammam_pre_opening(branch, period):
                    errors.append(
                        "E_BRANCH:dammam_pre_opening: BR-DMM has no "
                        f"activity before 2019-01-01 (got {period!r})")

    # 4+5. money (E_MONEY) then sign (E_SIGN) in fixed field order
    for field in MONEY_FIELDS_ORDER:
        if field not in rec or rec[field] is None:
            continue
        raw = rec[field]
        dec = _to_decimal(raw)
        if dec is None or not _is_2dp(dec) or abs(dec) >= MAX_ABS_MONEY:
            if dec is None:
                errors.append(f"E_MONEY:unparseable:{field}:{raw!r} "
                              f"must be 2dp SAR money")
            elif not _is_2dp(dec):
                errors.append(f"E_MONEY:precision:{field}:{raw!r} "
                              f"must carry exactly 2dp")
            else:
                errors.append(f"E_MONEY:range:{field}:{raw!r} "
                              f"abs must be < 1e12")
            continue
        if field in NONNEGATIVE_FIELDS and dec < 0:
            errors.append(f"E_SIGN:negative:{field}:{raw!r} must never "
                          f"be negative")

    # Generic metric/value pair (extraction rows shaped as one metric).
    if "metric" in rec and "value" in rec and rec["value"] is not None:
        metric = rec.get("metric")
        raw = rec.get("value")
        # Skip when the generic pair duplicates an explicit money field
        # already checked above (same field name + same raw object/value)
        # to avoid double-reporting one violation.
        duplicate = (isinstance(metric, str) and metric in MONEY_FIELDS_ORDER
                     and metric in rec and rec[metric] == raw)
        if not duplicate:
            dec = _to_decimal(raw)
            if dec is None or not _is_2dp(dec) or abs(dec) >= MAX_ABS_MONEY:
                errors.append(f"E_MONEY:unparseable:value:{raw!r} must be "
                              f"2dp SAR money")
            elif (isinstance(metric, str) and metric in NONNEGATIVE_FIELDS
                  and dec < 0):
                errors.append(f"E_SIGN:negative:value:{raw!r} for metric "
                              f"{metric!r} must never be negative")

    # 6. precision safety
    provenance = _provenance_of(rec)
    display = provenance.get("display_value", rec.get("display_value"))
    precision = provenance.get("precision", rec.get("precision"))
    claiming = _claims_exact(rec, provenance
                             if isinstance(provenance, dict) else {})
    errors.extend(check_precision_claim(display, precision, claiming))
    # 6b. rounded-value consistency (R5 gap): a parseable suffixed display
    # (e.g. 23.90M) paired with a mismatched value fabricates hidden digits.
    if precision == "display-rounded" and isinstance(display, str) and display.strip():
        norm = _normalize_display_value(display)
        if norm["normalized_value"] is not None and norm["precision"] == "display-rounded":
            # Only enforce when display carries a magnitude suffix (hidden
            # precision unknowable); coarse non-suffixed displays are warning-only.
            lowered = display.strip().casefold()
            for tok in _CURRENCY_TOKENS:
                lowered = lowered.replace(tok, "")
            lowered = lowered.strip()
            if lowered and lowered[-1] in _MAGNITUDE:
                for vkey in ("value", "actual", "revenue", "total", "net_income"):
                    if vkey in rec and rec[vkey] is not None:
                        dec = _to_decimal(rec[vkey])
                        if dec is not None and _is_2dp(dec) and abs(dec - norm["normalized_value"]) > MONEY_TOL:
                            errors.append(
                                f"E_PRECISION:rounded_value_mismatch:{vkey}:{rec[vkey]!r} "
                                f"!= normalized display {norm['normalized_value']} (display {display!r})"
                            )
                            break

    # 6c. variance_pct format (2dp when present)
    if "variance_pct" in rec and rec["variance_pct"] is not None:
        dec = _to_decimal(rec["variance_pct"])
        if dec is None or not _is_2dp(dec) or abs(dec) >= MAX_ABS_MONEY:
            errors.append(f"E_MONEY:unparseable:variance_pct:{rec['variance_pct']!r} must be 2dp")

    # 7. reconciliation (only over present + parseable subsets)
    errors.extend(_check_record_reconciliation(rec))

    # 8. unknown labels (error only in strict mode)
    errors.extend(_check_unknown_labels(rec, strict))

    return errors


def is_valid_record(rec: dict, strict: bool = False) -> bool:
    """True iff ``validate_record`` returns no errors."""
    return validate_record(rec, strict=strict) == []


def validate_batch(records, strict: bool = False) -> dict:
    """Validate many records deterministically (input order preserved).

    Keys are ``record_id`` when present, non-empty, and unique;
    otherwise the integer ``index``. Colliding ``record_id`` values
    are disambiguated as ``"{record_id}#{index}"``. Values are the
    per-record errors lists (``[]`` means valid).
    """
    if records is None:
        return {}
    try:
        items = list(records)
    except TypeError:
        return {}
    out: dict = {}
    seen: set = set()
    for index, rec in enumerate(items):
        key = index
        if isinstance(rec, dict):
            rid = rec.get("record_id")
            if isinstance(rid, str) and rid.strip():
                key = rid
            elif isinstance(rid, int) and not isinstance(rid, bool):
                key = rid
        if key in seen:
            key = f"{key}#{index}"
        seen.add(key)
        if isinstance(rec, dict):
            out[key] = validate_record(rec, strict=strict)
        else:
            out[key] = ["E_PROVENANCE:record_not_dict: record must be "
                        "a dict"]
    return out
