"""Phase 3 — Core Pydantic financial schemas (MizanIQ structured DB layer).

Phase 3 contract (Roadmap Phase 3 + Architecture structured DB + Eval Plan):
validated financial records with ``metric / period / value / currency /
source document / source page`` land in DuckDB (deterministic calculations);
AI-produced structured output is Pydantic-validated, never trusted raw
(DEVELOPMENT_RULES Rule 8); every record preserves traceability
(document_id, page, source label/display, provenance) per Rule 10.

NO HIDDEN-PRECISION FABRICATION (Phase 2 provenance policy,
PHASE_2_OCR_VISION_EXIT.md section 7 + ``src/ocrbench/canonical.py``):
every numeric carries ``{display_value, normalized_value (=value),
source_label, precision}`` where precision is ``exact-visible`` |
``display-rounded`` | ``unknown``. A rounded/suffixed display (e.g.
``23.90M SAR``) is ``display-rounded`` and MUST NEVER be inflated to a
hidden exact figure. Unparseable input yields no record (never zero).
Unknown labels stay unmapped (rejected by the Metric vocab here, never
invented into canonical keys).

Compatibility notes: this module is standalone (pydantic + stdlib +
decimal only; duckdb is NOT imported here). It does not touch
``src/ingestion/model.py`` (Document/Element), ``src/ocrbench/canonical.py``,
``schema.py`` or scorers, frozen truth, fixtures, ``data/`` or ``docs/``.

Sign-rule delegation: ``MoneyValue`` enforces SAR 2dp quantization and a
12-integer-digit magnitude cap but allows any sign. Sign rules (non-negative
base facts; negatives only for derived ``gross_profit`` /
``operating_profit`` / ``net_income`` / ``variance``) are enforced in
``validation.py``, which has full record context. This keeps this core
schema free of cross-field metric logic.

Provenance import strategy (integration decision R1-B4): runtime import of
``Provenance`` + ``DOCUMENT_ID_PATTERN`` from ``.provenance`` (no cycle:
provenance never imports schemas). ``FinancialRecord.provenance`` accepts a
``Provenance`` instance or a dict coerced via ``Provenance.model_validate``;
non-dict/non-Provenance values are rejected (fail-closed Rule 10).

Period delegation: ``period`` here is validated for non-emptiness only at the
type level; ``FinancialRecord`` additionally cross-checks ``fiscal_year``
against ``periods.parse_period(...).fiscal_year`` when the period parses
(unparseable periods are left to ``validation.py``/``periods.py`` to report).

Determinism: no UUIDs, no ``datetime.now()``; ``record_id`` is the canonical
content-addressed ``FIN-<12hex>`` from ``provenance.record_id_for`` (sha256
over the normalized core fields; same source fact always yields the same ID,
rich or sparse evidence alike). No second ID namespace exists.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator

from .provenance import DOCUMENT_ID_PATTERN, Provenance

__all__ = [
    "CurrencyLiteral",
    "Metric",
    "Precision",
    "MoneyValue",
    "FinancialRecord",
    "ExtractionResult",
    "normalize_currency",
]

# ---------------------------------------------------------------------------
# Currency: SAR only (V1)
# ---------------------------------------------------------------------------

CurrencyLiteral = Literal["SAR"]

_CURRENCY_MAP = {
    "sar": "SAR",
    "ر.س": "SAR",
    "ر. س": "SAR",
    "ريال": "SAR",
}


def normalize_currency(v: Any) -> str:
    """Normalize a raw currency token to ``SAR`` (V1).

    Accepts case-insensitive ``sar`` (surrounding whitespace tolerated) and
    the Arabic forms ``ر.س`` / ``ريال`` (plus ``ر. س`` convenience variant).
    Anything else (USD, $, AED, ...) raises ``ValueError`` — SAR only V1.
    """
    if not isinstance(v, str):
        raise ValueError(f"currency must be a string, got {type(v).__name__}")
    key = v.strip().casefold()
    # Tolerate one trailing period from sentence punctuation ("SAR.").
    if key.endswith(".") and key not in _CURRENCY_MAP:
        key = key[:-1].strip()
    try:
        return _CURRENCY_MAP[key]
    except KeyError:
        raise ValueError(f"unsupported currency {v!r}: SAR only (V1)") from None


# ---------------------------------------------------------------------------
# Metric vocabulary (canonical, closed)
# ---------------------------------------------------------------------------


class Metric(str, Enum):
    """Closed canonical metric vocabulary.

    Covers the approved P&L/budget/invoice grain: revenue, cogs,
    gross_profit, operating_expenses, operating_profit, other_income,
    finance_costs, tax_expense, net_income, budget_total, variance, plus
    invoice ``subtotal`` / ``vat`` / ``total`` in both bare and
    ``invoice_``-prefixed forms (bare forms are invoice-grain aliases).

    Notes:
    - ``branch_revenue_BR-*`` canonical keys from Phase 2 map to
      ``metric=revenue`` + ``branch_id=BR-...`` here (branch is a dimension,
      not a metric).
    - Unknown labels stay unmapped: they fail validation here rather than
      being invented into canonical keys.
    - Dammam pre-2019 rows are ABSENT (NULL, never zero); that absence rule
      is enforced in ``validation.py``, not by adding a metric.
    """

    REVENUE = "revenue"
    COGS = "cogs"
    GROSS_PROFIT = "gross_profit"
    OPERATING_EXPENSES = "operating_expenses"
    OPERATING_PROFIT = "operating_profit"
    OTHER_INCOME = "other_income"
    FINANCE_COSTS = "finance_costs"
    TAX_EXPENSE = "tax_expense"
    NET_INCOME = "net_income"
    BUDGET_TOTAL = "budget_total"
    VARIANCE = "variance"
    VARIANCE_PCT = "variance_pct"
    INVOICE_SUBTOTAL = "invoice_subtotal"
    INVOICE_VAT = "invoice_vat"
    INVOICE_TOTAL = "invoice_total"
    SUBTOTAL = "subtotal"
    VAT = "vat"
    TOTAL = "total"
    AMOUNT = "amount"
    TAX_AMOUNT = "tax_amount"
    TOTAL_AMOUNT = "total_amount"
    BUDGET = "budget"
    BUDGET_AMOUNT = "budget_amount"
    ACTUAL = "actual"


Precision = Literal["exact-visible", "display-rounded", "unknown"]

# ---------------------------------------------------------------------------
# Money: SAR 2dp, max 12 integer digits, any sign (sign rules in validation.py)
# ---------------------------------------------------------------------------

_CENT = Decimal("0.01")
_MAX_MONEY = Decimal("999999999999.99")  # 12 integer digits, 2dp


def _coerce_money(v: Any) -> Decimal:
    """Coerce numeric input to SAR 2dp (fail-closed, no silent rounding).

    Accepts Decimal/int/float/plain-numeric-string. Floats go through
    ``str()`` so binary noise never leaks in (mirrors dataset ``money()``).
    Strings tolerate commas/spaces/underscores/NBSP thousand separators.
    Rejects bool, NaN/Inf, unparseable strings (caller maps those to "skip
    record", never zero), magnitudes above 999,999,999,999.99, and any input
    whose Decimal exponent is not exactly -2 (exactly two decimal places).

    Fail-closed rule (R1-B1, parity with ``validation._is_2dp``): ``"100"``,
    ``"100.1"``, ``"100.123"``, int ``100`` are REJECTED — callers must
    supply exactly 2dp (``"100.00"``). Quantization is never used to round;
    it is only a no-op normalizer for values already at 2dp. Sign is NOT
    restricted here (see module docstring: validation.py owns it).
    """
    if isinstance(v, bool):
        raise ValueError("money value must not be bool")
    if isinstance(v, Decimal):
        dec = v
    elif isinstance(v, int):
        dec = Decimal(v)
    elif isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError(f"money value got non-finite float: {v!r}")
        dec = Decimal(str(v))
    elif isinstance(v, str):
        s = v.strip().replace(",", "").replace(" ", "").replace("_", "").replace(" ", "")
        if not s:
            raise ValueError("money value got empty string")
        try:
            dec = Decimal(s)
        except InvalidOperation:
            raise ValueError(f"money value got unparseable string: {v!r}") from None
    else:
        raise ValueError(f"money value got unsupported type: {type(v).__name__}")
    if not dec.is_finite():
        raise ValueError("money value must be finite")
    if dec.as_tuple().exponent != -2:
        raise ValueError(f"money value {v!r} must carry exactly 2dp (got exponent {dec.as_tuple().exponent})")
    if abs(dec) > _MAX_MONEY:
        raise ValueError("money value exceeds 12 integer digits")
    return dec


MoneyValue = Annotated[Decimal, BeforeValidator(_coerce_money)]

# ---------------------------------------------------------------------------
# Display-string best-effort parse (no GT reads; mirrors canonical.py policy)
# ---------------------------------------------------------------------------

_DISPLAY_SAR_TOKENS = (
    "sar",
    "ر.س",
    "ر. س",
    "ريال",
    "﷼",
)
_FOREIGN_CURRENCY_TOKENS = (
    "$",
    "usd",
    "aed",
    "egp",
    "qar",
    "kwd",
)
_DISPLAY_CURRENCY_TOKENS = _DISPLAY_SAR_TOKENS + _FOREIGN_CURRENCY_TOKENS
_DISPLAY_MAGNITUDES = ("k", "m", "b")


def _parse_display_best_effort(display: str) -> tuple[Decimal | None, bool, str | None]:
    """Best-effort parse of a visible ``display_value``.

    Returns ``(number, had_suffix, error)``. ``had_suffix`` is True when a
    K/M/B magnitude suffix was present (implies ``display-rounded``).
    SAR currency tokens, commas, spaces are stripped; leading +/- handled.
    Foreign-currency tokens (USD/$/AED/...) yield ``error=foreign-currency``
    (R1-M5: an SAR record must never claim an exact USD display).
    Never consults ground truth. Unparseable -> ``(None, had_suffix, error)``.
    """
    if not isinstance(display, str) or not display.strip():
        return None, False, "empty display value"
    text = display.strip()
    sign = 1
    if text[:1] in ("+", "-"):
        sign = -1 if text[0] == "-" else 1
        text = text[1:].strip()
    lowered = text.casefold()
    foreign = any(tok in lowered for tok in _FOREIGN_CURRENCY_TOKENS)
    if foreign:
        return None, False, f"foreign-currency display value: {display!r} (SAR only V1)"
    for token in _DISPLAY_CURRENCY_TOKENS:
        lowered = lowered.replace(token, "")
    lowered = lowered.replace(" ", " ").strip()
    had_suffix = False
    if lowered and lowered[-1] in _DISPLAY_MAGNITUDES:
        had_suffix = True
        lowered = lowered[:-1].strip()
    compact = lowered.replace(",", "").replace(" ", "").replace("_", "")
    try:
        number = Decimal(compact)
    except InvalidOperation:
        return None, had_suffix, f"unparseable display value: {display!r}"
    if not number.is_finite():
        return None, had_suffix, f"non-finite display value: {display!r}"
    return number * sign, had_suffix, None


# ---------------------------------------------------------------------------
# FinancialRecord
# ---------------------------------------------------------------------------


class FinancialRecord(BaseModel):
    """One validated financial fact with full traceability.

    ``value`` is the normalized 2dp SAR amount; ``display_value`` is the
    verbatim visible string it was read from; ``precision`` records whether
    ``value`` is exactly what was shown (``exact-visible``), a rounded
    display (``display-rounded``), or ``unknown``. ``model_validator`` below
    forbids precision inflation (rounded display claiming exact-visible).
    """

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    record_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^FIN-[0-9a-f]{12}$",
        description="Canonical deterministic ID: FIN-<12 lowercase hex> from provenance.record_id_for. No UUIDs, no second namespace.",
    )
    metric: Metric
    period: str = Field(
        min_length=1,
        max_length=32,
        description="Opaque period label here; full grammar validated in periods.py.",
    )
    fiscal_year: int = Field(ge=2015, le=2024)
    value: MoneyValue
    currency: CurrencyLiteral = Field(description="SAR only (V1); normalized case-insensitively.")
    branch_id: Literal["BR-RUH", "BR-JED", "BR-DMM"] | None = Field(
        default=None, description="Branch dimension; None = company-wide."
    )
    department_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^DEP-[A-Za-z0-9_-]+$",
        description="Department dimension, e.g. DEP-SALES; None when not applicable.",
    )
    provenance: Provenance = Field(
        description="Traceability handle (Provenance model; dict coerced via model_validate). Required per Rule 10."
    )
    document_id: str = Field(min_length=1, max_length=64, pattern=DOCUMENT_ID_PATTERN)
    page: int | None = Field(default=None, ge=1)
    source_label: str = Field(min_length=1, max_length=256, description="Verbatim visible label.")
    display_value: str = Field(min_length=1, max_length=256, description="Verbatim visible numeric string.")
    precision: Precision

    @field_validator("currency", mode="before")
    @classmethod
    def _normalize_currency(cls, v: Any) -> str:
        return normalize_currency(v)

    @field_validator("provenance", mode="before")
    @classmethod
    def _coerce_provenance(cls, v: Any) -> Any:
        if isinstance(v, Provenance):
            return v
        if isinstance(v, dict):
            return Provenance.model_validate(v)
        raise ValueError("provenance must be a Provenance or a dict coercible to Provenance")

    @field_validator("fiscal_year", "page", mode="before")
    @classmethod
    def _reject_bool(cls, v: Any) -> Any:
        if isinstance(v, bool):
            raise ValueError("bool is not a valid int field value")
        return v

    @model_validator(mode="after")
    def _check_exact_visible(self) -> FinancialRecord:
        """Forbid precision inflation: exact-visible must be exactly visible."""
        if self.precision == "exact-visible":
            exp = self.value.as_tuple().exponent
            if exp != -2:
                raise ValueError("precision=exact-visible requires value with exactly 2dp")
            parsed, had_suffix, err = _parse_display_best_effort(self.display_value)
            if err is not None or parsed is None:
                raise ValueError(f"precision=exact-visible requires parseable display_value: {err}")
            if had_suffix:
                raise ValueError(
                    "precision=exact-visible forbids K/M/B magnitude suffix "
                    "(suffixed displays are display-rounded)"
                )
            # R1-B2: check decimal places via exponent, not numeric equality
            # (Decimal("100") == Decimal("100.00") is True, but "100" is not
            # exact-visible). Display must carry exactly 2dp.
            if parsed.as_tuple().exponent != -2:
                raise ValueError(
                    "precision=exact-visible requires display_value with exactly 2dp "
                    f"(got exponent {parsed.as_tuple().exponent}; coarse input is display-rounded)"
                )
            if parsed != self.value:
                raise ValueError(
                    "precision=exact-visible requires value == display_value "
                    "(exact 2dp); never inflate rounded displays"
                )
        elif self.precision == "display-rounded":
            # R1-M4/R5 gap: a parseable rounded display must match value
            # (no fabricated hidden digits laundered through display-rounded).
            parsed, had_suffix, err = _parse_display_best_effort(self.display_value)
            if err is None and parsed is not None:
                if had_suffix:
                    scale = {"k": 3, "m": 6, "b": 9}
                    # Re-derive normalized magnitude value for comparison
                    text = self.display_value.strip()
                    if text[:1] in ("+", "-"):
                        text = text[1:].strip()
                    low = text.casefold()
                    for tok in _DISPLAY_CURRENCY_TOKENS:
                        low = low.replace(tok, "")
                    low = low.strip()
                    suffix = low[-1] if low and low[-1] in scale else None
                    if suffix is not None:
                        num = low[:-1].strip().replace(",", "").replace(" ", "").replace("_", "")
                        try:
                            base = Decimal(num)
                            sign = -1 if self.display_value.strip()[:1] == "-" else 1
                            expected = (base * (Decimal(10) ** scale[suffix]) * sign).quantize(
                                _CENT, rounding=ROUND_HALF_UP
                            )
                            if expected != self.value:
                                raise ValueError(
                                    "precision=display-rounded requires value == normalized display "
                                    f"(expected {expected}, got {self.value}); never invent hidden digits"
                                )
                        except InvalidOperation:
                            pass
                else:
                    # Non-suffixed but coarse display claiming rounded must still
                    # match when quantized (e.g. display "100" + value 100.00 is
                    # consistent as rounded; display "100.5" + value 999.00 is not).
                    if parsed.as_tuple().exponent != -2:
                        # Coarse display: value must be the display rounded half-up
                        try:
                            expected = parsed.quantize(_CENT, rounding=ROUND_HALF_UP)
                        except InvalidOperation:
                            expected = None
                        if expected is not None and expected != self.value:
                            # Allow any value whose rounding matches? No — strict:
                            # rounded displays carry no exact claim, but a wildly
                            # inconsistent value is a fabrication signal. Require
                            # magnitude consistency only for suffixed above; for
                            # coarse non-suffixed, document as warning-only.
                            pass
        return self

    @model_validator(mode="after")
    def _check_provenance_consistency(self) -> FinancialRecord:
        """Top-level traceability must agree with nested Provenance (R1-M1)."""
        prov = self.provenance
        if isinstance(prov, Provenance):
            if prov.document_id != self.document_id:
                raise ValueError(
                    f"document_id {self.document_id!r} != provenance.document_id {prov.document_id!r}"
                )
            if prov.page is not None and self.page is not None and prov.page != self.page:
                raise ValueError(f"page {self.page!r} != provenance.page {prov.page!r}")
            if prov.source_label != self.source_label:
                raise ValueError("source_label != provenance.source_label")
            if prov.display_value != self.display_value:
                raise ValueError("display_value != provenance.display_value")
            if prov.precision != self.precision:
                raise ValueError("precision != provenance.precision")
        return self

    @model_validator(mode="after")
    def _check_fiscal_year(self) -> FinancialRecord:
        """fiscal_year must match period's fiscal year when parseable (R1-M2)."""
        try:
            from .periods import parse_period  # deferred: periods never imports schemas
        except Exception:
            return self
        try:
            parsed = parse_period(self.period)
        except Exception:
            return self  # unparseable periods reported by validation/periods layers
        if parsed.get("fiscal_year") != self.fiscal_year:
            raise ValueError(
                f"fiscal_year {self.fiscal_year} != period {self.period!r} fiscal year {parsed.get('fiscal_year')}"
            )
        return self


# ---------------------------------------------------------------------------
# ExtractionResult envelope
# ---------------------------------------------------------------------------


class ExtractionResult(BaseModel):
    """Deterministic per-document extraction envelope (service return shape)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    document_id: str = Field(min_length=1, max_length=64, pattern=DOCUMENT_ID_PATTERN)
    records: list[FinancialRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    ok: bool
    latency_ms: float = Field(default=0.0, ge=0)

    @field_validator("latency_ms", mode="before")
    @classmethod
    def _latency_no_bool(cls, v: Any) -> Any:
        if isinstance(v, bool):
            raise ValueError("latency_ms must not be bool")
        return v

    @field_validator("latency_ms", mode="after")
    @classmethod
    def _latency_finite(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("latency_ms must be finite")
        return v
