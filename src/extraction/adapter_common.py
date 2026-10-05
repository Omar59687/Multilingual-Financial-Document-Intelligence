"""Shared thin helpers for the Phase 3 extraction adapters (bounded slice).

Contract: ``docs/STRUCTURED_EXTRACTION_DESIGN.md`` §19 item 5. Holds
the demonstrated-shared pure helpers previously duplicated across
``runner_native``/``runner_paddle``/``runner_qwen`` (third consumer —
the recorded deferred-unification point), plus the fleet-wide
foreign-currency guard.

Contents: metric identity/balance-sheet mapping core, period
derivation, verbatim display extraction, foreign-token detection.
Record assembly, provenance coordinates, and module-specific alias
tables stay in the owning adapters. No ``duckdb``/``store``/route
imports anywhere; stdlib + pydantic only (via the frozen leaf
modules). No GT reads, no I/O, no network, no randomness.
"""

from __future__ import annotations

from typing import Any

from .periods import normalize_date_display, parse_period
from .schemas import Metric
from . import validation as _V

__all__ = [
    "METRIC_VALUES",
    "FOREIGN_CURRENCY_TOKENS",
    "map_identity_or_bs",
    "derive_period",
    "display_for",
    "has_foreign_currency",
]

METRIC_VALUES: frozenset[str] = frozenset(m.value for m in Metric)

#: Foreign-currency tokens mirrored from schemas._FOREIGN_CURRENCY_TOKENS
#: (substring match is the frozen lenient convention there). Displays
#: carrying one must never source an SAR record: the shared normalizer
#: strips these tokens silently for non-exact claims, so adapters are
#: the fail-closed boundary.
FOREIGN_CURRENCY_TOKENS: tuple[str, ...] = (
    "$",
    "usd",
    "aed",
    "egp",
    "qar",
    "kwd",
)


def map_identity_or_bs(header: Any) -> str | None:
    """Map a label via identity casefold, else balance-sheet normalization.

    Covers verbatim metric headers (``amount``, ``Subtotal``) and the
    frozen balance-sheet vocabulary (incl. the
    ``property_and_equipment`` alias). Module-specific aliases
    (Paddle Arabic, statements P&L) layer on top in owning adapters.
    Unknown labels return None — never invented.
    """
    if not isinstance(header, str):
        return None
    key = header.strip()
    if not key:
        return None
    folded = key.casefold()
    if folded in METRIC_VALUES:
        return folded
    try:
        mapped = _V.normalize_bs_label(key)
    except Exception:
        mapped = None
    if mapped is not None and mapped in METRIC_VALUES:
        return mapped
    return None


def derive_period(raw: Any) -> tuple[str, int]:
    """Derive (canonical period, fiscal_year) or raise ValueError."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"period source must be a non-empty string, got {raw!r}")
    text = raw.strip()
    try:
        parsed = parse_period(text)
        return parsed["canonical"], parsed["fiscal_year"]
    except ValueError:
        pass
    if "/" in text:
        iso = normalize_date_display(text)
        parsed = parse_period(iso)
        return parsed["canonical"], parsed["fiscal_year"]
    raise ValueError(f"unparseable period {raw!r}")


def display_for(raw: Any) -> str | None:
    """Verbatim display string for a cell value (stored string unchanged).

    Outer whitespace is left for the frozen Pydantic config (strips on
    assignment); inner spacing is preserved end to end. ``None`` marks
    absent/non-money (never zero).
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, str):
        return raw if raw.strip() else None
    text = str(raw)
    return text if text.strip() else None


def has_foreign_currency(display: str) -> bool:
    """True iff a display carries a foreign-currency token (SAR-only V1)."""
    lowered = display.casefold()
    return any(tok in lowered for tok in FOREIGN_CURRENCY_TOKENS)
