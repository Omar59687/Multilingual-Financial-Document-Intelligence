"""Native statement-tables adapter (MizanIQ Phase 3, bounded slice).

Contract: ``docs/STRUCTURED_EXTRACTION_DESIGN.md`` §19 items 1–4, 6–7.
Deterministic adapter over native ``TABLE_CELL`` grids (PDF lattice
tables and DOCX tables: grouped by page/table/row, first cell = label
candidate, remaining cells = ``(column, value)`` pairs with caller
mapped periods) emitting validated ``FinancialRecord``s inside
``ExtractionResult`` with ``extraction_route="native"``.

Explicitly OUT: Arabic statement labels and commentary prose (no
frozen rules — deferred with rationale), scanned/visual routes (own
slices), G1/thresholds, persistence, period inference (periods come
only from the explicit ``column_periods`` mapping, never from headers
or filenames).

Compatibility: stdlib + pydantic only (no ``duckdb``/``store``/ingestion
imports anywhere — duck-typed ``Document`` inputs like the sibling
adapters). Shared thin helpers delegate to ``adapter_common``. No
ground-truth reads, no I/O, no network, no randomness, no wall-clock
except ``latency_ms``.

Fail-closed rules (§16 pattern): exactly-2dp money, shared foreign
guard, ``validate_record`` gating, unparseable displays yield no
record (never zero), Dammam pre-2019 rejected by validation.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from typing import Any

from . import adapter_common as _C
from .provenance import build_provenance, record_id_for
from .schemas import ExtractionResult, FinancialRecord
from . import validation as _V

__all__ = [
    "PL_ALIASES",
    "normalize_pl_label",
    "run_statements",
]

#: Statements-local P&L aliases (visible-label names from the native
#: statement grain; never enum duplication — the §4 alias-layer rule).
#: The generic casefold/collapse/parens-strip + space→underscore rule
#: already covers the other statement labels (``Gross profit`` →
#: ``gross_profit``); only genuinely divergent names are listed.
#: Invoice-grain display variants (``VAT 15%``, ``Total due``) are V1-
#: frozen renderings (flat-15% VAT is the approved V1 simplification);
#: revisit if historical VAT is ever scheduled.
PL_ALIASES: dict[str, str] = {
    "cost of sales": "cogs",
    "vat 15%": "vat",
    "total due": "total",
}


def normalize_pl_label(label: Any) -> str | None:
    """Normalize a free-text P&L statement label to a canonical metric.

    Mirrors the ``validation.normalize_bs_label`` pattern (attributed):
    casefold, strip, whitespace-collapse, trailing parenthetical-suffix
    strip (``Cost of sales (COGS)``), ``&`` → ``and``, ``-``/``,`` →
    space cleanup; then the minimal alias table above; then a generic
    space → underscore mapping gated by ``METRIC_VALUES`` membership.
    Returns None when unmapped — never invented.
    """
    if not isinstance(label, str):
        return None
    text = label.strip().casefold()
    if not text:
        return None
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*\([^()]*\)\s*$", "", text).strip()
    if not text:
        return None
    text = text.replace("&", "and").replace("-", " ").replace(",", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None
    if text in PL_ALIASES:
        return PL_ALIASES[text]
    candidate = text.replace(" ", "_")
    if candidate in _C.METRIC_VALUES:
        return candidate
    return None


def _map_label(header: Any) -> str | None:
    """Map a statement label: shared core first, then P&L normalization."""
    core = _C.map_identity_or_bs(header)
    if core is not None:
        return core
    try:
        return normalize_pl_label(header)
    except Exception:
        return None


def _el_get(element: Any, name: str, default: Any = None) -> Any:
    if isinstance(element, Mapping):
        return element.get(name, default)
    return getattr(element, name, default)


def _element_type_str(element: Any) -> str:
    raw = _el_get(element, "element_type", "")
    value = getattr(raw, "value", raw)
    return str(value).strip().casefold() if value is not None else ""


def run_statements(
    document: Any,
    *,
    column_periods: Mapping[Any, Any],
    label_column: int = 0,
    branch_id: str | None = None,
) -> ExtractionResult:
    """Adapt native statement table cells into validated financial records.

    ``document`` is an ingestion ``Document`` dataclass instance or a
    plain dict from ``Document.to_dict()``. ``column_periods`` maps
    value-column index → period string (required, validated upfront;
    columns absent from it are skipped silently as structural).
    ``label_column`` (default 0) holds the row labels. ``branch_id``
    defaults to None (company-wide grain). Returns an
    ``ExtractionResult``; ``ok=True`` iff the runner completed without
    exception.
    """
    started = time.perf_counter()

    def _elapsed_ms() -> float:
        return (time.perf_counter() - started) * 1000.0

    if isinstance(document, (str, bytes)) or document is None:
        raise TypeError("run_statements requires an ingestion Document or mapping, "
                        f"got {type(document).__name__}")
    if isinstance(document, Mapping):
        doc_id = document.get("document_id")
        elements = document.get("elements")
    elif hasattr(document, "document_id") and hasattr(document, "elements"):
        doc_id = getattr(document, "document_id")
        elements = getattr(document, "elements")
    else:
        raise TypeError("run_statements requires an ingestion Document or mapping with "
                        f"'document_id' + 'elements', got {type(document).__name__}")
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise TypeError("run_statements requires a non-empty document_id string")
    if elements is None or isinstance(elements, (str, bytes, Mapping)):
        raise TypeError("run_statements requires an elements list")
    try:
        element_list = list(elements)
    except TypeError:
        raise TypeError("run_statements requires an elements iterable") from None
    doc_id = doc_id.strip()
    if not isinstance(column_periods, Mapping):
        raise TypeError("run_statements requires a column_periods mapping")
    if isinstance(label_column, bool) or not isinstance(label_column, int) or label_column < 0:
        raise TypeError("run_statements label_column must be an int >= 0")
    if branch_id is not None and (not isinstance(branch_id, str) or not branch_id.strip()):
        raise TypeError("run_statements branch_id must be a non-empty string or None")
    branch = branch_id.strip() if isinstance(branch_id, str) else None
    periods: dict[Any, tuple[str, int]] = {}
    for col, raw in dict(column_periods).items():
        if isinstance(col, bool) or not isinstance(col, int) or col < 0:
            raise TypeError(f"run_statements column_periods keys must be non-negative "
                            f"column indices, got {col!r}")
        if not isinstance(raw, str) or not raw.strip():
            raise TypeError(f"run_statements column_periods[{col!r}] must be a non-empty string")
        try:
            periods[col] = _C.derive_period(raw)
        except ValueError as exc:
            raise ValueError(f"run_statements unparseable period for column {col!r}: {exc}") from None

    records: list[FinancialRecord] = []
    warnings: list[str] = []
    unknown_labels: set[str] = set()
    try:
        by_type: dict[str, int] = {}
        cells: list[Any] = []
        for el in element_list:
            etype = _element_type_str(el)
            by_type[etype] = by_type.get(etype, 0) + 1
            if etype == "table_cell":
                cells.append(el)
        unsupported = {k: n for k, n in by_type.items() if k and k != "table_cell"}
        if unsupported:
            parts = ", ".join(f"{n} {k}" for k, n in sorted(unsupported.items()))
            warnings.append(f"W_ROUTE:statements_deferred: skipped non-table elements "
                            f"({parts}; use the matching route adapter)")
        if not column_periods:
            warnings.append("W_CONFIG:empty: column_periods is empty (no value columns mapped)")
        groups: dict[tuple, list[tuple[Any, Any]]] = {}
        order: list[tuple] = []
        for el in cells:
            key = (_el_get(el, "page"), _el_get(el, "table_index"), _el_get(el, "row"))
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append((_el_get(el, "column"), el))
        for key in order:
            row_cells = sorted(groups[key], key=lambda pair: (pair[0] is None, pair[0]))
            by_col = {c: e for c, e in row_cells}
            label_el = by_col.get(label_column)
            label_raw = _el_get(label_el, "text") if label_el is not None else None
            if not isinstance(label_raw, str) or not label_raw.strip():
                continue  # spacer row: silent skip
            metric = _map_label(label_raw)
            if metric is None:
                unknown_labels.add(str(label_raw))
                continue
            for col, el in row_cells:
                if col == label_column:
                    continue
                if col not in periods:
                    continue  # structural column (comparative %): silent skip
                canonical, fiscal_year = periods[col]
                raw = _el_get(el, "text")
                if raw is None:
                    raw = _el_get(el, "value")
                display = _C.display_for(raw)
                if display is None:
                    continue
                if _C.has_foreign_currency(display):
                    warnings.append(f"W_VALUE:skip: foreign-currency display {display!r} "
                                    f"for {label_raw!r} (SAR only V1, never converted)")
                    continue
                norm = _V._normalize_display_value(display)
                normalized = norm.get("normalized_value")
                precision = norm.get("precision", "unknown")
                if normalized is None:
                    warnings.append(f"W_VALUE:skip: unparseable display {display!r} "
                                    f"for {label_raw!r} (never zero)")
                    continue
                try:
                    prov = build_provenance(
                        document, el,
                        source_label=label_raw if isinstance(label_raw, str) else str(label_raw),
                        display_value=display,
                        normalized_value=normalized,
                        precision=precision,
                        branch_id=branch,
                        extraction_route="native",
                    )
                    rid = record_id_for(
                        {"document_id": prov.document_id, "page": prov.page,
                         "element_id": prov.element_id, "metric": metric,
                         "period": canonical, "value": normalized})
                    rec = FinancialRecord(
                        record_id=rid, metric=metric, period=canonical,
                        fiscal_year=fiscal_year, value=normalized,
                        currency="SAR", branch_id=branch,
                        department_id=None, provenance=prov,
                        document_id=prov.document_id, page=prov.page,
                        source_label=label_raw if isinstance(label_raw, str) else str(label_raw),
                        display_value=display, precision=precision,
                    )
                except Exception as exc:
                    warnings.append(f"W_RECORD:skip: {label_raw!r}={display!r} "
                                    f"failed record construction: {type(exc).__name__}: {exc}")
                    continue
                errors = _V.validate_record(rec.model_dump(mode="python"), strict=False)
                if errors:
                    warnings.append(f"W_VALIDATE:skip: {metric}={display!r} "
                                    f"rejected: {errors[0]}")
                    continue
                records.append(rec)
        if not cells:
            warnings.append("W_ROUTE:statements_empty: document carries no table cells")
        for label in sorted(unknown_labels, key=lambda s: (s.casefold(), s)):
            warnings.append(f"W_LABEL:unknown_label:{label!r} passed through, "
                            f"not invented into canonical keys")
        ok = True
    except Exception as exc:
        warnings.append(f"W_RUNNER:failed: {type(exc).__name__}: {exc}")
        ok = False

    latency = _elapsed_ms()
    if latency != latency or latency in (float("inf"), float("-inf")):
        latency = 0.0
    if latency < 0:
        latency = 0.0
    return ExtractionResult(document_id=doc_id, records=records,
                            warnings=warnings, ok=ok, latency_ms=float(latency))
