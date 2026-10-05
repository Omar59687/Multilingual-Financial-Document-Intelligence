"""Native-only tabular extraction runner (MizanIQ Phase 3, bounded slice).

Contract: ``docs/STRUCTURED_EXTRACTION_DESIGN.md`` §16 (human-approved
2026-10-03). Deterministic field adapter over native tabular ingestion
elements (``CSV_ROW`` mappings + XLSX ``SHEET_CELL`` row-groups) emitting
validated ``FinancialRecord``s inside ``ExtractionResult``.

Explicitly OUT of this slice (still absent here):

- PDF/DOCX free-text and table-label inference (those elements yield zero
  records + summary warnings; deferred to a future slice with a frozen
  P&L label map),
- Paddle/Qwen routes, G1 measurement/thresholds, query library,
  forecasting, taxonomy or persistence changes,
- document-year inference (undated rows fail closed, never invented),
- non-SAR currency, month-name date parsing, environment repair.

Compatibility notes: stdlib + pydantic only (no ``duckdb``/``store`` import
anywhere in this module, so ``import extraction.runner_native`` works on the
core profile). Inputs are ingestion ``Document`` dataclass instances or
plain dicts from ``Document.to_dict()`` (duck-typed, no ingestion import to
avoid cycles). No ground-truth reads, no I/O, no network, no randomness,
no wall-clock except runner ``latency_ms`` measurement (mirrors the
ingestion-service pattern).

Fail-closed rules (from existing frozen layers, not invented here):

- ``MoneyValue`` (schemas) requires exactly 2dp — whole amounts such as
  ``100.0``/``100`` skip with a warning, never padded or rounded;
- ``validate_record`` (validation, ``strict=False``) must be clean or the
  record is skipped with warnings — invalid output is never emitted;
- unparseable displays yield no record (never zero);
- Dammam pre-2019 rows are rejected by validation (never zero-filled).
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from . import adapter_common as _C
from .provenance import build_provenance, record_id_for
from .schemas import ExtractionResult, FinancialRecord
from . import validation as _V

__all__ = [
    "SUPPORTED_ELEMENT_TYPES",
    "DATE_KEYS_PRIORITY",
    "run_native",
]

#: Element types consumed by this slice. Everything else is counted and
#: reported as skipped (PDF/DOCX label inference deferred).
SUPPORTED_ELEMENT_TYPES: tuple[str, ...] = ("csv_row", "sheet_cell")

#: Per-row period source priority: transaction grain first, then invoice
#: grain, then an explicit canonical period field. No document fallback.
DATE_KEYS_PRIORITY: tuple[str, ...] = ("date", "invoice_date", "period")

# Columns that are structurally never metrics (IDs, dates, dimensions,
# counterparties). Anything else that fails metric mapping is silently
# skipped (expected non-metric column, not a warning per column).
_NON_METRIC_HINTS: frozenset[str] = frozenset({
    "transaction_id", "invoice_id", "date", "invoice_date", "period",
    "branch_id", "department_id", "transaction_type", "category",
    "vendor_customer", "description_en", "description_ar", "description",
    "payment_method", "reference_number", "related_transaction_id",
    "status", "source_tables",
})


def _el_get(element: Any, name: str, default: Any = None) -> Any:
    if isinstance(element, Mapping):
        return element.get(name, default)
    return getattr(element, name, default)


def _element_type_str(element: Any) -> str:
    raw = _el_get(element, "element_type", "")
    value = getattr(raw, "value", raw)
    return str(value).strip().casefold() if value is not None else ""


def _doc_get(document: Any, name: str, default: Any = None) -> Any:
    if isinstance(document, Mapping):
        return document.get(name, default)
    return getattr(document, name, default)


def _map_label(header: Any) -> str | None:
    """Map a tabular header to a canonical metric (delegates to shared core)."""
    return _C.map_identity_or_bs(header)


def _derive_period(raw: Any) -> tuple[str, int]:
    """Derive (canonical period, fiscal_year) (delegates to shared core)."""
    return _C.derive_period(raw)


def _display_for(raw: Any) -> str | None:
    """Verbatim display string for a cell (delegates to shared core)."""
    return _C.display_for(raw)


def _is_unknown_label(header: Any) -> bool:
    """True iff a header is a mappable-label candidate with no mapping."""
    if not isinstance(header, str) or not header.strip():
        return False
    if _map_label(header) is not None:
        return False
    return header.strip().casefold() not in _NON_METRIC_HINTS


def _group_sheet_cells(elements: list[Any]) -> tuple[dict, list[Any]]:
    """Group SHEET_CELL elements by (sheet, row); return (groups, header_errors).

    groups: {(sheet, row): [(column, element), ...]} with columns ascending.
    Header row per sheet = minimum row number (normally 1).
    """
    groups: dict[tuple[Any, Any], list[tuple[Any, Any]]] = {}
    for el in elements:
        if _element_type_str(el) != "sheet_cell":
            continue
        sheet = _el_get(el, "sheet")
        row = _el_get(el, "row")
        column = _el_get(el, "column")
        if row is None or column is None or isinstance(row, bool) or isinstance(column, bool):
            continue
        groups.setdefault((sheet, row), []).append((column, el))
    for key in groups:
        groups[key].sort(key=lambda pair: pair[0])
    return groups, []


def _header_for_sheet(groups: dict, sheet: Any) -> tuple[dict, dict] | None:
    """Return ({column: header_match}, {column: header_original}) for row 1.

    Row 1 is required explicitly (never inferred from the minimum
    populated row): a sheet without row 1 returns None and the caller
    skips it with a warning (fail-closed). Matching uses stripped text
    (robust to padded headers); provenance uses the as-stored original
    (outer whitespace is stripped downstream by the frozen Pydantic
    config; inner spacing is preserved end to end).
    """
    if (sheet, 1) not in groups:
        return None
    header: dict[Any, str] = {}
    original: dict[Any, str] = {}
    for column, el in groups[(sheet, 1)]:
        text = _el_get(el, "text")
        value = _el_get(el, "value")
        raw_header = text if isinstance(text, str) and text.strip() else value
        if isinstance(raw_header, str) and raw_header.strip():
            header[column] = raw_header.strip()
            original[column] = raw_header
    if not header:
        return None
    return header, original


def run_native(document: Any) -> ExtractionResult:
    """Run native tabular extraction over one ingested document.

    Accepts an ingestion ``Document`` dataclass instance or a plain dict
    from ``Document.to_dict()``. ``elements`` is the stored list (any
    finite iterable with deterministic order is accepted; one-shot
    iterators are consumed). Returns an ``ExtractionResult`` envelope;
    ``ok=True`` iff the runner completed without exception
    (yield-independent — a commentary document with no tables correctly
    yields zero records with warnings). Wrong input types raise
    ``TypeError``; per-row domain failures become warnings + skips.
    """
    started = time.perf_counter()

    def _elapsed_ms() -> float:
        return (time.perf_counter() - started) * 1000.0

    if isinstance(document, (str, bytes)) or document is None:
        raise TypeError("run_native requires an ingestion Document or mapping, "
                        f"got {type(document).__name__}")
    if isinstance(document, Mapping):
        doc_id = document.get("document_id")
        elements = document.get("elements")
    elif hasattr(document, "document_id") and hasattr(document, "elements"):
        doc_id = getattr(document, "document_id")
        elements = getattr(document, "elements")
    else:
        raise TypeError("run_native requires an ingestion Document or mapping with "
                        f"'document_id' + 'elements', got {type(document).__name__}")
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise TypeError("run_native requires a non-empty document_id string")
    if elements is None or isinstance(elements, (str, bytes, Mapping)):
        raise TypeError("run_native requires an elements list")
    try:
        element_list = list(elements)
    except TypeError:
        raise TypeError("run_native requires an elements iterable") from None
    doc_id = doc_id.strip()

    records: list[FinancialRecord] = []
    warnings: list[str] = []
    unknown_labels: set[str] = set()
    try:
        by_type: dict[str, int] = {}
        csv_rows: list[Any] = []
        sheet_els: list[Any] = []
        for el in element_list:
            etype = _element_type_str(el)
            by_type[etype] = by_type.get(etype, 0) + 1
            if etype == "csv_row":
                csv_rows.append(el)
            elif etype == "sheet_cell":
                sheet_els.append(el)

        unsupported = {k: n for k, n in by_type.items()
                       if k not in SUPPORTED_ELEMENT_TYPES and k}
        if unsupported:
            parts = ", ".join(f"{n} {k}" for k, n in sorted(unsupported.items()))
            warnings.append(f"W_ROUTE:native_deferred: skipped unsupported elements "
                            f"({parts}; PDF/DOCX label inference deferred)")

        # ---- CSV_ROW mappings (insertion/header order preserved) ----
        for el in csv_rows:
            raw_mapping = _el_get(el, "value")
            if not isinstance(raw_mapping, Mapping):
                warnings.append("W_ROW:skip: csv_row element without mapping value "
                                f"(element {_el_get(el, 'element_id')!r})")
                continue
            mapping = dict(raw_mapping)
            for key in mapping:
                if _is_unknown_label(key):
                    unknown_labels.add(str(key))
            period_raw = next((mapping[k] for k in DATE_KEYS_PRIORITY if k in mapping), None)
            if period_raw is None:
                warnings.append("W_PERIOD:skip: row without date/invoice_date/period "
                                f"(element {_el_get(el, 'element_id')!r}; undated facts fail closed)")
                continue
            try:
                canonical, fiscal_year = _derive_period(period_raw)
            except ValueError as exc:
                warnings.append(f"W_PERIOD:skip: unparseable period {period_raw!r} "
                                f"(element {_el_get(el, 'element_id')!r}: {exc})")
                continue
            branch_raw = mapping.get("branch_id")
            dept_raw = mapping.get("department_id")
            branch_id = branch_raw.strip() if isinstance(branch_raw, str) and branch_raw.strip() else (
                branch_raw if branch_raw is None else branch_raw)
            if branch_id == "":
                branch_id = None
            department_id = dept_raw.strip() if isinstance(dept_raw, str) and dept_raw.strip() else (
                dept_raw if dept_raw is None else dept_raw)
            if department_id == "":
                department_id = None
            for header in mapping:
                metric = _map_label(header)
                if metric is None:
                    if _is_unknown_label(header):
                        unknown_labels.add(str(header))
                    continue
                display = _display_for(mapping[header])
                if display is None:
                    continue
                if _C.has_foreign_currency(display):
                    warnings.append(f"W_VALUE:skip: foreign-currency display {display!r} "
                                    f"for {header!r} (SAR only V1, never converted)")
                    continue
                norm = _V._normalize_display_value(display)
                normalized = norm.get("normalized_value")
                precision = norm.get("precision", "unknown")
                if normalized is None:
                    warnings.append(f"W_VALUE:skip: unparseable display {display!r} "
                                    f"for {header!r} (never zero)")
                    continue
                try:
                    prov = build_provenance(
                        document, el,
                        source_label=header if isinstance(header, str) else str(header),
                        display_value=display,
                        normalized_value=normalized,
                        precision=precision,
                        branch_id=branch_id,
                        extraction_route="native",
                    )
                    rid = record_id_for(
                        {"document_id": prov.document_id, "page": prov.page,
                         "element_id": prov.element_id, "metric": metric,
                         "period": canonical, "value": normalized})
                    rec = FinancialRecord(
                        record_id=rid, metric=metric, period=canonical,
                        fiscal_year=fiscal_year, value=normalized,
                        currency="SAR", branch_id=branch_id,
                        department_id=department_id, provenance=prov,
                        document_id=prov.document_id, page=prov.page,
                        source_label=header if isinstance(header, str) else str(header),
                        display_value=display,
                        precision=precision,
                    )
                except Exception as exc:
                    warnings.append(f"W_RECORD:skip: {header!r}={display!r} "
                                    f"failed record construction: {type(exc).__name__}: {exc}")
                    continue
                errors = _V.validate_record(rec.model_dump(mode="python"), strict=False)
                if errors:
                    warnings.append(f"W_VALIDATE:skip: {metric}={display!r} "
                                    f"rejected: {errors[0]}")
                    continue
                records.append(rec)

        # ---- XLSX SHEET_CELL row-groups (first-seen sheet order) ----
        if sheet_els:
            groups, _ = _group_sheet_cells(sheet_els)
            sheets: list[Any] = []
            for el in sheet_els:
                sheet_name = _el_get(el, "sheet")
                if sheet_name not in sheets and any(s == sheet_name for (s, _r) in groups):
                    sheets.append(sheet_name)
            for sheet in sheets:
                resolved = _header_for_sheet(groups, sheet)
                if not resolved:
                    warnings.append(f"W_ROW:skip: sheet {sheet!r} without header row 1")
                    continue
                header, header_orig = resolved
                for original in header_orig.values():
                    if _is_unknown_label(original):
                        unknown_labels.add(str(original))
                data_rows = sorted(r for (s, r) in groups if s == sheet and r != 1)
                for row in data_rows:
                    cells = groups[(sheet, row)]
                    mapping: dict[str, Any] = {}
                    cell_of: dict[str, Any] = {}
                    for column, el in cells:
                        header_text = header.get(column)
                        if header_text is None:
                            continue
                        text = _el_get(el, "text")
                        value = _el_get(el, "value")
                        raw = text if isinstance(text, str) and text.strip() else value
                        if isinstance(raw, str) and not raw.strip():
                            continue
                        if raw is None:
                            continue
                        mapping[header_text] = raw
                        cell_of[header_text] = el
                    if not mapping:
                        continue
                    period_raw = next((mapping[k] for k in DATE_KEYS_PRIORITY if k in mapping), None)
                    if period_raw is None:
                        warnings.append(f"W_PERIOD:skip: sheet {sheet!r} row {row!r} "
                                        f"without date/invoice_date/period")
                        continue
                    if not isinstance(period_raw, str) or not period_raw.strip():
                        # Native date cells may be non-string; use verbatim str.
                        period_raw = str(period_raw).strip() if period_raw is not None else ""
                    try:
                        canonical, fiscal_year = _derive_period(period_raw)
                    except ValueError as exc:
                        warnings.append(f"W_PERIOD:skip: sheet {sheet!r} row {row!r} "
                                        f"unparseable period {period_raw!r}: {exc}")
                        continue
                    branch_raw = mapping.get("branch_id")
                    dept_raw = mapping.get("department_id")
                    branch_id = branch_raw.strip() if isinstance(branch_raw, str) and branch_raw.strip() else (
                        branch_raw if branch_raw is None else branch_raw)
                    if branch_id == "":
                        branch_id = None
                    department_id = dept_raw.strip() if isinstance(dept_raw, str) and dept_raw.strip() else (
                        dept_raw if dept_raw is None else dept_raw)
                    if department_id == "":
                        department_id = None
                    for col in sorted(c for c, h in header.items() if h in mapping):
                        header_text = header[col]
                        label_orig = header_orig.get(col, header_text)
                        metric = _map_label(header_text)
                        if metric is None:
                            if _is_unknown_label(header_text):
                                unknown_labels.add(str(header_text))
                            continue
                        display = _display_for(mapping[header_text])
                        if display is None:
                            continue
                        if _C.has_foreign_currency(display):
                            warnings.append(f"W_VALUE:skip: foreign-currency display {display!r} "
                                            f"for {header_text!r} (SAR only V1, never converted)")
                            continue
                        norm = _V._normalize_display_value(display)
                        normalized = norm.get("normalized_value")
                        precision = norm.get("precision", "unknown")
                        if normalized is None:
                            warnings.append(f"W_VALUE:skip: unparseable display {display!r} "
                                            f"for {header_text!r} (never zero)")
                            continue
                        cell_el = cell_of[header_text]
                        try:
                            prov = build_provenance(
                                document, cell_el,
                                source_label=label_orig if isinstance(label_orig, str) else str(label_orig),
                                display_value=display,
                                normalized_value=normalized,
                                precision=precision,
                                branch_id=branch_id,
                                extraction_route="native",
                            )
                            rid = record_id_for(
                                {"document_id": prov.document_id, "page": prov.page,
                                 "element_id": prov.element_id, "metric": metric,
                                 "period": canonical, "value": normalized})
                            rec = FinancialRecord(
                                record_id=rid, metric=metric, period=canonical,
                                fiscal_year=fiscal_year, value=normalized,
                                currency="SAR", branch_id=branch_id,
                                department_id=department_id, provenance=prov,
                                document_id=prov.document_id, page=prov.page,
                                source_label=label_orig if isinstance(label_orig, str) else str(label_orig),
                                display_value=display, precision=precision,
                            )
                        except Exception as exc:
                            warnings.append(f"W_RECORD:skip: {header_text!r}={display!r} "
                                            f"failed record construction: {type(exc).__name__}: {exc}")
                            continue
                        errors = _V.validate_record(rec.model_dump(mode="python"), strict=False)
                        if errors:
                            warnings.append(f"W_VALIDATE:skip: {metric}={display!r} "
                                            f"rejected: {errors[0]}")
                            continue
                        records.append(rec)

        if not csv_rows and not sheet_els:
            warnings.append("W_ROUTE:native_deferred: no supported tabular elements "
                            "(CSV_ROW/XLSX SHEET_CELL); PDF/DOCX label inference deferred")
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
