"""Paddle scanned-grid field adapter (MizanIQ Phase 3, bounded slice).

Contract: ``docs/STRUCTURED_EXTRACTION_DESIGN.md`` §17. Deterministic
adapter over PaddleOCR-VL normalized result shapes (``{document_id,
tables: [grids], ...}`` as produced by ``ocrbench/paddle_adapter``
``adapt_output`` + ``build_result``: lists of row-lists of strings)
emitting validated ``FinancialRecord``s inside ``ExtractionResult``
with ``extraction_route="paddle"``.

Explicitly OUT: Tesseract/``light-ocr`` adapters (not in the §15 route
list), Qwen/visual pairs (own slice), G1/thresholds, persistence,
period/branch inference (both are explicit caller params, never
detected), Arabic prose parsing beyond the frozen 3-entry alias table.

Compatibility: stdlib + pydantic only (no ``duckdb``/``store``/paddle
imports anywhere). No ground-truth reads, no I/O, no network, no
randomness, no wall-clock except ``latency_ms`` measurement.

Fail-closed rules mirror the native slice (§16): exactly-2dp money,
``validate_record`` gating, unparseable displays yield no record (never
zero), Dammam pre-2019 rejected by validation, undated inference never
attempted (``document_period`` is required explicit input).

Shared thin helpers (metric core, period, display, foreign guard)
delegate to ``adapter_common`` (§19 unification); Paddle Arabic aliases
and record assembly stay local.
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
    "PADDLE_ARABIC_ALIASES",
    "run_paddle",
]

#: Module-local Arabic receipt aliases (benchmark grid labels, names only;
#: no numeric leakage). Checked after identity + balance-sheet
#: normalization; never enum duplication. ``الضريبة`` matches by prefix so
#: the rule is VAT-rate agnostic.
PADDLE_ARABIC_ALIASES: dict[str, str] = {
    "المبلغ": "amount",
    "الاجمالي": "total_amount",
}

PADDLE_ARABIC_PREFIXES: tuple[tuple[str, str], ...] = (
    ("الضريبة", "tax_amount"),
)

def _map_label(header: Any) -> str | None:
    """Map a grid label: shared core first, then paddle Arabic table/prefixes."""
    core = _C.map_identity_or_bs(header)
    if core is not None:
        return core
    if not isinstance(header, str):
        return None
    key = header.strip()
    if key in PADDLE_ARABIC_ALIASES:
        return PADDLE_ARABIC_ALIASES[key]
    for prefix, metric in PADDLE_ARABIC_PREFIXES:
        if key.startswith(prefix):
            return metric
    return None


def _derive_period(raw: Any) -> tuple[str, int]:
    """Derive (canonical period, fiscal_year) (delegates to shared core)."""
    return _C.derive_period(raw)


def _display_for(raw: Any) -> str | None:
    """Verbatim display string for a grid cell (delegates to shared core)."""
    return _C.display_for(raw)


def run_paddle(
    result: Any,
    *,
    document_period: str,
    branch_id: str | None = None,
    page: int | None = None,
    filename: str | None = None,
    file_type: str | None = None,
    language: str | None = None,
) -> ExtractionResult:
    """Adapt one Paddle normalized result into validated financial records.

    ``result`` is a mapping with ``document_id`` + ``tables`` (list of
    grids, each a list of row-lists). ``document_period`` is required
    explicit input (never inferred). ``branch_id`` defaults to None
    (company-wide grain). ``page``/``filename``/``file_type``/``language``
    are explicit provenance values only. Returns an ``ExtractionResult``;
    ``ok=True`` iff the runner completed without exception.
    """
    started = time.perf_counter()

    def _elapsed_ms() -> float:
        return (time.perf_counter() - started) * 1000.0

    if isinstance(result, (str, bytes)) or result is None or not isinstance(result, Mapping):
        raise TypeError("run_paddle requires a Paddle result mapping with "
                        f"'document_id' + 'tables', got {type(result).__name__}")
    doc_id = result.get("document_id")
    tables = result.get("tables")
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise TypeError("run_paddle requires a non-empty document_id string")
    if tables is None or isinstance(tables, (str, bytes, Mapping)) or not isinstance(tables, list):
        raise TypeError("run_paddle requires a tables list of grids")
    grid_list = tables
    doc_id = doc_id.strip()
    if not isinstance(document_period, str) or not document_period.strip():
        raise TypeError("run_paddle requires a non-empty document_period string")
    try:
        canonical, fiscal_year = _derive_period(document_period)
    except ValueError as exc:
        raise ValueError(f"run_paddle unparseable document_period: {exc}") from None
    if page is not None and (isinstance(page, bool) or not isinstance(page, int) or page < 1):
        raise TypeError("run_paddle page must be an int >= 1 or None")
    for name, val in (("filename", filename), ("file_type", file_type), ("language", language)):
        if val is not None and (not isinstance(val, str) or not val.strip()):
            raise TypeError(f"run_paddle {name} must be a non-empty string or None")
    if branch_id is not None and (not isinstance(branch_id, str) or not branch_id.strip()):
        raise TypeError("run_paddle branch_id must be a non-empty string or None")
    branch = branch_id.strip() if isinstance(branch_id, str) else None

    doc_like: dict[str, Any] = {"document_id": doc_id}
    if filename is not None:
        doc_like["filename"] = filename
    if file_type is not None:
        doc_like["file_type"] = file_type
    if language is not None:
        doc_like["language_hint"] = language

    records: list[FinancialRecord] = []
    warnings: list[str] = []
    unknown_labels: set[str] = set()
    try:
        for gi, grid in enumerate(grid_list):
            if isinstance(grid, (str, bytes)) or grid is None or not isinstance(grid, list):
                warnings.append(f"W_ROW:skip: table {gi} is not a grid list "
                                f"(got {type(grid).__name__})")
                continue
            rows = grid
            for ri, row in enumerate(rows):
                if row is None:
                    continue
                if isinstance(row, (str, bytes)) or not isinstance(row, list):
                    warnings.append(f"W_ROW:skip: table {gi} row {ri} is not a cell list")
                    continue
                cells = row
                if not cells or all(c is None or (isinstance(c, str) and not c.strip()) for c in cells):
                    continue
                if len(cells) == 1:
                    continue  # structural header (section title, stamp): correctly unmapped
                if len(cells) > 2:
                    warnings.append(f"W_ROW:skip: table {gi} row {ri} has {len(cells)} cells "
                                    f"(expected label/value pair)")
                    continue
                label_raw, value_raw = cells[0], cells[1]
                metric = _map_label(label_raw)
                if metric is None:
                    if isinstance(label_raw, str) and label_raw.strip():
                        unknown_labels.add(str(label_raw))
                    continue
                display = _display_for(value_raw)
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
                label_text = label_raw if isinstance(label_raw, str) else str(label_raw)
                try:
                    prov = build_provenance(
                        doc_like, None,
                        source_label=label_text,
                        display_value=display,
                        normalized_value=normalized,
                        precision=precision,
                        branch_id=branch,
                        page=page,
                        table_index=gi,
                        row=ri,
                        column=1,
                        extraction_route="paddle",
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
                        source_label=label_text, display_value=display,
                        precision=precision,
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
        if not grid_list:
            warnings.append("W_ROUTE:paddle_empty: result carries no tables")
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
