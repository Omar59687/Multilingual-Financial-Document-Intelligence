"""Qwen visual-pairs field adapter (MizanIQ Phase 3, bounded slice).

Contract: ``docs/STRUCTURED_EXTRACTION_DESIGN.md`` §18. Deterministic
adapter over Qwen visual pairs shapes (flat ``{label: raw-string}``
maps as produced by ``ocrbench/qwen_adapter.extract_json_fields``)
emitting validated ``FinancialRecord``s inside ``ExtractionResult``
with ``extraction_route="qwen-visual"``.

Explicitly OUT: Tesseract/``light-ocr`` adapters, Paddle grids (own
slice), G1/thresholds, persistence, period/branch inference (explicit
caller params, never detected), Qwen ``tables`` parsing (association
artifacts duplicating the fields verdict — one informational warning,
never readings), Arabic receipt aliases (Paddle-local semantics).

Compatibility: stdlib + pydantic only (no ``duckdb``/``store``/
transformers/``ocrbench`` imports anywhere). No ground-truth reads, no
I/O, no network, no randomness, no wall-clock except ``latency_ms``.

Fail-closed rules mirror §§16–17: exactly-2dp money (coarse displays
such as ``24M`` skip — hidden precision never fabricated),
``validate_record`` gating, unparseable displays yield no record (never
zero), Dammam pre-2019 rejected by validation.

Shared thin helpers (metric core, period, display, foreign guard)
delegate to ``adapter_common`` (§19 unification — third consumer
confirmed by the statements slice); record assembly stays local.
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
    "run_qwen",
]


def _has_foreign_currency(display: str) -> bool:
    """True iff a display carries a foreign token (delegates to shared core)."""
    return _C.has_foreign_currency(display)


def _map_label(header: Any) -> str | None:
    """Map a pairs label (delegates to shared core; no receipt Arabic)."""
    return _C.map_identity_or_bs(header)


def _derive_period(raw: Any) -> tuple[str, int]:
    """Derive (canonical period, fiscal_year) (delegates to shared core)."""
    return _C.derive_period(raw)


def _display_for(raw: Any) -> str | None:
    """Verbatim display string for a pairs value (delegates to shared core)."""
    return _C.display_for(raw)


def run_qwen(
    result: Any,
    *,
    document_period: str,
    branch_id: str | None = None,
    page: int | None = None,
    filename: str | None = None,
    file_type: str | None = None,
    language: str | None = None,
) -> ExtractionResult:
    """Adapt one Qwen visual-pairs result into validated financial records.

    ``result`` is a mapping with ``document_id`` + ``fields`` (flat
    string map, pair order preserved). ``document_period`` is required
    explicit input (never inferred). Returns an ``ExtractionResult``;
    ``ok=True`` iff the runner completed without exception (an empty
    visual reading correctly yields zero records with a warning).
    """
    started = time.perf_counter()

    def _elapsed_ms() -> float:
        return (time.perf_counter() - started) * 1000.0

    if isinstance(result, (str, bytes)) or result is None or not isinstance(result, Mapping):
        raise TypeError("run_qwen requires a Qwen result mapping with "
                        f"'document_id' + 'fields', got {type(result).__name__}")
    doc_id = result.get("document_id")
    fields = result.get("fields")
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise TypeError("run_qwen requires a non-empty document_id string")
    if fields is None or isinstance(fields, (str, bytes)) or not isinstance(fields, Mapping):
        raise TypeError("run_qwen requires a fields flat mapping")
    doc_id = doc_id.strip()
    if not isinstance(document_period, str) or not document_period.strip():
        raise TypeError("run_qwen requires a non-empty document_period string")
    try:
        canonical, fiscal_year = _derive_period(document_period)
    except ValueError as exc:
        raise ValueError(f"run_qwen unparseable document_period: {exc}") from None
    if page is not None and (isinstance(page, bool) or not isinstance(page, int) or page < 1):
        raise TypeError("run_qwen page must be an int >= 1 or None")
    for name, val in (("filename", filename), ("file_type", file_type), ("language", language)):
        if val is not None and (not isinstance(val, str) or not val.strip()):
            raise TypeError(f"run_qwen {name} must be a non-empty string or None")
    if branch_id is not None and (not isinstance(branch_id, str) or not branch_id.strip()):
        raise TypeError("run_qwen branch_id must be a non-empty string or None")
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
        pairs = dict(fields)
        if not pairs:
            warnings.append("W_ROUTE:qwen_empty: visual reading carries no pairs")
        tables = result.get("tables")
        if isinstance(tables, list) and tables:
            warnings.append("W_ROUTE:qwen_tables_ignored: qwen tables are association "
                            "artifacts, not readings (fields pairs only)")
        for label_raw, value_raw in pairs.items():
            metric = _map_label(label_raw)
            if metric is None:
                if isinstance(label_raw, str) and label_raw.strip():
                    unknown_labels.add(str(label_raw))
                continue
            display = _display_for(value_raw)
            if display is None:
                continue
            if _has_foreign_currency(display):
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
                    extraction_route="qwen-visual",
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
