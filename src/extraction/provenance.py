"""Provenance + deterministic record IDs (Phase 2 §7 + Dev Rule 10).

Chain of custody (canonical truth -> citation)::

    canonical truth  ->  document  ->  page / element  ->  record  ->  citation
    (frozen design)      (ingestion     (Element: page /        (metric +       (record_id +
                         Document:      sheet / row /            period +         provenance:
                         document_id,   column / table_index,    normalized       document_id,
                         filename,      element_id, text)        value)           filename, page,
                         file_type,                                              element/location,
                         language)                                              source_label,
                                                                                display_value)

Every extracted numeric record MUST carry a :class:`Provenance` so an
auditor can walk record -> element -> page -> document -> canonical
design without consulting ground truth (GT is never read here).

Hidden-precision rule (from ``ocrbench/canonical.py``):
  * ``precision`` is ``exact-visible`` ONLY when the displayed string
    fully specifies the value to 2dp (e.g. ``"100.00"``).
  * Suffixed / magnitude-rounded displays (``"23.90M SAR"``), short
    decimals, or any display whose full precision is unknowable is
    ``display-rounded`` — never fabricate hidden digits.
  * Unparseable displays yield ``normalized_value=None`` (never zero)
    with ``precision`` ``unknown`` (or ``display-rounded`` when the
    display is visibly rounded but unparseable — never
    ``exact-visible``; enforced by validator).

Determinism:
  * No UUID, no random, no timestamps. Record IDs are content-addressed:
    ``FIN-<12 lowercase hex>`` = first 12 chars of
    ``sha256(canonical JSON)``.
  * Canonical serialization (documented, stable across runs/platforms):
    ``json.dumps(key, sort_keys=True, separators=(",", ":"),
    ensure_ascii=False).encode("utf-8")``. Sorted keys, no whitespace,
    UTF-8 bytes. Same inputs -> same ID.
  * Mirrors ingestion ``document_id_for`` stability (``DEV-###`` preserved
    else ``DOC-<12hex>``); this module never imports ingestion or schemas
    (self-contained so ``schemas.py``/I1 can import from here without a
    cycle).

Compatibility:
  * Self-contained: stdlib + Pydantic v2 only. No ``duckdb``, no GT reads,
    no I/O. Pydantic v2 frozen ``BaseModel`` chosen to match I1's stack.
  * Balance-sheet compatibility (Phase 3 verified): document_id/source_label/display_value/precision
    + branch dims already supported; FIN-* hash is metric-agnostic via casefold; locations stay
    optional for scanned-table sources (document-level provenance valid for image-wide visuals).
"""

from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator

__all__ = ["Provenance", "canonical_key", "record_id_for", "build_provenance"]

DOCUMENT_ID_PATTERN = r"^(DEV-\d{3}|DOC-[0-9a-fA-F]{12})$"
"""Allowed ``document_id`` shape: ``DEV-###`` (frozen-design prefix,
mirrors ``ingestion.model.document_id_for`` which preserves the ``DEV-###``
filename prefix) or ``DOC-<12 hex>`` (content-addressed fallback:
first 12 hex chars of ``sha256`` bytes). No other shapes accepted."""


class Provenance(BaseModel):
    """One evidence pointer for one extracted value (Dev Rule 10 § traceability).

    Traceability ("where applicable" per Rule 10): ``document_id`` always;
    ``filename``, ``page``, element/location (``sheet``/``row``/``column``/
    ``table_index``/``element_id``), ``language``, document type
    (``doc_type``) whenever the source format provides them (document-level
    provenance with no page is legitimate for image-wide visuals).
    Phase 2 provenance (§7): ``display_value`` (raw string as seen),
    ``normalized_value`` (``Decimal`` 2dp or ``None``), ``source_label``
    (visible label), ``precision``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    document_id: str = Field(
        pattern=DOCUMENT_ID_PATTERN,
        description="Stable doc ID: DEV-### (frozen) or DOC-<12hex> (content-addressed).",
    )
    filename: Optional[str] = Field(default=None, description="Source filename, if known.")
    file_type: Optional[Literal["pdf", "docx", "xlsx", "csv", "png", "jpg"]] = Field(
        default=None, description="Source container type."
    )
    page: Optional[int] = Field(default=None, ge=1, description="1-based page number, if paged.")
    sheet: Optional[str] = Field(default=None, description="Workbook sheet name, if tabular.")
    row: Optional[int] = Field(default=None, ge=0, description="Row index (0-based table / native sheet).")
    column: Optional[int] = Field(default=None, ge=0, description="Column index (0-based).")
    table_index: Optional[int] = Field(
        default=None, ge=0, description="0-based table order on page/sheet."
    )
    element_id: Optional[str] = Field(default=None, description="Ingestion Element.element_id, if known.")
    branch_id: Optional[Literal["BR-RUH", "BR-JED", "BR-DMM"]] = Field(
        default=None, description="Branch dimension (R3-I5 fix: was missing; mirrors FinancialRecord/store)."
    )
    language: Optional[Literal["ar", "en", "mixed", "unknown"]] = Field(
        default=None, description="Document/element language hint."
    )
    doc_type: Optional[str] = Field(default=None, description="Document type label (design-side).")
    source_label: str = Field(min_length=1, description="Visible label as seen (non-empty).")
    display_value: str = Field(min_length=1, description="Raw displayed string as seen (non-empty after strip).")
    normalized_value: Optional[Decimal] = Field(
        default=None,
        description="Normalized numeric to at most 2dp, or None when unparseable (never zero).",
    )
    precision: Literal["exact-visible", "display-rounded", "unknown"] = Field(
        description="exact-visible only when display fully specifies 2dp; never fabricate hidden precision."
    )
    extraction_route: Optional[Literal["native", "light-ocr", "paddle", "qwen-visual"]] = Field(
        default=None,
        description="Phase 2 routing-policy vocab; explicit-only, never required; excluded from ID hash.",
    )

    @field_validator("page", "row", "column", "table_index", mode="before")
    @classmethod
    def _reject_bool_coords(cls, v: Any) -> Any:
        if isinstance(v, bool):
            raise ValueError("page/row/column/table_index must not be bool")
        return v

    @field_validator("normalized_value")
    @classmethod
    def _check_2dp(cls, v: Optional[Decimal]) -> Optional[Decimal]:
        if v is None:
            return None
        if not v.is_finite():
            raise ValueError("normalized_value must be finite (no NaN/Infinity).")
        if v.as_tuple().exponent < -2:
            raise ValueError("normalized_value must have at most 2 decimal places.")
        return v

    @model_validator(mode="after")
    def _check_precision_consistency(self) -> "Provenance":
        # Both contract clauses are contrapositives of one rule:
        #   precision == exact-visible  =>  normalized_value is not None.
        if self.precision == "exact-visible" and self.normalized_value is None:
            raise ValueError("precision 'exact-visible' requires normalized_value is not None.")
        # R3-H1 hardening (delegated authoritative check lives in
        # schemas.FinancialRecord + validation.check_precision_claim, which
        # verify suffix/coarse/mismatch): provenance additionally rejects
        # empty display claiming exact (display_value min_length=1 already
        # covers whitespace-only, this covers None-edge via strip).
        if self.precision == "exact-visible" and not self.display_value.strip():
            raise ValueError("precision 'exact-visible' requires non-empty display_value.")
        # Scope-correction: source locations are OPTIONAL ("where applicable"
        # per DEVELOPMENT_RULES Rule 10). Image-wide visuals (PNG/JPG) and
        # document-level facts may carry document-level provenance with no
        # page/element/sheet/row/column/table_index. Callers SHOULD supply
        # location whenever the source format provides it; record IDs remain
        # deterministic either way (absent coordinates hash as JSON null).
        return self

    @field_serializer("normalized_value", when_used="json")
    def _ser_decimal(self, v: Optional[Decimal]) -> Optional[str]:
        # String-serializable JSON form: Decimal -> "100.00", None -> None.
        return str(v) if v is not None else None


def _as_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, bool):
        return None
    text = str(value).strip()
    return text if text else None


def _normalize_metric(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text.casefold() if text else None


def _normalize_period(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    # Normalize lowercase quarter (2023-q4 -> 2023-Q4) for ID stability (R3-D3).
    text = re.sub(r"-q([1-4])$", lambda m: f"-Q{m.group(1)}", text, flags=re.IGNORECASE)
    # Prefer canonical form from periods.py when parseable (no hard dep: try/except).
    try:
        from .periods import parse_period  # periods never imports provenance: no cycle

        try:
            return parse_period(text)["canonical"]
        except Exception:
            return text
    except Exception:
        return text


def _normalize_value(value: Any) -> Optional[str]:
    """Normalize a value to canonical 2dp string for hashing (R3-D2)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        dec = value
    elif isinstance(value, int):
        dec = Decimal(value)
    elif isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        try:
            dec = Decimal(str(value))
        except InvalidOperation:
            return None
    else:
        text = str(value).strip().replace(",", "").replace(" ", "").replace("_", "")
        if not text:
            return None
        try:
            dec = Decimal(text)
        except InvalidOperation:
            # Unparseable: hash stripped raw for stability (documented).
            return str(value).strip()
    if not dec.is_finite():
        return None
    try:
        q = dec.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except InvalidOperation:
        return str(value).strip()
    return format(q, "f")


def canonical_key(
    data: Union[Provenance, dict, None] = None,
    /,
    *,
    document_id: Optional[str] = None,
    page: Optional[int] = None,
    element_id: Optional[str] = None,
    metric: Optional[str] = None,
    period: Optional[str] = None,
    period_canonical: Optional[str] = None,
    value: Any = None,
    value_str: Any = None,
    normalized_value: Any = None,
    source_label: Optional[str] = None,
    display_value: Optional[str] = None,
    precision: Optional[str] = None,
    extraction_route: Optional[str] = None,
) -> dict:
    """Return the canonical dict that is hashed to derive a record ID.

    Canonical key contains ONLY the 6 core fields (R3-D1 fix for
    cross-evidence stability + DuckDB dedup):
    ``document_id + page + element_id + metric + period + value``
    (``value`` = normalized 2dp string via ``_normalize_value``,
    ``period`` = canonical period via ``periods.parse_period`` when parseable,
    ``metric`` = casefolded). Evidence fields (``source_label`` /
    ``display_value`` / ``precision`` / ``extraction_route``) are intentionally EXCLUDED from the ID
    so re-extraction with richer evidence yields the same ``FIN-`` ID;
    they remain on ``Provenance`` for audit. ``page``/``element_id`` kept as
    ``None`` (JSON null) for explicitness.

    Inputs may come as a :class:`Provenance` instance, a plain ``dict``
    (keys: ``document_id``, ``page``, ``element_id``, ``metric``,
    ``period``/``period_canonical``, ``value``/``value_str``/
    ``normalized_value``), or explicit keyword args (kwargs win over ``data``
    when both are given and non-``None``). Unknown dict keys are ignored.
    ``source_label``/``display_value``/``precision``/``extraction_route`` kwargs are accepted for
    backward compatibility but IGNORED in the hash (documented breaking fix).

    Raises:
        ValueError: if ``document_id``/``metric``/``period``/``value`` is
            missing/empty, if ``document_id`` violates
            ``DOCUMENT_ID_PATTERN``, if conflicting aliases disagree
            (``period`` vs ``period_canonical``; ``value`` vs
            ``value_str`` vs ``normalized_value``), or if ``page`` is bool.
    """
    base: dict = {}
    if isinstance(data, Provenance):
        base = {
            "document_id": data.document_id,
            "page": data.page,
            "element_id": data.element_id,
            "metric": None,  # caller must supply metric (Provenance has no metric)
            "period": None,
            "value": data.normalized_value,
        }
    elif isinstance(data, dict):
        base = dict(data)
    elif data is not None:
        raise ValueError("data must be a Provenance, dict, or None.")

    def pick(*names: str) -> Any:
        for name in names:
            if name in base and base[name] is not None:
                return base[name]
        return None

    # Explicit kwargs override data-derived values when not None.
    doc_id_raw = document_id if document_id is not None else pick("document_id")
    page_raw = page if page is not None else pick("page")
    el_raw = element_id if element_id is not None else pick("element_id")
    metric_raw = metric if metric is not None else pick("metric")

    # Alias resolution with conflict detection (normalized comparison).
    # Collect each alias source separately so dicts carrying both
    # period + period_canonical (or value + value_str) are compared.
    raw_periods: list[Any] = [period, period_canonical]
    if isinstance(base.get("period"), str) and base.get("period") is not None:
        raw_periods.append(base.get("period"))
    if isinstance(base.get("period_canonical"), str) and base.get("period_canonical") is not None:
        raw_periods.append(base.get("period_canonical"))
    period_candidates = [_normalize_period(r) for r in raw_periods]
    period_candidates = [c for c in period_candidates if c is not None]
    if len(set(period_candidates)) > 1:
        raise ValueError(f"conflicting period aliases: {period_candidates!r}.")
    period_resolved = period_candidates[0] if period_candidates else None

    raw_values: list[Any] = [value, value_str, normalized_value]
    for k in ("value", "value_str", "normalized_value"):
        if k in base and base[k] is not None:
            raw_values.append(base[k])
    value_candidates = [_normalize_value(r) for r in raw_values]
    value_candidates = [c for c in value_candidates if c is not None]
    if len(set(value_candidates)) > 1:
        raise ValueError(f"conflicting value aliases: {value_candidates!r}.")
    value_resolved = value_candidates[0] if value_candidates else None

    document_id_s = _as_str(doc_id_raw)
    metric_s = _normalize_metric(metric_raw)
    if not document_id_s:
        raise ValueError("canonical_key requires non-empty 'document_id'.")
    if not re.match(DOCUMENT_ID_PATTERN, document_id_s):
        raise ValueError(f"canonical_key document_id {document_id_s!r} violates {DOCUMENT_ID_PATTERN}")
    if not metric_s:
        raise ValueError("canonical_key requires non-empty 'metric'.")
    if not period_resolved:
        raise ValueError("canonical_key requires non-empty 'period'/'period_canonical'.")
    if not value_resolved:
        raise ValueError("canonical_key requires non-empty 'value'/'value_str'/'normalized_value'.")
    if isinstance(page_raw, bool):
        raise ValueError("canonical_key page must not be bool")

    key: dict = {
        "document_id": document_id_s,
        "element_id": _as_str(el_raw),
        "metric": metric_s,
        "page": int(page_raw) if page_raw is not None else None,
        "period": period_resolved,
        "value": value_resolved,
    }
    return key


def record_id_for(data: Union[Provenance, dict, None] = None, /, **kwargs: Any) -> str:
    """Derive a deterministic record ID ``FIN-<12 hex>``.

    Hashes :func:`canonical_key` output with canonical JSON
    (``sort_keys=True``, ``separators=(",", ":")``, ``ensure_ascii=False``,
    UTF-8 bytes) via ``sha256``; the ID is the first 12 hex chars
    (lowercase, mirroring ingestion ``DOC-<12hex>``) prefixed with
    ``FIN-``. Same inputs -> same ID; no UUID/random/timestamps.

    Accepts the same inputs as :func:`canonical_key`: a ``dict`` /
    :class:`Provenance` positional arg, explicit kwargs, or both, e.g.::

        record_id_for({"document_id": "DEV-001", "page": 1,
                       "element_id": "e1", "metric": "revenue",
                       "period": "2023-01-01", "value": "100.00", ...})
        record_id_for(prov, metric="revenue", period="2023-01-01",
                      value_str="100.00")
    """
    key = canonical_key(data, **kwargs) if (data is not None or kwargs) else None
    if key is None:
        raise ValueError("record_id_for requires a Provenance/dict or keyword fields.")
    canonical = json.dumps(key, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"FIN-{digest[:12]}"


def _field_of(obj: Any, name: str, *aliases: str) -> Any:
    """Read ``name`` (or first present alias) from a dict or object; None if absent."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        if name in obj:
            return obj[name]
        for alias in aliases:
            if alias in obj:
                return obj[alias]
        return None
    for candidate in (name, *aliases):
        if hasattr(obj, candidate):
            return getattr(obj, candidate)
    return None


def _enum_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    text = str(raw).strip()
    return text if text else None


def build_provenance(
    document: Any = None,
    element: Any = None,
    *,
    source_label: str,
    display_value: str,
    normalized_value: Union[Decimal, str, None] = None,
    precision: str,
    document_id: Optional[str] = None,
    filename: Optional[str] = None,
    file_type: Optional[str] = None,
    language: Optional[str] = None,
    doc_type: Optional[str] = None,
    page: Optional[int] = None,
    sheet: Optional[str] = None,
    row: Optional[int] = None,
    column: Optional[int] = None,
    table_index: Optional[int] = None,
    element_id: Optional[str] = None,
    branch_id: Optional[str] = None,
    extraction_route: Optional[str] = None,
) -> Provenance:
    """Convenience constructor mapping ingestion Document/Element -> Provenance.

    Pure mapping, no I/O, no GT reads. ``document``/``element`` may be
    ingestion ``Document``/``Element`` dataclass instances or plain dicts
    (as returned by ``to_dict()``); explicit kwargs win over mapped values
    when both are given and non-``None``. ``language`` defaults from the
    document's ``language_hint``; ``file_type`` from ``file_type``.
    Row base note: ingestion ``Element.row`` is format-dependent (PDF/DOCX
    0-based, XLSX/CSV 1-based); native base is preserved, callers must not
    compare across formats. ``branch_id`` may be supplied explicitly
    (provenance has no ingestion source for it). ``extraction_route`` is
    explicit-kwarg-only (never auto-detected; sources never self-declare routes).
    """
    doc_id = document_id if document_id is not None else _as_str(_field_of(document, "document_id"))
    fname = filename if filename is not None else _field_of(document, "filename")
    ftype = file_type if file_type is not None else _enum_str(_field_of(document, "file_type"))
    lang = language if language is not None else _enum_str(_field_of(document, "language_hint", "language"))
    dtype = doc_type if doc_type is not None else _field_of(document, "doc_type")
    if dtype is not None:
        dtype = _enum_str(dtype)

    pg = page if page is not None else _field_of(element, "page")
    sh = sheet if sheet is not None else _field_of(element, "sheet")
    rw = row if row is not None else _field_of(element, "row")
    col = column if column is not None else _field_of(element, "column")
    ti = table_index if table_index is not None else _field_of(element, "table_index")
    el = element_id if element_id is not None else _field_of(element, "element_id")
    br = branch_id if branch_id is not None else _field_of(element, "branch_id", "branchId")

    if doc_id is None:
        raise ValueError("build_provenance requires 'document_id' (or document with document_id).")
    for coord_name, coord_val in (("page", pg), ("row", rw), ("column", col), ("table_index", ti)):
        if isinstance(coord_val, bool):
            raise ValueError(f"build_provenance {coord_name} must not be bool")

    norm: Union[Decimal, str, None] = normalized_value
    if isinstance(norm, str):
        norm = norm.strip() if norm.strip() else None  # type: ignore[assignment]

    return Provenance(
        document_id=str(doc_id).strip(),
        filename=str(fname).strip() if fname is not None else None,
        file_type=ftype,  # type: ignore[arg-type]
        page=pg,
        sheet=str(sh).strip() if sh is not None else None,
        row=rw,
        column=col,
        table_index=ti,
        element_id=str(el).strip() if el is not None else None,
        branch_id=str(br).strip() if br is not None else None,  # type: ignore[arg-type]
        language=lang,  # type: ignore[arg-type]
        doc_type=str(dtype).strip() if dtype is not None else None,
        source_label=source_label,
        display_value=display_value,
        normalized_value=norm,  # type: ignore[arg-type]
        precision=precision,  # type: ignore[arg-type]
        extraction_route=extraction_route,  # type: ignore[arg-type]
    )
