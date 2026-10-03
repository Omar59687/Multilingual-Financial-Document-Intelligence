"""Deterministic tests for src/extraction/schemas.py (R1-B5 coverage fix)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pytest
from pydantic import ValidationError

from extraction.provenance import Provenance
from extraction.schemas import (
    ExtractionResult,
    FinancialRecord,
    normalize_currency,
)


def _prov(**kw):
    base = dict(document_id="DEV-001", page=1, element_id="e1",
                source_label="Revenue", display_value="100.00",
                normalized_value="100.00", precision="exact-visible")
    base.update(kw)
    return Provenance.model_validate(base)


def _rec(**kw):
    base = dict(record_id="FIN-abcdef123456", metric="revenue",
                period="2023-01-01", fiscal_year=2023, value="100.00",
                currency="SAR", document_id="DEV-001", page=1,
                source_label="Revenue", display_value="100.00",
                precision="exact-visible", provenance=_prov())
    base.update(kw)
    return FinancialRecord.model_validate(base)


def test_currency_table():
    assert normalize_currency("sar") == "SAR"
    assert normalize_currency("SAR.") == "SAR"
    assert normalize_currency("ريال") == "SAR"
    with pytest.raises(ValueError):
        normalize_currency("USD")


def test_money_strict_2dp():
    _rec(value="100.00")
    for bad in ("100", "100.1", "100.123", "100.000"):
        with pytest.raises(ValidationError):
            _rec(value=bad)


def test_exact_visible_matrix():
    _rec(display_value="52,904.62", value="52904.62",
         provenance=_prov(display_value="52,904.62", normalized_value="52904.62"))
    # coarse display claiming exact -> reject
    for disp in ("100", "100.1", "100.123"):
        with pytest.raises(ValidationError):
            _rec(display_value=disp, value="100.00",
                 provenance=_prov(display_value=disp, normalized_value="100.00"))
    # suffixed claiming exact -> reject
    with pytest.raises(ValidationError):
        _rec(display_value="23.90M SAR", value="23900000.00",
             provenance=_prov(display_value="23.90M SAR", normalized_value="23900000.00"))
    # foreign currency claiming exact -> reject
    with pytest.raises(ValidationError):
        _rec(display_value="100 USD", value="100.00",
             provenance=_prov(display_value="100 USD", normalized_value="100.00"))


def test_record_id_canonical_fin_only():
    # Canonical ID is FIN-<12hex> only (scope correction §5: REC-* retired).
    _rec(record_id="FIN-abcdef123456")
    for bad in ("R-001", "REC-DEV-001-0001", "FIN-ABCDEF123456", "FIN-abc"):
        with pytest.raises(ValidationError):
            _rec(record_id=bad)


def test_record_id_matches_content_address():
    # The accepted ID is the content address of the normalized core fields.
    from extraction.provenance import record_id_for
    rid = record_id_for({"document_id": "DEV-001", "page": 1, "element_id": "e1",
                         "metric": "revenue", "period": "2023-01-01", "value": "100.00"})
    _rec(record_id=rid)


def test_provenance_coercion_and_consistency():
    r = _rec(provenance=dict(document_id="DEV-001", page=1, element_id="e1",
                             source_label="Revenue", display_value="100.00",
                             normalized_value="100.00", precision="exact-visible"))
    assert isinstance(r.provenance, Provenance)
    with pytest.raises(ValidationError):
        _rec(document_id="DEV-002")  # mismatches provenance DEV-001
    with pytest.raises(ValidationError):
        _rec(provenance=5)


def test_fiscal_year_consistency():
    _rec(period="2023-Q4", fiscal_year=2023)
    with pytest.raises(ValidationError):
        _rec(period="2023-Q4", fiscal_year=2022)


def test_display_rounded_consistency():
    _rec(display_value="23.90M SAR", value="23900000.00", precision="display-rounded",
         provenance=_prov(display_value="23.90M SAR", normalized_value="23900000.00",
                          precision="display-rounded"))
    with pytest.raises(ValidationError):
        _rec(display_value="23.90M SAR", value="23901234.56", precision="display-rounded",
             provenance=_prov(display_value="23.90M SAR", normalized_value="23900000.00",
                              precision="display-rounded"))


def test_extra_forbid_frozen_and_envelope():
    with pytest.raises(ValidationError):
        _rec(extra_field="x")
    r = _rec()
    with pytest.raises(ValidationError):
        r.record_id = "FIN-000000000000"
    env = ExtractionResult(document_id="DEV-001", records=[r], warnings=[], ok=True, latency_ms=1.5)
    assert env.ok is True
    with pytest.raises(ValidationError):
        ExtractionResult(document_id="DEV-001", records=[], warnings=[], ok=True, latency_ms=float("nan"))
