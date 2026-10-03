"""Deterministic tests for src/extraction/provenance.py (R3-T1 coverage fix)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from extraction.provenance import (
    Provenance,
    build_provenance,
    canonical_key,
    record_id_for,
)


def test_golden_vector_stable():
    a = record_id_for({"document_id": "DEV-001", "page": 1, "element_id": "e1",
                       "metric": "revenue", "period": "2023-01-01", "value": "100.00"})
    b = record_id_for({"document_id": "DEV-001", "page": 1, "element_id": "e1",
                       "metric": "revenue", "period": "2023-01-01", "value": "100.00"})
    assert a == b
    assert a.startswith("FIN-") and len(a) == 16


def test_core_only_evidence_ignored():
    bare = record_id_for({"document_id": "DEV-001", "page": 1, "element_id": "e1",
                          "metric": "revenue", "period": "2023-01-01", "value": "100.00"})
    rich = record_id_for({"document_id": "DEV-001", "page": 1, "element_id": "e1",
                          "metric": "revenue", "period": "2023-01-01", "value": "100.00",
                          "source_label": "Revenue", "display_value": "100.00",
                          "precision": "exact-visible"})
    assert bare == rich  # R3-D1 fix: evidence excluded from ID


def test_value_period_metric_normalization():
    assert record_id_for({"document_id": "DEV-001", "page": 1, "element_id": "e1",
                          "metric": "Revenue", "period": "2023-q1", "value": "100.0"}) == \
           record_id_for({"document_id": "DEV-001", "page": 1, "element_id": "e1",
                          "metric": "revenue", "period": "2023-Q1", "value": "100.00"})


def test_conflict_aliases_raise():
    with pytest.raises(ValueError):
        canonical_key({"document_id": "DEV-001", "metric": "revenue",
                       "period": "2023-Q1", "period_canonical": "2023-Q2", "value": "1.00"})
    with pytest.raises(ValueError):
        canonical_key({"document_id": "DEV-001", "metric": "revenue",
                       "period": "2023-Q1", "value": "1.00", "value_str": "2.00"})


def test_document_pattern():
    Provenance(document_id="DEV-001", page=1, element_id="e1", source_label="R",
               display_value="1.00", normalized_value="1.00", precision="exact-visible")
    Provenance(document_id="DOC-abcdef123456", page=1, element_id="e1", source_label="R",
               display_value="1.00", normalized_value="1.00", precision="exact-visible")
    with pytest.raises(Exception):
        Provenance(document_id="BAD-1", page=1, element_id="e1", source_label="R",
                   display_value="1.00", normalized_value="1.00", precision="exact-visible")
    with pytest.raises(ValueError):
        canonical_key({"document_id": "BAD-1", "metric": "revenue",
                       "period": "2023-01-01", "value": "1.00"})


def test_exact_requires_value():
    with pytest.raises(Exception):
        Provenance(document_id="DEV-001", page=1, element_id="e1", source_label="R",
                   display_value="1.00", normalized_value=None, precision="exact-visible")
    with pytest.raises(Exception):
        Provenance(document_id="DEV-001", page=1, element_id="e1", source_label="R",
                   display_value="   ", normalized_value="1.00", precision="exact-visible")


def test_document_level_provenance_allowed():
    # Scope correction §4: locations optional "where applicable" (Rule 10).
    # Image-wide visuals / document-level facts are legitimate without a page.
    p = Provenance(document_id="DEV-001", source_label="R",
                   display_value="1.00", normalized_value="1.00", precision="exact-visible")
    assert p.page is None and p.element_id is None
    # Deterministic IDs remain stable with absent coordinates (null in JSON).
    a = record_id_for({"document_id": "DEV-001", "metric": "revenue",
                       "period": "2023-01-01", "value": "1.00"})
    b = record_id_for({"document_id": "DEV-001", "metric": "revenue",
                       "period": "2023-01-01", "value": "1.00"})
    assert a == b


def test_build_from_dicts():
    doc = {"document_id": "DEV-001", "filename": "DEV-001.pdf", "file_type": "pdf",
           "language_hint": "en"}
    el = {"page": 2, "element_id": "p2e3", "row": 0, "column": 1}
    p = build_provenance(doc, el, source_label="Revenue", display_value="10.00",
                         normalized_value="10.00", precision="exact-visible",
                         branch_id="BR-RUH")
    assert p.document_id == "DEV-001" and p.page == 2 and p.branch_id == "BR-RUH"
