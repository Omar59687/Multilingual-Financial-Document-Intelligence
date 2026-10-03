"""Rounding-policy boundary test (scope correction §3).

Project policy: ROUND_HALF_UP for normalization of source-visible financial
numbers (matches the dataset generator, dataset validators, and all of
``src/extraction``). ``src/ocrbench/canonical.py`` was unified to explicit
HALF_UP after verifying zero impact on saved Phase 2 outputs.

This test pins a halfway decimal (``1.005``) that differs between HALF_UP
(``1.01``) and HALF_EVEN (``1.00``) so the conflict cannot silently return:
both normalization entry points must agree.
"""

import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ocrbench.canonical import normalize_display_value as canonical_norm
from extraction.validation import _normalize_display_value as validation_norm


def test_halfway_boundary_agrees_half_up():
    c = canonical_norm("1.005")
    v = validation_norm("1.005")
    assert c["normalized_value"] == Decimal("1.01"), c
    assert v["normalized_value"] == Decimal("1.01"), v
    assert c["precision"] == "display-rounded"
    assert v["precision"] == "display-rounded"
    assert c["normalized_value"] == v["normalized_value"]


def test_exact_2dp_unaffected():
    assert canonical_norm("+52,904.62 SAR")["normalized_value"] == Decimal("52904.62")
    assert canonical_norm("+52,904.62 SAR")["precision"] == "exact-visible"
    assert canonical_norm("23.90M SAR")["normalized_value"] == Decimal("23900000")
    assert canonical_norm("23.90M SAR")["precision"] == "display-rounded"
