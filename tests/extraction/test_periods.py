"""Deterministic tests for src/extraction/periods.py (R2 coverage fix)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pytest

import extraction.periods as P
from dataset import config as C
from dataset import validators as DV


def test_regex_byte_identical():
    assert P.PERIOD_RE.pattern == DV.PERIOD_RE.pattern
    assert P.QUARTER_RE.pattern == DV.QUARTER_RE.pattern
    assert P.DATE_RE.pattern == DV.DATE_RE.pattern
    assert P.START_YEAR == C.START_YEAR
    assert P.END_YEAR == C.END_YEAR


def test_parse_quarter():
    out = P.parse_period("2023-Q4")
    assert out == {"kind": "quarter", "iso_start": "2023-10-01",
                   "iso_end": "2023-12-31", "fiscal_year": 2023, "canonical": "2023-Q4"}
    assert P.parse_period("2023-q4")["canonical"] == "2023-Q4"


def test_parse_month_is_month_not_day():
    out = P.parse_period("2023-03-01")
    assert out["kind"] == "month"
    assert out["iso_start"] == "2023-03-01" and out["iso_end"] == "2023-03-31"


def test_month_bounds_leap():
    assert P.month_bounds("2020-02-01") == ("2020-02-01", "2020-02-29")
    assert P.month_bounds("2023-02-01") == ("2023-02-01", "2023-02-28")
    assert P.month_bounds("2019-02-01") == ("2019-02-01", "2019-02-28")


def test_quarter_bounds():
    assert P.quarter_bounds("2023-Q4") == ("2023-10-01", "2023-12-31")
    assert P.quarter_bounds("2023-q1") == ("2023-01-01", "2023-03-31")


def test_year_bounds():
    assert P.year_bounds(2023) == ("2023-01-01", "2023-12-31")
    assert P.year_bounds("2024") == ("2024-01-01", "2024-12-31")
    with pytest.raises(ValueError):
        P.year_bounds(True)


def test_dammam_boundary():
    assert P.is_active_branch_month("BR-DMM", "2018-12-01") is False
    assert P.is_active_branch_month("BR-DMM", "2019-01-01") is True
    assert P.is_active_branch_month("BR-RUH", "2018-12-01") is True
    with pytest.raises(ValueError):
        P.is_active_branch_month("BR-XX", "2020-01-01")


def test_normalize_displays():
    assert P.normalize_date_display("2023-03-15") == "2023-03-15"
    assert P.normalize_date_display("2023/03/15") == "2023-03-15"
    assert P.normalize_date_display("15/03/2023") == "2023-03-15"
    assert P.normalize_date_display("١٥/٠٣/٢٠٢٣") == "2023-03-15"


def test_rejections():
    for bad in ("Q3", "03/04/2023x", "2023-02-30", "2014-01-01", "2025-Q1",
                "2015-Q5", "not-a-date", "15 مارس 2023"):
        with pytest.raises(ValueError):
            P.parse_period(bad) if "/" not in bad and "مارس" not in bad else P.normalize_date_display(bad)
    with pytest.raises(ValueError):
        P.parse_period("2021-13-01")
