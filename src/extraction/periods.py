"""Financial period/date modeling (Phase 3 extraction).

Canonical model (canonical §2 — explicit dates only):

- Full ISO dates: ``YYYY-MM-DD`` for day-grain facts.
- Monthly aggregates: ``period`` is the first day ``YYYY-MM-01``
  (validators.py ``PERIOD_RE``).
- Quarterly budgets: ``YYYY-Qn`` with ``n`` in 1..4, range 2015-Q1..2024-Q4,
  with defined month bounds (Q1 Jan-Mar, Q2 Apr-Jun, Q3 Jul-Sep, Q4 Oct-Dec).
- Annual grain: ``YYYY`` (company-level balance sheet), range 2015..2024.
- Global admissible range: 2015-01-01..2024-12-31, mirroring
  ``src/dataset/config.py`` ``START_YEAR``/``END_YEAR`` (mirrored here as
  literals so this module stays stdlib-only and decoupled from the dataset
  package; keep in sync manually).
- 2024 is the final forecasting holdout. This module applies NO special
  casing for 2024 — bounds/parsing treat it like any other in-range year.
  Callers implementing train/holdout splits must handle that downstream.

Alignment with ``src/dataset/validators.py``:

- ``PERIOD_RE`` (``^(\\d{4})-(\\d{2})-01$``), ``QUARTER_RE``
  (``^(\\d{4})-Q([1-4])$``), ``DATE_RE`` (``^(\\d{4})-(\\d{2})-(\\d{2})$``)
  are reproduced EXACTLY here so extraction output passes validation.
  Quarter *parsing* additionally accepts a lowercase ``q`` (normalized to
  canonical uppercase ``Q``); the stored regexes stay uppercase-only to
  match validators.py byte-for-byte.

Dammam NULL rule (BR-DMM opened 2019-01-01):

- Pre-2019 periods for branch ``BR-DMM`` are NOT-APPLICABLE and must be
  represented as NULL / absent — NEVER zero-filled. ``is_active_branch_month``
  returns ``False`` for ``BR-DMM`` with ``period_month < "2019-01-01"`` so
  callers can skip those branch-months. Zero-filling would corrupt revenue
  aggregates and violate the structural-break invariant (cf. config.py
  ``opening_ramp_months`` and validators.py Dammam guards).

Out of scope:

- Month-name parsing (Arabic/English month names such as "مارس"/"March")
  is OUT OF SCOPE for this module. ``normalize_date_display`` handles only
  numeric forms (see its docstring). Name-based parsing belongs to a
  locale-aware layer, not here.

Determinism: all functions are pure — no ``now()``, no randomness, no
ground-truth reads. Stdlib only (``re``, ``datetime.date``,
``calendar.monthrange``).
"""

import calendar
import re
from datetime import date

PERIOD_KINDS = ("day", "month", "quarter", "year")

# Regexes reproduced EXACTLY from src/dataset/validators.py — do not alter.
PERIOD_RE = re.compile(r"^(\d{4})-(\d{2})-01$")
QUARTER_RE = re.compile(r"^(\d{4})-Q([1-4])$")
DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")

# Local parse helpers (not in validators.py):
_YEAR_RE = re.compile(r"^(\d{4})$")
_QUARTER_PARSE_RE = re.compile(r"^(\d{4})-[Qq]([1-4])$")

# Mirrors src/dataset/config.py START_YEAR / END_YEAR (kept as literals to
# keep this module stdlib-only and decoupled; update together on change).
START_YEAR = 2015
END_YEAR = 2024

# Branch registry mirror (cf. config.py BRANCHES). Unknown IDs raise.
KNOWN_BRANCHES = ("BR-RUH", "BR-JED", "BR-DMM")
DMM_OPEN_MONTH = "2019-01-01"  # BR-DMM opened_date; pre-open months inactive.

_QUARTER_MONTHS = {
    1: (1, 3),
    2: (4, 6),
    3: (7, 9),
    4: (10, 12),
}

_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _check_year_in_range(year: int, raw: object) -> None:
    if not (START_YEAR <= year <= END_YEAR):
        raise ValueError(
            f"invalid period {raw!r}: year {year} out of range "
            f"{START_YEAR}..{END_YEAR}"
        )


def _iso(y: int, m: int, d: int) -> str:
    return date(y, m, d).isoformat()


def parse_period(s: str) -> dict:
    """Parse one canonical period string into its normalized components.

    Accepted inputs (after stripping surrounding whitespace):

    - ``YYYY``                      -> kind ``"year"``
    - ``YYYY-Qn`` (``q`` ok)        -> kind ``"quarter"``, canonical ``YYYY-Qn``
    - ``YYYY-MM-01``                -> kind ``"month"`` (monthly aggregate)
    - any other valid ``YYYY-MM-DD`` -> kind ``"day"``

    Returns ``{"kind", "iso_start", "iso_end", "fiscal_year", "canonical"}``
    where ``canonical`` is the normalized form (``YYYY-MM-DD`` day,
    ``YYYY-MM-01`` month, ``YYYY-Qn`` quarter, ``YYYY`` year) and
    ``iso_start``/``iso_end`` are inclusive ``YYYY-MM-DD`` bounds.

    Raises:
        ValueError: on bare quarters (``"Q3"``), slash dates
            (``MM/DD/YYYY`` or ``DD/MM/YYYY`` — use
            ``normalize_date_display`` first), impossible calendar dates,
            bad quarter numbers, or years outside 2015..2024.
    """
    if not isinstance(s, str):
        raise ValueError(f"invalid period {s!r}: expected str, got {type(s).__name__}")
    t = s.strip()
    if not t:
        raise ValueError("invalid period '': empty string after stripping whitespace")
    if "/" in t:
        raise ValueError(
            f"invalid period {s!r}: slash-separated dates (MM/DD/YYYY or "
            "DD/MM/YYYY) are not periods; use normalize_date_display first"
        )

    m = _YEAR_RE.match(t)
    if m:
        year = int(m.group(1))
        _check_year_in_range(year, s)
        start, end = year_bounds(year)
        return {
            "kind": "year",
            "iso_start": start,
            "iso_end": end,
            "fiscal_year": year,
            "canonical": f"{year:04d}",
        }

    m = _QUARTER_PARSE_RE.match(t)
    if m:
        year, q = int(m.group(1)), int(m.group(2))
        _check_year_in_range(year, s)
        start, end = quarter_bounds(f"{year:04d}-Q{q}")
        return {
            "kind": "quarter",
            "iso_start": start,
            "iso_end": end,
            "fiscal_year": year,
            "canonical": f"{year:04d}-Q{q}",
        }
    if re.match(r"^[Qq][1-4]$", t):
        raise ValueError(
            f"invalid period {s!r}: bare quarter without year; "
            "expected YYYY-Qn with year in 2015..2024"
        )

    m = PERIOD_RE.match(t)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        _check_year_in_range(year, s)
        if not 1 <= month <= 12:
            raise ValueError(f"invalid period {s!r}: month {month:02d} out of 01..12")
        start, end = month_bounds(t)
        return {
            "kind": "month",
            "iso_start": start,
            "iso_end": end,
            "fiscal_year": year,
            "canonical": f"{year:04d}-{month:02d}-01",
        }

    m = DATE_RE.match(t)
    if m:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        _check_year_in_range(year, s)
        try:
            d = date(year, month, day)
        except ValueError:
            raise ValueError(f"invalid period {s!r}: not a real calendar date")
        iso = d.isoformat()
        return {
            "kind": "day",
            "iso_start": iso,
            "iso_end": iso,
            "fiscal_year": year,
            "canonical": iso,
        }

    raise ValueError(
        f"invalid period {s!r}: expected YYYY, YYYY-Qn, YYYY-MM-01, or "
        "YYYY-MM-DD with year in 2015..2024"
    )


def month_bounds(period_month: str) -> tuple:
    """Return inclusive ``(start, end)`` ISO dates for a ``YYYY-MM-01`` month.

    Last-day aware, including leap years (e.g. ``2020-02-01`` ->
    ``("2020-02-01", "2020-02-29")``; ``2023-02-01`` -> end ``2023-02-28``).

    Raises:
        ValueError: if input is not ``YYYY-MM-01``, has a month outside
            01..12, or a year outside 2015..2024.
    """
    if not isinstance(period_month, str):
        raise ValueError(
            f"invalid period_month {period_month!r}: expected str 'YYYY-MM-01'"
        )
    t = period_month.strip()
    m = PERIOD_RE.match(t)
    if not m:
        raise ValueError(
            f"invalid period_month {period_month!r}: expected 'YYYY-MM-01' "
            "(first day of month)"
        )
    year, month = int(m.group(1)), int(m.group(2))
    _check_year_in_range(year, period_month)
    if not 1 <= month <= 12:
        raise ValueError(
            f"invalid period_month {period_month!r}: month out of 01..12"
        )
    last = calendar.monthrange(year, month)[1]
    return (_iso(year, month, 1), _iso(year, month, last))


def quarter_bounds(period_q: str) -> tuple:
    """Return inclusive ``(start, end)`` ISO dates for a ``YYYY-Qn`` quarter.

    Mapping: Q1 Jan 01–Mar 31, Q2 Apr 01–Jun 30, Q3 Jul 01–Sep 30,
    Q4 Oct 01–Dec 31. E.g. ``2023-Q4`` -> ``("2023-10-01", "2023-12-31")``.
    Lowercase ``q`` accepted (``2023-q4``); surrounding whitespace stripped.

    Raises:
        ValueError: on bare quarters, quarter numbers outside 1..4, or
            years outside 2015..2024.
    """
    if not isinstance(period_q, str):
        raise ValueError(
            f"invalid quarter {period_q!r}: expected str 'YYYY-Qn'"
        )
    t = period_q.strip()
    m = _QUARTER_PARSE_RE.match(t)
    if not m:
        if re.match(r"^[Qq][1-4]$", t):
            raise ValueError(
                f"invalid quarter {period_q!r}: bare quarter without year; "
                "expected YYYY-Qn"
            )
        raise ValueError(
            f"invalid quarter {period_q!r}: expected 'YYYY-Qn' with "
            f"n in 1..4 and year in {START_YEAR}..{END_YEAR}"
        )
    year, q = int(m.group(1)), int(m.group(2))
    _check_year_in_range(year, period_q)
    m_start, m_end = _QUARTER_MONTHS[q]
    last = calendar.monthrange(year, m_end)[1]
    return (_iso(year, m_start, 1), _iso(year, m_end, last))


def year_bounds(year) -> tuple:
    """Return inclusive ``(start, end)`` ISO dates for a calendar year.

    Accepts ``int`` (e.g. ``2023``) or ``str`` (e.g. ``"2023"``, whitespace
    stripped). E.g. ``2023`` -> ``("2023-01-01", "2023-12-31")``.

    Raises:
        ValueError: if the year is not exactly four digits / an int, or is
            outside 2015..2024.
    """
    raw = year
    if isinstance(year, str):
        t = year.strip()
        if not _YEAR_RE.match(t):
            raise ValueError(
                f"invalid year {raw!r}: expected 'YYYY' with year in "
                f"{START_YEAR}..{END_YEAR}"
            )
        year = int(t)
    elif isinstance(year, bool) or not isinstance(year, int):
        raise ValueError(
            f"invalid year {raw!r}: expected int or str 'YYYY' with year in "
            f"{START_YEAR}..{END_YEAR}"
        )
    _check_year_in_range(year, raw)
    return (_iso(year, 1, 1), _iso(year, 12, 31))


def is_active_branch_month(branch_id, period_month: str) -> bool:
    """Report whether a branch was open in a given ``YYYY-MM-01`` month.

    - ``BR-DMM`` opened 2019-01-01: returns ``False`` for any
      ``period_month < "2019-01-01"`` (callers must SKIP / emit NULL, never
      zero-fill — pre-opening periods are not-applicable), ``True`` from
      2019-01 onward.
    - ``BR-RUH`` / ``BR-JED``: ``True`` for any in-range month.
    - Unknown ``branch_id`` raises ``ValueError`` (never silently ``False``).

    Raises:
        ValueError: on unknown branch IDs, or on malformed/out-of-range
            ``period_month`` (must be ``YYYY-MM-01`` in 2015..2024).
    """
    if branch_id not in KNOWN_BRANCHES:
        raise ValueError(
            f"unknown branch_id {branch_id!r}: expected one of "
            f"{list(KNOWN_BRANCHES)}"
        )
    start, _ = month_bounds(period_month)  # validates format + range
    if branch_id == "BR-DMM":
        return start >= DMM_OPEN_MONTH
    return True


def normalize_date_display(s: str) -> str:
    """Normalize a displayed date to ISO ``YYYY-MM-DD``.

    Steps: strip surrounding whitespace, map Arabic-Indic digits
    (``٠١٢٣٤٥٦٧٨٩`` -> ``0-9``) to Western digits, then accept:

    - ``YYYY-MM-DD`` and ``YYYY/MM/DD`` (year-first, 1–2 digit month/day),
    - ``DD/MM/YYYY`` (day-first, slash-separated, 1–2 digit day/month).

    Returns zero-padded ISO ``YYYY-MM-DD``. Day-first interpretation applies
    to ambiguous slash dates: ``03/04/2023`` reads as 3 April 2023
    (``DD/MM/YYYY``). US ``MM/DD/YYYY`` ordering is NOT supported — callers
    must not feed it here.

    Month-name parsing (Arabic/English names like "مارس"/"March") is
    explicitly OUT OF SCOPE and raises ``ValueError``.

    Raises:
        ValueError: on empty input, non-numeric/month-name input, impossible
            calendar dates (e.g. 2023-02-30), or dates outside
            2015-01-01..2024-12-31.
    """
    if not isinstance(s, str):
        raise ValueError(f"invalid date {s!r}: expected str")
    t = s.strip().translate(_ARABIC_INDIC)
    if not t:
        raise ValueError("invalid date '': empty string after stripping whitespace")

    m = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$", t)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    else:
        m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", t)
        if m:
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        else:
            raise ValueError(
                f"invalid date {s!r}: expected YYYY-MM-DD, YYYY/MM/DD, or "
                "DD/MM/YYYY (month names are out of scope)"
            )
    try:
        dt = date(y, mo, d)
    except ValueError:
        raise ValueError(f"invalid date {s!r}: not a real calendar date")
    iso = dt.isoformat()
    if not (f"{START_YEAR:04d}-01-01" <= iso <= f"{END_YEAR:04d}-12-31"):
        raise ValueError(
            f"invalid date {s!r}: resolved {iso} out of range "
            f"{START_YEAR}-01-01..{END_YEAR}-12-31"
        )
    return iso
