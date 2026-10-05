"""Forecasting handoff (MizanIQ Phase 3 → Phase 8, bounded slice).

Contract: ``docs/STRUCTURED_EXTRACTION_DESIGN.md`` §21. Deterministic
monthly time-series preparation from validated DuckDB records. No
models, no metrics, no training here (all Phase 8): this module only
shapes approved records into ascending month-grain series.

Holdout rule: ``end_period`` defaults to ``"2023-12-01"`` so the 2024
forecasting holdout (Design §15, Evaluation Plan §10) is excluded
unless a caller explicitly opts into a later window. Absent months
stay absent (never zero-filled — the Dammam pre-opening NULL rule
flows through as missing points, never fabricated zeros).

No ``duckdb`` import here (callers/tests own the live dependency). No
ground-truth reads, no I/O, no network, no randomness.
"""

from __future__ import annotations

from typing import Any

from .periods import parse_period
from .store import TABLE_NAME

__all__ = [
    "DEFAULT_END_PERIOD",
    "fetch_monthly_series",
]

#: Default series end: last month before the 2024 forecasting holdout.
DEFAULT_END_PERIOD = "2023-12-01"


def _escape_literal(text: str) -> str:
    """Escape a Python string for a SQL single-quoted literal (mirrors store)."""
    return text.replace("'", "''")


def fetch_monthly_series(
    connection: Any,
    metric: str,
    *,
    branch_id: str | None = None,
    end_period: str = DEFAULT_END_PERIOD,
) -> list[tuple[str, Any]]:
    """Fetch an ascending month-grain ``[(period, Decimal)]`` series.

    ``connection`` is a caller-owned DuckDB handle on an initialized
    target (see ``store.init_db``). Company scope (``branch_id=None``)
    sums branch rows only (R6); a branch scope reads that branch.
    Only ``YYYY-MM-01`` month rows at or before ``end_period`` are
    returned. Months with no rows are absent from the output.
    """
    if connection is None or not hasattr(connection, "execute"):
        raise TypeError("fetch_monthly_series requires a DB-API connection with .execute")
    if not isinstance(metric, str) or not metric:
        raise TypeError("metric must be a non-empty str")
    if branch_id is not None and (not isinstance(branch_id, str) or not branch_id.strip()):
        raise TypeError("branch_id must be a BR-* string or None")
    if not isinstance(end_period, str) or not end_period.strip():
        raise TypeError("end_period must be a non-empty str")
    try:
        parsed = parse_period(end_period.strip())
    except ValueError as exc:
        raise ValueError(f"end_period unparseable: {exc}") from None
    if parsed["kind"] != "month":
        raise ValueError(f"end_period must be month grain YYYY-MM-01, got {end_period!r}")
    end = parsed["canonical"]
    m = _escape_literal(metric)
    e = _escape_literal(end)
    if branch_id is None:
        scope = "branch_id IS NOT NULL"
    else:
        scope = f"branch_id = '{_escape_literal(branch_id.strip())}'"
    sql = (
        'SELECT period, ROUND(SUM("value"), 2) AS total '
        f"FROM {TABLE_NAME} "
        f"WHERE metric = '{m}' AND {scope} AND period LIKE '____-__-01' "
        f"AND period <= '{e}' "
        "GROUP BY period ORDER BY period;"
    )
    return [(period, total) for period, total in connection.execute(sql).fetchall()]
