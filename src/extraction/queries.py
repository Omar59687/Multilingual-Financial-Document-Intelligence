"""Calculation-query library (MizanIQ Phase 3, bounded slice).

Contract: ``docs/STRUCTURED_EXTRACTION_DESIGN.md`` §20. Executable SQL
builders for the Evaluation Plan §9 evaluated operations over the
live DuckDB target, with live-DB logic tests on synthetic validated
records (see ``tests/extraction/test_queries.py``).

Reuse rule: company totals and per-branch long-form comparison
delegate to the frozen ``store.sql_company_total`` /
``store.sql_branch_comparison`` (read-only import, never modified).
Only uncovered §9 ops are built here: YoY growth/difference, period
averages, margins/ratios, executable budget variance (R8: computed,
never stored), branch rank+gap.

R6/R7 semantics: every branch aggregate filters
``branch_id IS NOT NULL`` so company-source NULL rows never
double-count; 2dp via ``ROUND(..., 2)``; deterministic ``ORDER BY``.
Builders are pure strings (never executed in-module); literal escaping
replicates the store escapers (attributed 6-line copy — single second
consumer, no shared change to the DONE module).

No ``duckdb`` import here (tests own the live dependency). No
ground-truth reads, no I/O, no network, no randomness.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Union

from .store import TABLE_NAME, sql_branch_comparison, sql_company_total

__all__ = [
    "sql_period_sum",
    "sql_period_average",
    "sql_yoy_growth",
    "sql_margin",
    "sql_budget_variance",
    "sql_branch_rank_gap",
]


def _escape_literal(text: str) -> str:
    """Escape a Python string for a SQL single-quoted literal (mirrors store)."""
    return text.replace("'", "''")


def _escape_like(text: str) -> str:
    """Escape LIKE wildcards/backslash in a prefix match (mirrors store)."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _require_str(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty str")
    return value


def _budget_literal(amount: Any) -> str:
    """Format an exact 2dp budget literal or raise (never float-nearest)."""
    if isinstance(amount, bool):
        raise TypeError("budget_amount must not be bool")
    if isinstance(amount, Decimal):
        dec = amount
    elif isinstance(amount, int):
        dec = Decimal(amount)
    elif isinstance(amount, float):
        dec = Decimal(str(amount))
    elif isinstance(amount, str):
        try:
            dec = Decimal(amount.strip().replace(",", ""))
        except InvalidOperation:
            raise ValueError(f"budget_amount unparseable: {amount!r}") from None
    else:
        raise TypeError(f"budget_amount got unsupported type: {type(amount).__name__}")
    if not dec.is_finite():
        raise ValueError("budget_amount must be finite")
    if dec.as_tuple().exponent != -2:
        raise ValueError(f"budget_amount {amount!r} must carry exactly 2dp")
    if abs(dec) >= Decimal("1000000000000"):
        raise ValueError("budget_amount abs must be < 1e12")
    return format(dec, "f")


def _branch_scope(branch_id: Any) -> str:
    """SQL fragment for an optional branch scope (validated literal)."""
    if branch_id is None:
        return "branch_id IS NOT NULL"
    if not isinstance(branch_id, str) or not branch_id.strip():
        raise TypeError("branch_id must be a BR-* string or None")
    return f"branch_id = '{_escape_literal(branch_id.strip())}'"


def sql_period_sum(metric: str, start: str, end: str, branch_id: str | None = None) -> str:
    """Company/branch sum of ``metric`` over an inclusive lexicographic window.

    Month-grain use (``YYYY-MM-01`` bounds). Company scope (``None``)
    sums branch rows only (R6); a branch scope reads that branch.
    """
    m = _escape_literal(_require_str("metric", metric))
    s = _escape_literal(_require_str("start", start))
    e = _escape_literal(_require_str("end", end))
    scope = _branch_scope(branch_id)
    return (
        'SELECT ROUND(COALESCE(SUM("value"), 0), 2) AS total '
        f"FROM {TABLE_NAME} "
        f"WHERE metric = '{m}' AND {scope} AND period >= '{s}' AND period <= '{e}';"
    )


def sql_period_average(metric: str, period_prefix: str, branch_id: str | None = None) -> str:
    """Average of ``metric`` monthly values under a prefix.

    Company scope (``None``) averages monthly company totals (branch
    rows summed per period first — a plain row average would vary with
    branch count); a branch scope averages that branch's month rows.
    """
    m = _escape_literal(_require_str("metric", metric))
    p = _escape_like(_escape_literal(_require_str("period_prefix", period_prefix)))
    if branch_id is None:
        return (
            "WITH monthly AS ("
            'SELECT period, SUM("value") AS t '
            f"FROM {TABLE_NAME} "
            f"WHERE metric = '{m}' AND branch_id IS NOT NULL AND period LIKE '{p}%' ESCAPE '\\' "
            "GROUP BY period) "
            "SELECT ROUND(AVG(t), 2) AS average FROM monthly;"
        )
    scope = _branch_scope(branch_id)
    return (
        'SELECT ROUND(AVG("value"), 2) AS average '
        f"FROM {TABLE_NAME} "
        f"WHERE metric = '{m}' AND {scope} AND period LIKE '{p}%' ESCAPE '\\';"
    )


def sql_yoy_growth(
    metric: str, period: str, prior_period: str, branch_id: str | None = None
) -> str:
    """YoY growth/difference: current, prior, absolute delta, pct (NULLIF)."""
    m = _escape_literal(_require_str("metric", metric))
    cur = _escape_literal(_require_str("period", period))
    prv = _escape_literal(_require_str("prior_period", prior_period))
    scope = _branch_scope(branch_id)
    return (
        "WITH cur AS ("
        f'SELECT ROUND(COALESCE(SUM("value"), 0), 2) AS v FROM {TABLE_NAME} '
        f"WHERE metric = '{m}' AND {scope} AND period = '{cur}'), "
        "prv AS ("
        f'SELECT ROUND(COALESCE(SUM("value"), 0), 2) AS v FROM {TABLE_NAME} '
        f"WHERE metric = '{m}' AND {scope} AND period = '{prv}') "
        "SELECT cur.v AS current, prv.v AS prior, "
        "ROUND(cur.v - prv.v, 2) AS delta, "
        "ROUND((cur.v - prv.v) / NULLIF(prv.v, 0) * 100, 2) AS pct "
        "FROM cur, prv;"
    )


def sql_margin(
    numerator_metric: str, denominator_metric: str, period: str, branch_id: str | None = None
) -> str:
    """Margin/ratio ×100 at one period (gross/net margin, expense ratios)."""
    num = _escape_literal(_require_str("numerator_metric", numerator_metric))
    den = _escape_literal(_require_str("denominator_metric", denominator_metric))
    p = _escape_literal(_require_str("period", period))
    scope = _branch_scope(branch_id)
    return (
        "WITH num AS ("
        f'SELECT ROUND(COALESCE(SUM("value"), 0), 2) AS v FROM {TABLE_NAME} '
        f"WHERE metric = '{num}' AND {scope} AND period = '{p}'), "
        "den AS ("
        f'SELECT ROUND(COALESCE(SUM("value"), 0), 2) AS v FROM {TABLE_NAME} '
        f"WHERE metric = '{den}' AND {scope} AND period = '{p}') "
        "SELECT num.v AS numerator, den.v AS denominator, "
        "ROUND(num.v / NULLIF(den.v, 0) * 100, 2) AS pct "
        "FROM num, den;"
    )


def sql_budget_variance(
    metric: str,
    period_prefix: str,
    budget_amount: Union[Decimal, int, float, str],
    branch_id: str | None = None,
) -> str:
    """Budget variance over a window: actual (derived), variance, pct.

    R8: ``actual`` aggregates the window (never stored); the budget
    binds client-side as an exact 2dp literal; zero budget yields NULL
    pct (never divide by zero).
    """
    m = _escape_literal(_require_str("metric", metric))
    p = _escape_like(_escape_literal(_require_str("period_prefix", period_prefix)))
    budget = _budget_literal(budget_amount)
    scope = _branch_scope(branch_id)
    return (
        "WITH actual AS ("
        f'SELECT ROUND(COALESCE(SUM("value"), 0), 2) AS v FROM {TABLE_NAME} '
        f"WHERE metric = '{m}' AND {scope} AND period LIKE '{p}%' ESCAPE '\\') "
        f"SELECT actual.v AS actual, {budget} AS budget, "
        f"ROUND(actual.v - {budget}, 2) AS variance, "
        f"ROUND((actual.v - {budget}) / NULLIF({budget}, 0) * 100, 2) AS variance_pct "
        "FROM actual;"
    )


def sql_branch_rank_gap(metric: str, period: str) -> str:
    """Per-branch rank + gap vs leader at one period (branch comparisons)."""
    m = _escape_literal(_require_str("metric", metric))
    p = _escape_literal(_require_str("period", period))
    return (
        'SELECT branch_id, "value", '
        'RANK() OVER (ORDER BY "value" DESC) AS rank, '
        'ROUND(MAX("value") OVER () - "value", 2) AS gap_vs_leader '
        f"FROM {TABLE_NAME} "
        f"WHERE metric = '{m}' AND period = '{p}' AND branch_id IS NOT NULL "
        "ORDER BY rank, branch_id;"
    )
