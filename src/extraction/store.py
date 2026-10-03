"""DuckDB TARGET DESIGN for approved financial records (MizanIQ Phase 3).

Foundation milestone — declarative ONLY (scope correction).

This module describes the DuckDB target shape that a LATER bounded Phase 3
task will implement persistence against. It performs NO persistence itself:

- no connections (no ``connect``),
- no database initialization (no ``init_db``),
- no record insertion (no ``insert_records``),
- no physical database file is ever created here,
- ``duckdb`` is not imported, not required, and must not be added to
  ``requirements.txt`` for this milestone.

Contents are pure data + pure functions (stdlib only):

- :data:`SCHEMA_SQL` — the proposed DDL (table + indexes + R6 view),
- :data:`TABLE_NAME` / :data:`VIEW_NAME` / :data:`COLUMNS`,
- :func:`ddl_statements` — deterministic split of :data:`SCHEMA_SQL`,
- :func:`sql_company_total` / :func:`sql_branch_comparison` /
  :func:`sql_budget_variance_note` — DOCUMENTED query patterns (plain SQL
  strings, never executed here) showing how deterministic calculations run
  in SQL per Evaluation Plan §9. They exist so the design is reviewable
  without a live database.

Contract reminder: only pre-validated records (see ``validation.py``) may
ever be written to this target. Validation lives in ``validation.py``;
this module performs no business validation. Corrupted ingest must never be
able to self-grade: the grader recomputes from frozen truth.

Full design: ``docs/STRUCTURED_EXTRACTION_DESIGN.md`` §12.
"""

from __future__ import annotations

__all__ = [
    "SCHEMA_SQL",
    "TABLE_NAME",
    "VIEW_NAME",
    "COLUMNS",
    "ddl_statements",
    "sql_company_total",
    "sql_branch_comparison",
    "sql_budget_variance_note",
]

TABLE_NAME = "financial_records"
VIEW_NAME = "v_monthly_company_totals"

# Canonical column order (matches the DDL below).
COLUMNS: tuple[str, ...] = (
    "record_id",
    "metric",
    "period",
    "fiscal_year",
    "value",
    "currency",
    "branch_id",
    "department_id",
    "document_id",
    "page",
    "source_label",
    "display_value",
    "precision",
    "created_batch",
)

SCHEMA_SQL: str = """-- MizanIQ Phase 3 structured store (DuckDB, SAR-only, branch-month grain)
-- TARGET DESIGN ONLY: no persistence behavior in this milestone
-- Approved-only writes: only validated records enter this table
-- Deterministic numerical calculations run in SQL over this table
-- NOTE Dammam (BR-DMM) pre-2019 rows must never be inserted at all
-- That absence rule is enforced at the validation layer, not by a DB trigger
-- Currency is SAR only and values are stored at 2dp as DECIMAL
-- Derived formulas R1-R3 and invoice total equals subtotal plus VAT are
-- checked before insert, budget rows store budget_amount only with actual
-- always derived by aggregation, never stored
-- NOTE value and precision are quoted ("value","precision") because both are
-- fragile keywords across DuckDB versions and drivers (R4-F1)
CREATE TABLE IF NOT EXISTS financial_records (
    record_id VARCHAR PRIMARY KEY,
    metric VARCHAR NOT NULL,
    period VARCHAR NOT NULL,
    fiscal_year INTEGER NOT NULL CHECK (fiscal_year BETWEEN 2015 AND 2024),
    "value" DECIMAL(18, 2) NOT NULL,
    currency VARCHAR NOT NULL DEFAULT 'SAR' CHECK (currency = 'SAR'),
    branch_id VARCHAR CHECK (branch_id IS NULL OR branch_id IN ('BR-RUH', 'BR-JED', 'BR-DMM')),
    department_id VARCHAR,
    document_id VARCHAR NOT NULL,
    page INTEGER CHECK (page IS NULL OR page >= 1),
    source_label VARCHAR NOT NULL,
    display_value VARCHAR NOT NULL,
    "precision" VARCHAR NOT NULL CHECK ("precision" IN ('exact-visible', 'display-rounded', 'unknown')),
    created_batch VARCHAR
);
CREATE INDEX IF NOT EXISTS idx_metric_period ON financial_records (metric, period);
CREATE INDEX IF NOT EXISTS idx_branch_period ON financial_records (branch_id, period);
CREATE INDEX IF NOT EXISTS idx_document ON financial_records (document_id);
-- R6 helper view: company-month totals are the sum over branch-month rows
-- Company-level source rows with NULL branch_id are excluded from the sum
-- so they can never double-count against their own branch constituents
-- SCOPE: this view is branch-sum ONLY, not a company-source reader.
-- Company-grain rows (annual balance sheet, company budgets with NULL
-- branch_id) persist in the table — read them with a direct SELECT on
-- branch_id IS NULL, never from this view (R4-F5).
CREATE OR REPLACE VIEW v_monthly_company_totals AS
SELECT metric, period, ROUND(SUM("value"), 2) AS company_total, COUNT(*) AS branch_count
FROM financial_records
WHERE branch_id IS NOT NULL
GROUP BY metric, period;
"""


def ddl_statements() -> list[str]:
    """Split :data:`SCHEMA_SQL` into individual declarative statements.

    Pure string split — no database connection and no third-party package
    required. Each returned string is stripped of surrounding whitespace.
    """
    return [part.strip() for part in SCHEMA_SQL.split(";") if part.strip()]


def _escape_literal(text: str) -> str:
    """Escape a Python string for embedding as a SQL single-quoted literal."""
    return text.replace("'", "''")


def _escape_like(text: str) -> str:
    """Escape LIKE wildcards (%, _) and backslash in a prefix match."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def sql_company_total(metric: str, period_prefix: str) -> str:
    """Documented query pattern: company totals of ``metric`` over a prefix.

    Sums branch-month rows only (``branch_id IS NOT NULL``, R6: sum over
    branches equals company total, R7: month sums roll into year sums) with
    ``ROUND(..., 2)`` preserving the SAR 2dp invariant (Evaluation Plan §9).

    Returns a SQL string only — never executed in this milestone. Example::

        sql_company_total('revenue', '2023')
    """
    if not isinstance(metric, str) or not isinstance(period_prefix, str):
        raise TypeError("metric and period_prefix must both be str")
    m = _escape_literal(metric)
    p = _escape_like(_escape_literal(period_prefix))
    return (
        'SELECT period, ROUND(SUM("value"), 2) AS company_total '
        f"FROM {TABLE_NAME} "
        f"WHERE metric = '{m}' AND branch_id IS NOT NULL AND period LIKE '{p}%' ESCAPE '\\' "
        "GROUP BY period ORDER BY period;"
    )


def sql_branch_comparison(metric: str, period: str) -> str:
    """Documented query pattern: per-branch ``metric`` at one ``period``.

    Long-form (one row per branch, ordered by branch) comparison for
    Evaluation Plan §9 branch-comparison questions. Exact-period match;
    branch rows only (``branch_id IS NOT NULL``).

    Returns a SQL string only — never executed in this milestone.
    """
    if not isinstance(metric, str) or not isinstance(period, str):
        raise TypeError("metric and period must both be str")
    m = _escape_literal(metric)
    p = _escape_literal(period)
    return (
        f'SELECT branch_id, "value" FROM {TABLE_NAME} '
        f"WHERE metric = '{m}' AND period = '{p}' AND branch_id IS NOT NULL "
        "ORDER BY branch_id;"
    )


def sql_budget_variance_note() -> str:
    """Documented query pattern note for budget variance (no DB needed).

    There is deliberately NO budget table in this target: budgets persist
    only ``budget_amount`` per quarter grain, while ``actual`` is always
    derived by aggregating :data:`TABLE_NAME` over the budget window.
    Variance (``actual - budget``) and variance percent are computed in SQL
    at query time and never stored (R8).
    """
    return (
        "-- Budget variance note (MizanIQ Phase 3, R8): no budget table lives in\n"
        "-- this DuckDB target. Budgets persist budget_amount only (quarter grain\n"
        "-- YYYY-Qn); actual is derived by aggregating financial_records over the\n"
        "-- budget window and variance is computed, never stored:\n"
        "--   actual = ROUND(SUM(value), 2) over the window\n"
        "--   variance = actual - budget_amount\n"
        "--   variance_pct = variance / NULLIF(budget_amount, 0)\n"
        "-- Example sketch (bind budget_amount client-side, never store actual):\n"
        "--   WITH actual AS (\n"
        "--     SELECT ROUND(SUM(value), 2) AS actual\n"
        "--     FROM financial_records\n"
        "--     WHERE metric = 'revenue' AND period LIKE '2023-Q1%'\n"
        "--   )\n"
        "--   SELECT actual, :budget_amount AS budget,\n"
        "--          actual - :budget_amount AS variance\n"
        "--   FROM actual;"
    )
