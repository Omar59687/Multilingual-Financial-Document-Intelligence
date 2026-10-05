"""G2 retrieval-measurement scoring (Phase 4 R6, evaluation-side only).

Implements the measurement half of docs/RETRIEVAL_DESIGN.md section 10
and docs/EVALUATION_PLAN.md sections 4/7/11 as pure stdlib scoring over
ranked hit lists. Never imported by product code; never imports
retrieval machinery (retrieval-blindness is asserted by
tests/eval/test_gold_integrity.py). The retrieval-running half lives in
scripts/measure_retrieval.py, which calls the accepted R4 unified
interface and scores with this module.

Frozen measurement protocol (design section 10):

- Metrics: Recall@10 (primary) + Recall@5 (primary) + NDCG@10 (graded).
  Recall@K = fraction of PRIMARY gold targets present in the top K.
  NDCG@10 uses graded gains PRIMARY_GAIN=2 / ACCEPTABLE_GAIN=1 with the
  ideal ordering (all primary targets first, then acceptable) for IDCG.
  The gain values are an R6-chosen PROVISIONAL measurement parameter
  (the design requires graded primary > acceptable but pins no numbers):
  they do not affect any current measurement (all frozen gold is
  primary-only) and remain PENDING explicit human/architect approval at
  the G2 gate (human policy gate preserved; not decided here). Single
  named constants so the architect can adjust without code changes.
- Same questions / same K (retrieval top_k=10 for every mode) / same
  gold for all three modes (bm25, semantic, hybrid).
- Slices: overall + per-language (ar/en from the gold ``lang`` field) +
  per-category (gold ``category``) + the minimum across language slices
  (EVAL section 7 rule: no blended-only claims).
- Latency p50/p95 per mode (nearest-rank percentiles over per-call
  timings; no thresholds).
- Thresholds TBD: this module reports measurements only and never
  passes/fails a gate.

Verification gating (EVAL section 2 + gold-file R6 instruction): only
items whose gold header verification record satisfies full sign-off
(``verified is True`` plus non-blank ``verified_by`` and
``verified_date`` per the gold verification schema) enter REPORTED
aggregates. ``verified=True`` alone is never sufficient; missing, null,
empty, or whitespace-only sign-off is excluded (never silently
counted). Unverified items keep their question-level evidence (hits +
provisional scores, each flagged) but are excluded from every reported
aggregate; an aggregate over zero verified items reports ``None`` with
an explicit reason (never a fake zero). G1 human sign-off flips items
to verified; G2 architect approval consumes the verified aggregates.

Split discipline: each split (dev file / eval file) is measured and
reported independently; splits are never pooled. Gold chunk IDs are
scoring-only inputs here (never passed to any retriever).
"""

from __future__ import annotations

import math
import time
from typing import Any

__all__ = [
    "MODES",
    "K_TOP",
    "K_RECALL_5",
    "K_RECALL_10",
    "K_NDCG",
    "PRIMARY_GAIN",
    "ACCEPTABLE_GAIN",
    "GRADE_GAINS",
    "REQUIRED_LANGS",
    "is_verified_record",
    "recall_at_k",
    "ndcg_at_k",
    "score_question",
    "percentile_nearest_rank",
    "latency_summary",
    "aggregate_metric",
    "slice_report",
    "measure_split",
]

#: Retrieval modes measured (design section 10: all three, same protocol).
MODES: tuple[str, ...] = ("bm25", "semantic", "hybrid")

#: Retrieval depth used for every mode (covers Recall@5/Recall@10/NDCG@10
#: from one ranked list; within the frozen 1..100 top_k contract).
K_TOP = 10
K_RECALL_5 = 5
K_RECALL_10 = 10
K_NDCG = 10

#: NDCG graded gains (R6-chosen parameter, flagged for G2 confirmation;
#: all frozen gold is primary-only so current outputs are unaffected).
#: HUMAN POLICY GATE: the contract requires primary > acceptable but does
#: NOT authorize a numeric gain mapping. PRIMARY_GAIN=2 / ACCEPTABLE_GAIN=1
#: remain PROVISIONAL pending explicit human/architect approval. Do NOT
#: silently approve, change, or remove this pending policy here.
PRIMARY_GAIN = 2
ACCEPTABLE_GAIN = 1
GRADE_GAINS = {"primary": PRIMARY_GAIN, "acceptable": ACCEPTABLE_GAIN}

#: Required language slices for the EVAL section 7 minimum rule.
#: Repository-defined set from the frozen gold schema (question ``lang``
#: in {"ar", "en"}; see tests/eval/test_gold_integrity.py schema checks
#: and gold header ``language_rule``). Do not invent additional languages.
REQUIRED_LANGS: tuple[str, ...] = ("ar", "en")


def _is_nonblank_str(value: Any) -> bool:
    """True only for str with non-whitespace content."""
    return isinstance(value, str) and bool(value.strip())


def is_verified_record(record: Any) -> bool:
    """True only when a gold verification record satisfies sign-off.

    Mirrors the authoritative gold verification schema asserted in
    tests/eval/test_gold_integrity.py::_assert_valid_signoff_state and
    documented in the gold headers (``verification`` map +
    ``r6_integration_note``):

    - ``record["verified"] is True`` alone is NOT sufficient;
    - ``record["verified_by"]`` must be a non-blank str (missing, null,
      empty, or whitespace-only is rejected);
    - ``record["verified_date"]`` must be a non-blank str (same rejects).

    Any malformed/incomplete record returns False (caller excludes it
    from VERIFIED/OFFICIAL aggregates; question-level provisional
    evidence is preserved separately). ``verified is False`` always
    returns False here.
    """
    if not isinstance(record, dict):
        return False
    if record.get("verified") is not True:
        return False
    return (_is_nonblank_str(record.get("verified_by"))
            and _is_nonblank_str(record.get("verified_date")))


def _is_verified_row(row: Any) -> bool:
    """Row-level gate for official aggregates (same sign-off rule).

    ``measure_split``/``build_mode_reports`` propagate ``verified_by`` /
    ``verified_date`` onto rows so aggregates can re-validate. Rows with
    bare ``verified=True`` but missing/null/blank sign-off are treated
    as unverified (excluded from official aggregates).
    """
    return is_verified_record(row)


def recall_at_k(retrieved_ids: list, primary_ids: list, k: int) -> float:
    """Fraction of PRIMARY gold targets present in the top K.

    Empty ``primary_ids`` returns 0.0 (the frozen gold contract always
    carries >= 1 primary target per question; the denominator is
    recorded on the question row for audit).
    """
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise TypeError("k must be a positive int, got %r" % (k,))
    if not primary_ids:
        return 0.0
    top = set(retrieved_ids[:k])
    hits = sum(1 for cid in primary_ids if cid in top)
    return hits / len(primary_ids)


def ndcg_at_k(retrieved_ids: list, gains: dict, k: int) -> float:
    """Graded NDCG@K with PRIMARY_GAIN/ACCEPTABLE_GAIN relevance.

    ``gains`` maps gold chunk_id -> gain (primary 2, acceptable 1);
    unlisted retrieved IDs gain 0. IDCG comes from the ideal ordering
    (gains descending) truncated at K. Zero IDCG (no gold) returns 0.0.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise TypeError("k must be a positive int, got %r" % (k,))
    if not isinstance(gains, dict):
        raise TypeError("gains must be a dict, got %s"
                        % type(gains).__name__)
    dcg = 0.0
    for rank, cid in enumerate(retrieved_ids[:k], start=1):
        gain = gains.get(cid, 0)
        if gain:
            dcg += gain / math.log2(rank + 1)
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(gain / math.log2(rank + 1)
               for rank, gain in enumerate(ideal, start=1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def score_question(retrieved_ids: list, targets: list) -> dict:
    """Score one question's ranked IDs against graded gold targets.

    ``targets`` is the frozen gold list of ``{chunk_id, grade}``;
    grades outside {primary, acceptable} raise ``TypeError`` (gold
    shape is enforced, never coerced). Returns recall@5, recall@10,
    ndcg@10 plus audit denominators. Deterministic.
    """
    if not isinstance(retrieved_ids, list):
        raise TypeError("retrieved_ids must be a list, got %s"
                        % type(retrieved_ids).__name__)
    if not isinstance(targets, list):
        raise TypeError("targets must be a list, got %s"
                        % type(targets).__name__)
    primary_ids = []
    gains = {}
    for position, target in enumerate(targets):
        if not isinstance(target, dict):
            raise TypeError("target %d must be a dict, got %s"
                            % (position, type(target).__name__))
        cid = target.get("chunk_id")
        grade = target.get("grade")
        if not isinstance(cid, str):
            raise TypeError("target %d has non-str chunk_id: %r"
                            % (position, cid))
        if grade not in GRADE_GAINS:
            raise TypeError("target %d has unknown grade: %r "
                            "(expected 'primary'/'acceptable')"
                            % (position, grade))
        if cid not in gains:
            if grade == "primary":
                primary_ids.append(cid)
            gains[cid] = GRADE_GAINS[grade]
    return {
        "recall@5": recall_at_k(retrieved_ids, primary_ids, K_RECALL_5),
        "recall@10": recall_at_k(retrieved_ids, primary_ids, K_RECALL_10),
        "ndcg@10": ndcg_at_k(retrieved_ids, gains, K_NDCG),
        "n_primary": len(primary_ids),
        "n_targets": len(gains),
    }


def percentile_nearest_rank(values: list, pct: float) -> float | None:
    """Nearest-rank percentile (pct in [0, 100]); None for empty input.

    Rank = ceil(pct/100 * n) over ascending sorted values (pct=0 takes
    the minimum). Deterministic; documented for EVAL section 11
    p50/p95 reporting.
    """
    if not isinstance(pct, (int, float)) or isinstance(pct, bool):
        raise TypeError("pct must be a number, got %r" % (pct,))
    if not 0 <= pct <= 100:
        raise TypeError("pct must satisfy 0 <= pct <= 100, got %r" % (pct,))
    if not values:
        return None
    ordered = sorted(values)
    if pct == 0:
        return float(ordered[0])
    rank = math.ceil(pct / 100.0 * len(ordered))
    return float(ordered[max(rank, 1) - 1])


def latency_summary(latencies: list) -> dict:
    """p50/p95 (+n/min/max) over per-call latency_ms values."""
    for position, value in enumerate(latencies):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError("latency %d must be a number, got %r"
                            % (position, value))
    if not latencies:
        return {"n": 0, "p50": None, "p95": None,
                "min": None, "max": None}
    return {
        "n": len(latencies),
        "p50": percentile_nearest_rank(latencies, 50),
        "p95": percentile_nearest_rank(latencies, 95),
        "min": float(min(latencies)),
        "max": float(max(latencies)),
    }


def aggregate_metric(rows: list, metric: str,
                     verified_only: bool = True) -> dict:
    """Mean of ``metric`` over included question rows (reported aggregate).

    Official (``verified_only=True``, default) inclusion requires full
    verification sign-off per :func:`is_verified_record`: ``verified is
    True`` plus non-blank ``verified_by`` and ``verified_date``. Bare
    ``verified=True`` with missing/null/empty/whitespace-only sign-off
    is excluded (never silently counted). Unverified rows are excluded
    per EVAL section 2; zero included rows yields ``{"mean": None, "n":
    0, "reason": ...}`` (never a fake zero). ``verified_only=False`` is
    available for explicitly labelled provisional views, never for gate
    evidence.
    """
    included = [row for row in rows
                if (_is_verified_row(row) or not verified_only)]
    excluded = len(rows) - len(included)
    if not included:
        return {"mean": None, "n": 0, "n_excluded": excluded,
                "reason": "no verified gold (pending G1 human check)"}
    values = [row[metric] for row in included]
    return {"mean": sum(values) / len(values), "n": len(included),
            "n_excluded": excluded, "reason": None}


_METRICS = ("recall@5", "recall@10", "ndcg@10")


def slice_report(question_rows: list, verified_only: bool = True) -> dict:
    """Overall + per-language + per-category aggregates + language minimum.

    ``question_rows`` carry ``lang``, ``category``, ``verified``,
    ``verified_by``/``verified_date`` (see :func:`is_verified_record`),
    and the three metric values. The ``min`` block reports the minimum
    across REQUIRED language slices per metric (EVAL section 7 rule;
    ``REQUIRED_LANGS``). A deficient/missing required slice is never
    dropped or averaged away: while any required language slice lacks
    verified evidence (mean None under the same ``verified_only`` mode),
    the minimum stays None with an explicit incomplete-coverage reason.
    Only once every required slice has verified evidence is the minimum
    computed normally.
    """
    report: dict[str, Any] = {"overall": {},
                              "by_lang": {}, "by_category": {},
                              "min_lang_slice": {},
                              "min_lang_slice_reason": {},
                              "min_lang_slice_missing": {}}
    for metric in _METRICS:
        report["overall"][metric] = aggregate_metric(
            question_rows, metric, verified_only=verified_only)
    langs = sorted({row["lang"] for row in question_rows})
    for lang in langs:
        group = [row for row in question_rows if row["lang"] == lang]
        report["by_lang"][lang] = {
            metric: aggregate_metric(group, metric,
                                     verified_only=verified_only)
            for metric in _METRICS}
    categories = sorted({row["category"] for row in question_rows})
    for category in categories:
        group = [row for row in question_rows if row["category"] == category]
        report["by_category"][category] = {
            metric: aggregate_metric(group, metric,
                                     verified_only=verified_only)
            for metric in _METRICS}
    for metric in _METRICS:
        missing: list[str] = []
        values: list[float] = []
        for lang in REQUIRED_LANGS:
            agg = report["by_lang"].get(lang, {}).get(metric)
            if agg is None or agg.get("mean") is None:
                missing.append(lang)
            else:
                values.append(agg["mean"])
        if missing:
            report["min_lang_slice"][metric] = None
            report["min_lang_slice_reason"][metric] = (
                "incomplete language coverage: missing verified evidence "
                "for %s (pending G1 human check; minimum unavailable)"
                % (missing,))
            report["min_lang_slice_missing"][metric] = list(missing)
        else:
            report["min_lang_slice"][metric] = min(values)
            report["min_lang_slice_reason"][metric] = None
            report["min_lang_slice_missing"][metric] = []
    return report


def measure_split(retriever: Any, questions: list,
                  verification: dict) -> dict:
    """Run all MODES over one split's questions; return audit evidence.

    ``retriever`` is the accepted R4 unified interface (duck-typed:
    ``retrieve(query, top_k=K_TOP, mode=mode)`` plus per-call timing).
    Only ``question["text"]`` is passed to retrieval — gold chunk IDs
    are scoring-only inputs and never enter a retrieval call. Returns
    per-mode, per-question rows with ranked hit IDs, latency_ms, gold
    targets, verified flag, and scores. Deterministic given a
    deterministic retriever (timings excluded from the claim).
    """
    if not isinstance(questions, list):
        raise TypeError("questions must be a list, got %s"
                        % type(questions).__name__)
    if not isinstance(verification, dict):
        raise TypeError("verification must be a dict, got %s"
                        % type(verification).__name__)
    modes = list(MODES)
    evidence: dict[str, Any] = {"modes": modes, "top_k": K_TOP,
                                "questions": []}
    for question in questions:
        if not isinstance(question, dict):
            raise TypeError("question must be a dict, got %s"
                            % type(question).__name__)
        qid = question.get("qid")
        text = question.get("text")
        if not isinstance(qid, str) or not isinstance(text, str):
            raise TypeError("question needs str qid/text, got %r" % (question,))
        record = verification.get(qid, {})
        verified = is_verified_record(record)
        verified_by = record.get("verified_by") if isinstance(record, dict) else None
        verified_date = record.get("verified_date") if isinstance(record, dict) else None
        row: dict[str, Any] = {
            "qid": qid, "lang": question.get("lang"),
            "category": question.get("category"), "doc": question.get("doc"),
            "verified": verified,
            "verified_by": verified_by,
            "verified_date": verified_date,
            "targets": question.get("targets"),
            "per_mode": {},
        }
        for mode in modes:
            started = time.perf_counter()
            hits = retriever.retrieve(text, top_k=K_TOP, mode=mode)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if not isinstance(hits, list):
                raise TypeError("retriever returned non-list for %s/%s"
                                % (qid, mode))
            retrieved_ids = [hit["chunk_id"] for hit in hits]
            row["per_mode"][mode] = {
                "retrieved_ids": retrieved_ids,
                "ranks": [hit["rank"] for hit in hits],
                "scores": [hit["score"] for hit in hits],
                "sources": [hit["source"] for hit in hits],
                "latency_ms": elapsed_ms,
                "scores_provisional": score_question(
                    retrieved_ids, question.get("targets")),
            }
        evidence["questions"].append(row)
    return evidence


def build_mode_reports(evidence: dict,
                       verified_only: bool = True) -> dict:
    """Assemble per-mode slice reports + latency summaries from evidence.

    Rows are rebuilt per mode from the stored per-question evidence so
    aggregates stay auditable against question-level hits. Latency comes
    from the recorded per-call timings.
    """
    reports = {}
    for mode in evidence["modes"]:
        rows = []
        latencies = []
        for question in evidence["questions"]:
            entry = question["per_mode"][mode]
            rows.append({
                "qid": question["qid"], "lang": question["lang"],
                "category": question["category"],
                "verified": question["verified"],
                "verified_by": question.get("verified_by"),
                "verified_date": question.get("verified_date"),
                **entry["scores_provisional"],
            })
            latencies.append(entry["latency_ms"])
        reports[mode] = {
            "slices": slice_report(rows, verified_only=verified_only),
            "latency": latency_summary(latencies),
        }
    return reports
