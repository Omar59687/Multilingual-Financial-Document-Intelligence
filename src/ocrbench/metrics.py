"""OCR/vision scoring primitives (evaluation-side, stdlib only).

Two independent layers — NEVER merged or averaged:

  LAYER A — RECOGNITION ("did the model see it in raw text?"):
    identifier_text_accuracy (score_identifier_text),
    numeric_text_exact_accuracy (score_numeric_text),
    table_label_text_recall (score_table_label_text).
    Raw OCR with no fields/tables CAN score here.

  LAYER B — STRUCTURING ("did the model attach it to the right
  field/row?"):
    identifier_structured_accuracy (score_identifiers),
    numeric_structured_exact_accuracy (score_numerics, exact),
    table_association_accuracy (score_table).
    Requires populated fields/tables; prose mentions do NOT count.

Conventions (see docs/OCR_VISION_DESIGN.md §6, EVALUATION_PLAN §3):
  - identifiers: exact after outer-whitespace strip (hyphens/case matter).
  - numerics: Decimal-2dp equality; every miss listed expected-vs-got.
  - anchors: normalized-substring recall.
  - CER/WER: Levenshtein-based; reported raw AND normalized.
  - tables: (label, value) association scoring — never plain-text only.
  - charts/KPIs: recognition (A) and semantic interpretation (B) scored
    separately and deterministically (no LLM judge).
"""

from decimal import Decimal
from typing import Mapping, Optional, Sequence
import re

from .normalize import normalize_amount, normalize_text


# --------------------------------------------------------------------------
# LAYER A — RECOGNITION vs LAYER B — STRUCTURING
#
# Recognition (text) metrics ask: "did the model visibly recover this
# string/number/label somewhere in its raw text?" Structuring (field)
# metrics ask: "did the model associate it with the correct semantic
# field/table row?" A raw OCR engine MUST get recognition credit without
# field/table objects, and MUST NOT get structured credit it did not
# produce. The two layers are never merged or averaged.
# --------------------------------------------------------------------------
def identifier_in_text(pred_text: str, expected: str) -> bool:
    """Strict substring check for one identifier in raw OCR text.

    Both sides go through normalize_text (whitespace collapse + documented
    Arabic folding) but NOTHING else: hyphens, case, and digit values stay
    significant; no fuzzy matching; no GT-driven parser logic.
    """
    if pred_text is None or expected is None:
        return False
    needle = normalize_text(expected)
    if not needle:
        return False
    return needle in normalize_text(pred_text or "")


def score_identifier_text(pred_text: str, expected: Mapping) -> dict:
    """LAYER A for identifiers: per-identifier text-presence scoring.

    Returns {per_field, n_total, n_correct, accuracy, misses} where misses
    list {field, expected, found_in_text: False}. This is the recognition
    companion to score_identifiers (which remains the LAYER B structured
    score and is reported as identifier_structured_accuracy).
    """
    per_field = {k: identifier_in_text(pred_text or "", v)
                 for k, v in (expected or {}).items()}
    total = len(per_field)
    correct = sum(1 for v in per_field.values() if v)
    return {
        "per_field": per_field,
        "n_total": total,
        "n_correct": correct,
        "accuracy": (correct / total) if total else 1.0,
        "misses": [{"field": k, "expected": expected[k],
                    "found_in_text": False}
                   for k, v in per_field.items() if not v],
    }


# --------------------------------------------------------------------------
# Identifiers — LAYER B (structured) is score_identifiers; LAYER A
# (recognition) is score_identifier_text above.
# --------------------------------------------------------------------------
def identifier_match(pred: str, expected: str) -> bool:
    """Byte-exact modulo outer whitespace. No case/hyphen folding, ever."""
    if pred is None or expected is None:
        return False
    return pred.strip() == expected.strip()


def score_identifiers(pred_fields: Mapping, expected: Mapping) -> dict:
    """LAYER B structured identifier score (reported as
    identifier_structured_accuracy). Exact-match per identifier field:
    {per_field, accuracy, misses}. Raw text presence does NOT count here —
    use score_identifier_text (identifier_text_accuracy) for Layer A."""
    per_field = {k: identifier_match(pred_fields.get(k), v)
                 for k, v in expected.items()}
    total = len(per_field)
    correct = sum(1 for v in per_field.values() if v)
    return {
        "per_field": per_field,
        "n_total": total,
        "n_correct": correct,
        "accuracy": (correct / total) if total else 1.0,
        "misses": [{"field": k, "expected": expected[k],
                    "got": pred_fields.get(k)} for k, v in per_field.items()
                   if not v],
    }


# --------------------------------------------------------------------------
# Numerics — LAYER B (structured) is score_numerics; LAYER A (recognition)
# is score_numeric_text below.
# --------------------------------------------------------------------------
def score_numerics(pred_fields: Mapping, expected: Mapping,
                   rel_tol: Optional[float] = None) -> dict:
    """LAYER B structured numeric score (reported as
    numeric_structured_exact_accuracy when rel_tol is None). 2dp-exact per
    numeric field. rel_tol (e.g. 0.005) is an explicit opt-in for OCR-noise
    slices only; default None = exact.

    Every miss is listed with expected vs. got (EVALUATION_PLAN §3).
    A correct number existing somewhere in raw prose does NOT count here —
    use score_numeric_text (numeric_text_exact_accuracy) for Layer A.
    """
    per_field, misses = {}, []
    for field, exp_raw in expected.items():
        try:
            exp = normalize_amount(exp_raw)
        except ValueError:
            per_field[field] = False
            misses.append({"field": field, "expected": exp_raw,
                           "got": pred_fields.get(field),
                           "reason": "bad expected value"})
            continue
        got_raw = pred_fields.get(field)
        try:
            got = normalize_amount(got_raw) if got_raw is not None else None
        except ValueError:
            got = None
        if got is not None and (got == exp or (
                rel_tol is not None and exp != 0
                and abs(got - exp) / abs(exp) <= Decimal(str(rel_tol)))):
            per_field[field] = True
        else:
            per_field[field] = False
            misses.append({"field": field, "expected": str(exp),
                           "got": (str(got) if got is not None
                                   else repr(got_raw))})
    total = len(per_field)
    correct = sum(1 for v in per_field.values() if v)
    return {
        "per_field": per_field,
        "n_total": total,
        "n_correct": correct,
        "accuracy": (correct / total) if total else 1.0,
        "misses": misses,
    }


# Matches one Western or Arabic-Indic digit, then a run of digits and
# intra-number separators (comma, period, Arabic separators, grouping
# space/NBSP), ending on a digit — plus a lone-digit fallback. Newlines,
# letters, hyphens, and currency words are NOT part of a token, so dates
# like 2019-06-09 split into 2019 / 06 / 09 and "24,371.25 SAR" yields
# "24,371.25". Every candidate is validated by normalize_amount; the
# regex never decides equality by itself.
_AMOUNT_TOKEN_RE = re.compile(
    r"[0-9\u0660-\u0669\u06F0-\u06F9]"
    r"[0-9\u0660-\u0669\u06F0-\u06F9,.\u066B\u066C \u00A0]*"
    r"[0-9\u0660-\u0669\u06F0-\u06F9]"
    r"|[0-9\u0660-\u0669\u06F0-\u06F9]")


def amounts_in_text(pred_text: str) -> set:
    """All 2dp-normalized amounts recoverable from raw OCR text.

    Each regex candidate is parsed with normalize_amount (same 2dp
    Decimal rules as structured scoring: ','/space/NBSP grouping removed,
    SAR tokens stripped, Arabic-Indic digits folded). Unparseable
    candidates are skipped. Digit substitutions NEVER match: equality is
    exact Decimal equality.
    """
    found = set()
    for match in _AMOUNT_TOKEN_RE.finditer(pred_text or ""):
        try:
            found.add(normalize_amount(match.group(0)))
        except ValueError:
            continue
    return found


def score_numeric_text(pred_text: str, expected: Mapping) -> dict:
    """LAYER A for numerics (reported as numeric_text_exact_accuracy).

    For each expected value: credit iff its canonical 2dp Decimal occurs
    among amounts_in_text(pred_text). "24,371.25" and "24371.25" are the
    same visible value (grouping-insensitive); "28026.95" never matches
    "28026.94" (no digit tolerance); ±0.5% is NEVER applied here — it
    remains a secondary structured diagnostic only.

    Means: "the model saw this correct number somewhere." It does NOT
    mean the number was assigned to the correct field (Layer B).
    """
    found = amounts_in_text(pred_text or "")
    per_field, misses = {}, []
    for field, exp_raw in (expected or {}).items():
        try:
            exp = normalize_amount(exp_raw)
        except ValueError:
            per_field[field] = False
            misses.append({"field": field, "expected": exp_raw,
                           "found_in_text": False,
                           "reason": "bad expected value"})
            continue
        ok = exp in found
        per_field[field] = ok
        if not ok:
            misses.append({"field": field, "expected": str(exp),
                           "found_in_text": False})
    total = len(per_field)
    correct = sum(1 for v in per_field.values() if v)
    return {
        "per_field": per_field,
        "n_total": total,
        "n_correct": correct,
        "accuracy": (correct / total) if total else 1.0,
        "misses": misses,
    }


# --------------------------------------------------------------------------
# Text anchors
# --------------------------------------------------------------------------
def score_anchors(pred_text: str, anchors: Sequence[str]) -> dict:
    """Normalized-substring recall of expected text anchors."""
    hay = normalize_text(pred_text or "")
    found = [a for a in anchors if normalize_text(a) in hay]
    missing = [a for a in anchors if normalize_text(a) not in hay]
    total = len(anchors)
    return {
        "n_total": total,
        "n_found": len(found),
        "recall": (len(found) / total) if total else 1.0,
        "found": found,
        "missing": missing,
    }


# --------------------------------------------------------------------------
# CER / WER
# --------------------------------------------------------------------------
def _levenshtein(a: Sequence, b: Sequence) -> int:
    """Single-row Levenshtein distance (stdlib; refs are short strings)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str, normalized: bool = True) -> float:
    """Char error rate = edit distance / reference chars (1.0 if empty)."""
    if normalized:
        reference, hypothesis = normalize_text(reference), \
            normalize_text(hypothesis or "")
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _levenshtein(reference, hypothesis or "") / max(len(reference), 1)


def wer(reference: str, hypothesis: str, normalized: bool = True) -> float:
    """Word error rate over whitespace tokens (1.0 if empty reference)."""
    if normalized:
        reference, hypothesis = normalize_text(reference), \
            normalize_text(hypothesis or "")
    ref_tokens = reference.split()
    if not ref_tokens:
        return 0.0 if not (hypothesis or "").split() else 1.0
    return _levenshtein(ref_tokens,
                        (hypothesis or "").split()) / len(ref_tokens)


# --------------------------------------------------------------------------
# Tables: (row-label, value) association scoring
# --------------------------------------------------------------------------
def pairs_from_table_grid(tables: Sequence[Sequence[Sequence[str]]],
                          expected_labels: Sequence[str]) -> list:
    """Adapt a model's recovered grid to (label, value) pairs.

    tables: list of tables; each table a list of rows; each row a list of
    cell strings. For every expected label, takes the FIRST row whose
    joined normalized text contains the normalized label, and the LAST
    cell in that row that parses as an amount. Rows without a parseable
    amount yield no pair (visible as missing labels downstream).
    Deterministic; no guessing beyond this documented rule.
    """
    pairs = []
    for label in expected_labels:
        needle = normalize_text(label)
        for table in tables or []:
            matched = False
            for row in table or []:
                cells = [str(c) for c in row]
                if needle and needle in normalize_text(" ".join(cells)):
                    for cell in reversed(cells):
                        try:
                            normalize_amount(cell)
                        except ValueError:
                            continue
                        pairs.append((label, cell))
                        matched = True
                        break
                if matched:
                    break
            if matched:
                break
    return pairs


def score_table(pred_pairs: Sequence[tuple],
                expected_pairs: Sequence[tuple]) -> dict:
    """Score table recovery as label/value associations.

    pred_pairs / expected_pairs: (label_string, value_string) each.
    A pair counts correct ONLY when the normalized label matches AND the
    2dp value matches — right digits on the wrong row are incorrect.
    Reports label recall, value accuracy (on matched labels), and
    association accuracy separately (never plain-text OCR alone).
    """
    exp_labels = [normalize_text(label) for label, _ in expected_pairs]
    pred_map = {}
    for label, value in pred_pairs:
        pred_map.setdefault(normalize_text(label), value)
    matched, assoc_ok, value_ok = 0, 0, 0
    missing_labels = []
    for (label, exp_value), norm_label in zip(expected_pairs, exp_labels):
        if norm_label not in pred_map:
            missing_labels.append(label)
            continue
        matched += 1
        try:
            ok = normalize_amount(pred_map[norm_label]) == \
                normalize_amount(exp_value)
        except ValueError:
            ok = False
        value_ok += ok
        assoc_ok += ok  # association == matched label + matched value
    total = len(expected_pairs)
    return {
        "n_expected_rows": total,
        "n_matched_labels": matched,
        "label_recall": (matched / total) if total else 1.0,
        "value_accuracy": (value_ok / matched) if matched else 0.0,
        "association_accuracy": (assoc_ok / total) if total else 1.0,
        "missing_labels": missing_labels,
    }


def score_table_label_text(pred_text: str,
                            expected_labels: Sequence[str]) -> dict:
    """LAYER A for tables (reported as table_label_text_recall).

    For each expected visible row label (e.g. "Cash", "TOTAL ASSETS",
    "Debt"): credit iff its normalized form occurs as a substring of the
    normalized recovered OCR text. No table object is required — raw text
    containing "Cash ... 5,087,893.71" passes here even with tables == [].
    Structured credit (table_association_accuracy, Layer B) still requires
    a correct (label, value) pair via score_table.
    """
    hay = normalize_text(pred_text or "")
    found = [label for label in (expected_labels or [])
             if normalize_text(label) and normalize_text(label) in hay]
    missing = [label for label in (expected_labels or [])
               if not (normalize_text(label)
                       and normalize_text(label) in hay)]
    total = len(list(expected_labels or []))
    return {
        "n_expected_labels": total,
        "n_found_labels": len(found),
        "label_text_recall": (len(found) / total) if total else 1.0,
        "recall": (len(found) / total) if total else 1.0,
        "found": found,
        "missing": missing,
    }


# --------------------------------------------------------------------------
# Charts / KPIs: recognition (A) vs semantic interpretation (B)
# --------------------------------------------------------------------------
def score_chart_understanding(fields: Mapping, visual_description: str,
                              chart_truth: Mapping) -> dict:
    """Deterministic semantic scoring for trend charts (DEV-008 class).

    chart_truth: derived truth 'chart' block (x labels, series values,
    trend sentence). fields: model-extracted structured values, expected
    to carry 'peak_year' and optionally per-year values.
    """
    series = (chart_truth.get("series") or [{}])[0]
    values = [Decimal(v) for v in series.get("values", [])]
    x_labels = chart_truth.get("x", [])
    peak_year = x_labels[values.index(max(values))] if values else None
    trend_words = [w.strip(".,;:").lower()
                   for w in chart_truth.get("trend", "").split()]
    trend_keywords = [w for w in ("growth", "highest", "surge", "shift",
                                  "steady", "trend", "increase")
                      if w in trend_words]
    desc = normalize_text(visual_description or "").lower()
    out = {
        "expected_peak_year": peak_year,
        "peak_year_ok": (str(fields.get("peak_year", "")).strip()
                         == peak_year),
        "trend_keywords_expected": trend_keywords,
        "trend_keywords_found": [w for w in trend_keywords if w in desc],
    }
    out["trend_recall"] = (len(out["trend_keywords_found"])
                           / len(trend_keywords)) if trend_keywords else 1.0
    if "year_values" in fields and values:
        year_values = fields["year_values"]
        per_year = {}
        for year, exp in zip(x_labels, values):
            try:
                per_year[year] = (normalize_amount(year_values.get(year))
                                  == exp)
            except (ValueError, AttributeError):
                per_year[year] = False
        out["per_year_values"] = per_year
        out["values_accuracy"] = (sum(per_year.values()) / len(per_year)
                                  if per_year else 1.0)
    return out


def score_kpi_understanding(fields: Mapping, visual_description: str,
                            kpi_truth: Mapping) -> dict:
    """Deterministic semantic scoring for KPI dashboards (DEV-009 class).

    kpi_truth: derived truth with 'numeric_values' + 'budget_verdict'.
    Checks KPI value accuracy, branch ranking, and budget verdict +
    variance — the interpretation layer beyond raw transcription.
    """
    numeric = score_numerics(fields, kpi_truth.get("numeric_values", {}))
    branches = {k: v for k, v in kpi_truth.get("numeric_values", {}).items()
                if k.startswith("branch_revenue_")}
    expected_top = (max(branches, key=lambda k: Decimal(branches[k]))
                    if branches else None)
    got_top = fields.get("top_branch")
    verdict = kpi_truth.get("budget_verdict", {})
    desc = normalize_text(visual_description or "").lower()
    verdict_ok = (not verdict.get("verdict_text")
                  or verdict["verdict_text"].lower() in desc
                  or str(fields.get("budget_verdict", "")).strip().lower()
                  == verdict.get("verdict_text", "").lower())
    try:
        variance_ok = (normalize_amount(fields.get("variance"))
                       == normalize_amount(verdict["variance"])) \
            if verdict.get("variance") and fields.get("variance") \
            is not None else None
    except ValueError:
        variance_ok = False
    return {
        "kpi_values": numeric,
        "expected_top_branch": expected_top,
        "top_branch_ok": (got_top == expected_top),
        "expected_verdict": verdict.get("verdict_text"),
        "verdict_ok": verdict_ok,
        "variance_ok": variance_ok,
    }
