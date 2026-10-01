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

  VISUAL REASONING (DEV-008 chart / DEV-009 KPI dashboard):
    chart_label_recall, chart_numeric_exact_accuracy (displayed "M"
    values), chart_association_accuracy, chart_trend_semantic_accuracy,
    kpi_label_recall, kpi_numeric_exact_accuracy, kpi_association_accuracy,
    budget_status_accuracy, arabic/english anchor recall, hallucination
    counts, unsupported-claim flags, latency_ms.
    Every visual metric is reported INDEPENDENTLY — never merged or
    averaged into one overall score. Faithfulness first; latency stays
    a separate reported number.

Conventions (see docs/OCR_VISION_DESIGN.md §6, EVALUATION_PLAN §3):
  - identifiers: exact after outer-whitespace strip (hyphens/case matter).
  - numerics: Decimal-2dp equality; every miss listed expected-vs-got.
  - anchors: normalized-substring recall.
  - CER/WER: Levenshtein-based; reported raw AND normalized.
  - tables: (label, value) association scoring — never plain-text only.
  - charts/KPIs: recognition (A) and semantic interpretation (B) scored
    separately and deterministically (no LLM judge).
  - visual displays: on-chart/dashboard strings ("51.9M", "23.90M") are
    DISPLAY values, distinct from canonical exact sums ("51855389.81").
    A faithful vision model reports what is DISPLAYED; exact canonical
    sums are NOT visible on DEV-008/009 (M-abbreviated) except the
    DEV-009 variance panel, which prints the exact figure.
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


# --------------------------------------------------------------------------
# Visual reasoning (DEV-008 chart / DEV-009 KPI): independent metrics.
#
# Every function below reports ONE independent slice — recognition,
# association, trend semantics, hallucination, language split. Callers
# must NEVER average them into a single "overall" number. Accuracy /
# faithfulness comes first; latency_ms is reported separately by the
# result schema, never mixed into correctness.
#
# Display vs canonical: DEV-008 value labels read "51.9M" (1dp millions)
# and DEV-009 cards/bars read "23.90M"-style (2dp millions); the frozen
# exact sums ("51855389.81", "23902434.34") are NOT printed on those
# visuals (only the DEV-009 variance panel prints its exact figure).
# Display metrics therefore compare DISPLAY strings; canonical exact
# metrics keep their existing exact-Decimal semantics elsewhere.
# --------------------------------------------------------------------------
_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")

# Causal-language markers. Presence is a flag for HUMAN review, never an
# automatic penalty: the scorer cannot know from keywords alone whether a
# causal claim is evidence-supported, so it reports the flag
# independently (see flag_unsupported_claims).
_CAUSAL_MARKERS = ("because", "due to", "caused by", "led to",
                   "resulted in", "triggered by", "thanks to")

# Amount-like candidates including an optional millions/percent suffix and
# an explicit sign. Year labels ("2015") also match; callers put expected
# year strings into the expected set so faithful years never count as
# hallucinations. NOTE: a plain ASCII space is deliberately NOT an
# intra-number character here (unlike the legacy amount regex): "2016
# 55.4M" must yield two candidates ("2016", "55.4M"), never one merged
# "201655.4M". NBSP/Arabic-thousands grouping still supported.
_NUM_CANDIDATE_RE = re.compile(
    r"[+-]?[0-9\u0660-\u0669\u06F0-\u06F9]"
    r"[0-9\u0660-\u0669\u06F0-\u06F9,.\u066B\u066C\u00A0]*"
    r"[0-9\u0660-\u0669\u06F0-\u06F9]\s*[Mm%]?"
    r"|[+-]?[0-9\u0660-\u0669\u06F0-\u06F9]\s*[Mm%]?")

_CURRENCY_SUFFIX_RE = re.compile(r"(SAR|sar|SAR\.|\u0631\.\u0633"
                                 r"|\u0631\s*\u0633)", re.IGNORECASE)


def _norm_num_token(token: str) -> str:
    """Normalize one numeric candidate for set comparison.

    Folds digits/whitespace via normalize_text, strips grouping
    separators, unifies a trailing "m" to "M". Digit content stays
    significant: "28026.95" never equals "28026.94".
    """
    out = normalize_text(token or "")
    out = out.replace(",", "").replace(" ", "").replace("\u00A0", "")
    if out.endswith("m"):
        out = out[:-1] + "M"
    return out


def _norm_display_value(value) -> str:
    """Normalize one DISPLAY value ("23.90M SAR" -> "23.90M").

    Strips currency tokens, grouping separators, and whitespace; unifies
    trailing "m" to "M". Digits and "%" stay significant.
    """
    if value is None:
        return ""
    out = normalize_text(str(value))
    out = _CURRENCY_SUFFIX_RE.sub("", out)
    out = out.replace(",", "").replace(" ", "").replace("\u00A0", "")
    if out.endswith("m"):
        out = out[:-1] + "M"
    return out


def numeric_candidates_in_text(pred_text: str) -> list:
    """Amount-like candidate strings in reading order, deduplicated.

    Includes M-suffixed ("51.9M"), percents ("0.22%"), and signed figures
    ("+52,904.62"). Raw strings only — no validity judgment here.
    """
    seen, ordered = set(), []
    for match in _NUM_CANDIDATE_RE.finditer(pred_text or ""):
        raw = match.group(0).strip()
        if not raw or raw in seen:
            continue
        seen.add(raw)
        ordered.append(raw)
    return ordered


def count_hallucinated_numerics(pred_text: str,
                                expected_strings: Sequence[str]) -> dict:
    """Count numeric candidates with no expected counterpart.

    Both sides go through _norm_num_token, so "24,371.25" matches
    "24371.25" (grouping-insensitive) while "28026.95" never matches
    "28026.94" and "99.9M" never matches "51.9M". Callers put every
    legitimately visible number (display values, exact panel figures,
    year labels) into expected_strings; anything else counts.
    """
    expected = {_norm_num_token(e) for e in (expected_strings or [])
                if _norm_num_token(e)}
    cands = numeric_candidates_in_text(pred_text)
    hallucinated = [c for c in cands
                    if _norm_num_token(c) not in expected]
    return {
        "n_candidates": len(cands),
        "n_expected": len(expected),
        "hallucinated": hallucinated,
        "hallucinated_numeric_count": len(hallucinated),
    }


def count_hallucinated_labels(pred_labels: Sequence[str],
                              expected_labels: Sequence[str]) -> dict:
    """Count structured labels with no expected counterpart.

    Compares normalized exact forms. pred_labels are caller-extracted
    STRUCTURED labels (fields keys/values, table first-cells) — raw prose
    cannot yield reliable label inventories, so text is never scanned
    here. Returns the count plus the offending list for review.
    """
    expected = {normalize_text(e) for e in (expected_labels or [])
                if normalize_text(e)}
    seen, hallucin = set(), []
    for label in (pred_labels or []):
        norm = normalize_text(str(label))
        if not norm or norm in seen:
            continue
        seen.add(norm)
        if norm not in expected:
            hallucin.append(str(label))
    return {
        "n_pred_labels": len(seen),
        "n_expected_labels": len(expected),
        "hallucinated": hallucin,
        "hallucinated_label_count": len(hallucin),
    }


def split_anchors_by_script(anchors: Sequence[str]) -> dict:
    """Split required anchors into Arabic vs English/other lists.

    An anchor is Arabic iff it contains any U+0600–U+06FF character;
    everything else (English, digits, neutral) goes to English. Purely a
    script split for independent recall reporting.
    """
    arabic = [a for a in (anchors or []) if _ARABIC_RE.search(a or "")]
    english = [a for a in (anchors or []) if not _ARABIC_RE.search(a or "")]
    return {"arabic": arabic, "english": english}


def _score_label_list(pred_text: str,
                      expected_labels: Sequence[str]) -> dict:
    """Shared normalized-substring label recall core."""
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
        "recall": (len(found) / total) if total else 1.0,
        "found": found,
        "missing": missing,
    }


def score_chart_labels(pred_text: str,
                       expected_labels: Sequence[str]) -> dict:
    """LAYER A for charts: title/year/series label recall in raw text.

    Reports chart_label_recall. No table/field object required.
    """
    scored = _score_label_list(pred_text, expected_labels)
    scored["label_recall"] = scored["recall"]
    scored["chart_label_recall"] = scored["recall"]
    return scored


def score_kpi_labels(pred_text: str,
                     expected_labels: Sequence[str]) -> dict:
    """LAYER A for KPI dashboards: card/bar/panel label recall in text.

    Reports kpi_label_recall. Association still needs score pairing.
    """
    scored = _score_label_list(pred_text, expected_labels)
    scored["label_recall"] = scored["recall"]
    scored["kpi_label_recall"] = scored["recall"]
    return scored


def display_millions(amount_str, decimals: int = 1) -> str:
    """Render a canonical amount as an on-visual millions label.

    display_millions("51855389.81") == "51.9M" (decimals=1, DEV-008);
    display_millions("23902434.34", 2) == "23.90M" (DEV-009 cards/bars
    before the " SAR" suffix, which callers add when needed). Decimal
    half-even rounding; deterministic. Raises ValueError when unparseable.
    """
    base = normalize_amount(amount_str)
    quantum = Decimal("1." + "0" * decimals) if decimals else Decimal("1")
    scaled = (base / Decimal(1000000)).quantize(quantum)
    return f"{scaled:.{decimals}f}M" if decimals else f"{scaled:.0f}M"


def expected_chart_display_values(chart_truth: Mapping,
                                  decimals: int = 1) -> list:
    """On-chart M labels for a chart truth block, in x order.

    Derived deterministically from the frozen series values (1dp millions
    per the DEV-008 renderer); the frozen truth itself is untouched.
    """
    series = (chart_truth.get("series") or [{}])[0]
    return [display_millions(v, decimals)
            for v in series.get("values", [])]


def expected_chart_labels(chart_truth: Mapping,
                          text_anchors: Sequence[str]) -> list:
    """Visible chart labels: title anchors + year labels + series label."""
    labels = list(text_anchors or [])
    labels.extend(chart_truth.get("x", []) or [])
    series = (chart_truth.get("series") or [{}])[0]
    if series.get("label"):
        labels.append(series["label"])
    return labels


def score_chart_display_text(pred_text: str,
                             expected_display: Sequence[str]) -> dict:
    """LAYER A for chart numerics: displayed M-label recall in raw text.

    Reports chart_numeric_exact_accuracy. "51.9M" in text matches only
    "51.9M" (a digit substitution like "51.8M" fails); grouping inside
    the millions figure is insignificant only where _norm_num_token says
    so. Canonical exact sums are NOT expected here — they are not printed
    on the visual.
    """
    hay = {_norm_num_token(c) for c in numeric_candidates_in_text(pred_text)}
    per_value, misses = {}, []
    for disp in (expected_display or []):
        ok = _norm_num_token(disp) in hay
        per_value[disp] = ok
        if not ok:
            misses.append({"value": disp, "found_in_text": False})
    total = len(per_value)
    correct = sum(1 for v in per_value.values() if v)
    return {
        "per_value": per_value,
        "n_total": total,
        "n_correct": correct,
        "accuracy": (correct / total) if total else 1.0,
        "chart_numeric_exact_accuracy": (correct / total) if total else 1.0,
        "misses": misses,
    }


def score_kpi_display_text(pred_text: str,
                           expected_display: Sequence[str]) -> dict:
    """LAYER A for KPI numerics: displayed M-label recall in raw text.

    Reports kpi_numeric_exact_accuracy with the same exactness semantics
    as the chart twin. The exact variance panel figure ("+52,904.62")
    belongs in expected_display as well when the caller scores DEV-009.
    """
    hay = {_norm_num_token(c) for c in numeric_candidates_in_text(pred_text)}
    per_value, misses = {}, []
    for disp in (expected_display or []):
        ok = _norm_num_token(disp) in hay
        per_value[disp] = ok
        if not ok:
            misses.append({"value": disp, "found_in_text": False})
    total = len(per_value)
    correct = sum(1 for v in per_value.values() if v)
    return {
        "per_value": per_value,
        "n_total": total,
        "n_correct": correct,
        "accuracy": (correct / total) if total else 1.0,
        "kpi_numeric_exact_accuracy": (correct / total) if total else 1.0,
        "misses": misses,
    }


def pairs_from_display_grid(tables: Sequence[Sequence[Sequence[str]]],
                            expected_labels: Sequence[str]) -> list:
    """Adapt recovered grids to (label, DISPLAY-value) pairs.

    For every expected label, takes the FIRST row (across tables, in
    order) whose joined normalized text contains the normalized label.
    The row must have at least TWO cells; the value is the LAST cell
    (stripped). Single-cell rows yield no pair. Deterministic; display
    values are compared as strings downstream (no amount parsing, so
    "51.9M" and "23.90M SAR" survive intact).
    """
    pairs = []
    for label in expected_labels:
        needle = normalize_text(label)
        for table in tables or []:
            matched = False
            for row in table or []:
                cells = [str(c) for c in row]
                if len(cells) < 2:
                    continue
                if needle and needle in normalize_text(" ".join(cells)):
                    value = cells[-1].strip()
                    if value:
                        pairs.append((label, value))
                        matched = True
                        break
                if matched:
                    break
            if matched:
                break
    return pairs


def pairs_from_fields_map(mapping: Mapping) -> list:
    """Adapt a structured {label: value} mapping to (label, value) pairs.

    Used for chart year_values / KPI field dicts. Stringifies both sides,
    skips empty labels/values, preserves caller order. Deterministic.
    """
    pairs = []
    for key, value in (mapping or {}).items():
        if key is None or value is None:
            continue
        label, val = str(key).strip(), str(value).strip()
        if label and val:
            pairs.append((label, val))
    return pairs


def score_display_association(pred_pairs: Sequence[tuple],
                              expected_pairs: Sequence[tuple]) -> dict:
    """Score DISPLAY-value association as (label, value) pairs.

    A pair counts correct ONLY when the normalized label matches AND the
    normalized DISPLAY value matches (_norm_display_value: "23.90M SAR"
    == "23.90M", but "51.8M" never equals "51.9M"). Right value on the
    wrong label is incorrect — exactly the failure mode Layer B exists
    to catch. Reports label recall, value accuracy (on matched labels),
    and association accuracy separately.
    """
    exp_norm = [(normalize_text(label), _norm_display_value(value))
                for label, value in expected_pairs]
    pred_map = {}
    for label, value in pred_pairs:
        pred_map.setdefault(normalize_text(label),
                            _norm_display_value(value))
    matched, assoc_ok, value_ok = 0, 0, 0
    missing_labels = []
    for (label, _), (norm_label, norm_exp) in zip(expected_pairs, exp_norm):
        if norm_label not in pred_map:
            missing_labels.append(label)
            continue
        matched += 1
        ok = pred_map[norm_label] == norm_exp
        value_ok += ok
        assoc_ok += ok
    total = len(expected_pairs)
    return {
        "n_expected": total,
        "n_matched_labels": matched,
        "label_recall": (matched / total) if total else 1.0,
        "value_accuracy": (value_ok / matched) if matched else 0.0,
        "association_accuracy": (assoc_ok / total) if total else 1.0,
        "missing_labels": missing_labels,
    }


def score_trend_semantics(visual_description: str, fields: Mapping,
                          chart_truth: Mapping) -> dict:
    """Trend semantics for charts: peak/lowest years + keyword recall.

    - peak_year_ok / lowest_year_ok compare fields values to the frozen
      series extrema. A missing key yields None (not a penalty: refusing
      to infer beyond the visual is allowed).
    - trend keyword recall reuses the deterministic keyword rule (words
      shared by the frozen trend sentence and the fixed vocabulary).
    - causal_language_flag marks because/due-to style wording for HUMAN
      review; it is reported independently, never subtracted.
    - chart_trend_semantic_accuracy mirrors trend recall (the only
      scalar), so callers have one named number without hiding parts.
    """
    series = (chart_truth.get("series") or [{}])[0]
    raw_values = series.get("values", [])
    values = [Decimal(v) for v in raw_values]
    x_labels = chart_truth.get("x", [])
    peak = x_labels[values.index(max(values))] if values else None
    lowest = x_labels[values.index(min(values))] if values else None
    trend_words = [w.strip(".,;:").lower()
                   for w in chart_truth.get("trend", "").split()]
    keywords = [w for w in ("growth", "highest", "surge", "shift",
                            "steady", "trend", "increase")
                if w in trend_words]
    desc = normalize_text(visual_description or "").lower()
    found = [w for w in keywords if w in desc]
    fields = fields or {}
    peak_ok = (str(fields.get("peak_year", "")).strip() == peak) \
        if "peak_year" in fields else None
    lowest_ok = (str(fields.get("lowest_year", "")).strip() == lowest) \
        if "lowest_year" in fields else None
    causal = [m for m in _CAUSAL_MARKERS if m in desc]
    recall = (len(found) / len(keywords)) if keywords else 1.0
    return {
        "expected_peak_year": peak,
        "peak_year_ok": peak_ok,
        "expected_lowest_year": lowest,
        "lowest_year_ok": lowest_ok,
        "trend_keywords_expected": keywords,
        "trend_keywords_found": found,
        "trend_recall": recall,
        "chart_trend_semantic_accuracy": recall,
        "causal_language_flag": bool(causal),
        "causal_markers_found": causal,
    }


def flag_unsupported_claims(visual_description: str,
                            supported_keywords: Sequence[str]) -> dict:
    """Flag causal sentences with no supported keyword (human-review aid).

    Splits the description into sentences; a sentence counts when it
    contains a causal marker but NONE of the supported keywords (frozen
    trend words, year labels, series terms the caller supplies).
    Deterministic heuristic — reported independently, never a score
    penalty on its own.
    """
    import re as _re
    supported = [s.lower() for s in (supported_keywords or []) if s]
    sentences = [s.strip() for s in
                 _re.split(r"[.!?\n]+", visual_description or "")
                 if s.strip()]
    flagged = [s for s in sentences
               if any(m in s.lower() for m in _CAUSAL_MARKERS)
               and not any(k and k in s.lower() for k in supported)]
    return {
        "n_sentences": len(sentences),
        "unsupported_claim_count": len(flagged),
        "flagged": flagged,
    }


def score_budget_status(pred_text: str, visual_description: str,
                        actual_str, budget_str, kind: str = "revenue") -> dict:
    """Visible budget-status accuracy (DEV-009 class).

    The expected statement follows the frozen directional rule:
    revenue targets are met iff actual >= budget; expense targets iff
    actual <= budget. Accuracy is 1.0 ONLY when the expected visible
    statement ("Budget met" / "Budget missed") occurs in the model's
    text or visual description. A correct independent recomputation that
    is never quoted earns NO credit here (by design — the experiment
    tests what the dashboard visibly states). Unparseable figures yield
    accuracy 0.0 with a reason, never an exception.
    """
    try:
        actual = normalize_amount(actual_str)
        budget = normalize_amount(budget_str)
    except ValueError:
        return {
            "expected": None,
            "found_in_visual": False,
            "accuracy": 0.0,
            "budget_status_accuracy": 0.0,
            "reason": "unparseable actual/budget figures",
        }
    if kind == "expense":
        expected = "Budget met" if actual <= budget else "Budget missed"
    else:
        expected = "Budget met" if actual >= budget else "Budget missed"
    hay = normalize_text(
        f"{pred_text or ''} {visual_description or ''}").lower()
    present = expected.lower() in hay
    return {
        "expected": expected,
        "kind": kind,
        "actual": str(actual),
        "budget": str(budget),
        "found_in_visual": present,
        "accuracy": 1.0 if present else 0.0,
        "budget_status_accuracy": 1.0 if present else 0.0,
        "note": ("visible quote required; recomputation alone earns "
                 "no credit"),
    }
